"""Offline outcomes kept separate from feature eligibility and scoring inputs.

Modern source vintage, adjustment, opening execution and terminal-event limits
remain unresolved. Clock tests inspect 2015/2017 marks solely for availability
and accounting checks, consistent with the research protocol's exposure note;
they are not strategy selection or an untouched historical preregistration.
"""

import math
import re
from dataclasses import dataclass
from datetime import date, datetime

from quantlab.data.archive import Bar, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.returns import monthly_next_open_label

TARGET_ID = "monthly_next_scheduled_open_to_following_month_next_scheduled_open_v1"


@dataclass(frozen=True)
class MonthlyLabel:
    asset_id: str
    signal_session: date
    signal_at: datetime
    entry_session: date
    entry_at: datetime
    exit_session: date
    exit_at: datetime
    planned_available_at: datetime
    label_available_at: datetime | None
    entry_mark: float | None
    exit_mark: float | None
    return_value: float | None
    status: str
    source_status: str
    reasons: tuple[str, ...]
    entry_row: Bar | None
    exit_row: Bar | None
    entry_row_id: tuple[str, int, str] | None
    exit_row_id: tuple[str, int, str] | None
    quality_flags: tuple[str, ...]
    archive_endpoint: date
    archive_sha256: str
    calendar_sha256: str
    target_id: str = TARGET_ID
    units: str = "decimal_simple_opening_price_return"
    price_basis: str = "source_OHLC_adjustment_unverified"
    usage: str = "offline_outcome_never_feature_eligibility_input"


@dataclass(frozen=True)
class LabelExclusion:
    asset_id: str
    signal_session: date
    reason: str
    label_status: str
    label_available_at: datetime | None


@dataclass(frozen=True)
class LabelSelection:
    accepted: tuple[MonthlyLabel, ...]
    exclusions: tuple[LabelExclusion, ...]
    information_cutoff: datetime
    signal_start: date | None
    signal_end: date | None
    purpose: str


def _aware(value):
    return (isinstance(value, datetime) and value.tzinfo is not None
            and value.utcoffset() is not None)


def label_from_history(asset_id, bars, schedule, *, signal_session, archive_endpoint,
                       archive_sha256):
    """Use exact scheduled marks; never choose a later observed entry or exit.

    The caller authenticates low-level histories or a snapshot adapter. The
    archive endpoint is global snapshot coverage, never the asset's last row.
    Flags follow the shared returns helper conservatively, including flags not
    related directly to the price field. No missing return is inferred.
    """
    if not isinstance(schedule, XNYSSchedule):
        raise ValueError("Actual XNYSSchedule required")
    if (not isinstance(archive_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", archive_sha256)):
        raise ValueError("Authenticated archive SHA256 required")
    if not isinstance(asset_id, str) or not re.fullmatch(
        r"stock:[a-z0-9][a-z0-9._^-]*\.us", asset_id
    ):
        raise ValueError("Original source-scoped stock identity required")
    if type(archive_endpoint) is not date:
        raise ValueError("Declared global snapshot endpoint date required")
    timing = schedule.month_end(signal_session.year, signal_session.month)
    if timing.session != signal_session:
        raise ValueError("Actual monthly final signal session required")
    month = signal_session.month
    next_year, next_month = ((signal_session.year + 1, 1) if month == 12
                             else (signal_session.year, month + 1))
    next_signal = schedule.month_end(next_year, next_month)
    entry = schedule.next_session(signal_session)
    exit_timing = schedule.next_session(next_signal.session)
    bars = tuple(bars)
    if any(not isinstance(b, Bar) or b.asset_id != asset_id for b in bars):
        raise ValueError("Identity-aligned actual source Bar records required")
    if tuple(b.session for b in bars) != tuple(sorted({b.session for b in bars})):
        raise ValueError("Ordered unique source sessions required")
    if any(b.session > archive_endpoint for b in bars):
        raise ValueError("Source rows exceed declared global snapshot endpoint")
    indexed = {b.session: b for b in bars}
    left, right = indexed.get(entry.session), indexed.get(exit_timing.session)
    marks = tuple(b for b in (left, right) if b is not None)
    if len({(b.source_member, b.source_sha256) for b in marks}) > 1:
        raise ValueError("Conflicting source identity/vintage at label marks")
    if any(not re.fullmatch(r"[0-9a-f]{64}", b.source_sha256)
           or not re.fullmatch(r"[0-9a-f]{64}", b.row_sha256) for b in marks):
        raise ValueError("Source member and row SHA256 provenance required")
    if bars:
        interval = monthly_next_open_label(
            bars, schedule, signal_session.year, month, archive_endpoint=archive_endpoint
        )
        source_status, value = interval.status, interval.return_value
        flags = interval.quality_flags
        left_id, right_id = interval.start_row, interval.end_row
    else:
        source_status = ("right_censored" if exit_timing.session > archive_endpoint
                         else "missing_mark")
        value, flags, left_id, right_id = None, (), None, None
    status = source_status
    if source_status == "missing_mark":
        status = ("missing_entry_and_exit" if left is None and right is None else
                  "missing_entry" if left is None else "missing_exit")
    reasons = () if status == "valid" else (status.upper(),)
    if status == "right_censored":
        reasons += ("PLANNED_EXIT_AFTER_GLOBAL_ARCHIVE_ENDPOINT",)
    if left is None:
        reasons += ("NO_EXACT_SCHEDULED_ENTRY_ROW",)
    if right is None:
        reasons += ("NO_EXACT_SCHEDULED_EXIT_ROW",)
    available = exit_timing.decision_at if status == "valid" else None
    return MonthlyLabel(
        asset_id, signal_session, timing.decision_at, entry.session, entry.open_at,
        exit_timing.session, exit_timing.open_at, exit_timing.decision_at, available,
        left.open if left else None, right.open if right else None, value, status,
        source_status, reasons, left, right, left_id, right_id, flags, archive_endpoint,
        archive_sha256, schedule.provenance["schedule_sha256"],
    )


def read_labels(source, symbols, schedule, *, signal_sessions, archive_endpoint, archive_sha256):
    """Read once per symbol from an authenticated archive or explicit verified loader.

    Callable snapshot adapters implement (asset_type, symbol)->tuple[Bar,...]
    and are authenticated by their caller before this function is invoked.
    """
    if isinstance(source, MarketArchive):
        if source.sha256 != archive_sha256:
            raise ValueError("Archive handle differs from declared SHA256")
        loader = source.bars
    elif callable(source):
        loader = source
    else:
        raise ValueError("Authenticated MarketArchive or explicit verified loader required")
    signals = tuple(signal_sessions)
    if len(set(signals)) != len(signals):
        raise ValueError("Unique signal sessions required")
    records, assets = [], set()
    for symbol in symbols:
        if not isinstance(symbol, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._^-]*", symbol):
            raise ValueError("Source symbol required")
        canonical = symbol.lower().removesuffix(".us")
        asset = "stock:" + canonical + ".us"
        if asset in assets:
            raise ValueError("Duplicate source identity")
        assets.add(asset)
        bars = tuple(loader("stock", canonical))
        records.extend(label_from_history(
            asset, bars, schedule, signal_session=signal, archive_endpoint=archive_endpoint,
            archive_sha256=archive_sha256
        ) for signal in signals)
    return tuple(sorted(records, key=lambda r: (r.signal_session, r.asset_id)))


def _select(records, fit_at, signal_start, signal_end, purpose):
    if not _aware(fit_at):
        raise ValueError("Timezone-aware information cutoff required")
    if (signal_start is not None and type(signal_start) is not date
            or signal_end is not None and type(signal_end) is not date
            or signal_start is not None and signal_end is not None and signal_start > signal_end):
        raise ValueError("Ordered signal-date scope required")
    records = tuple(records)
    if any(not isinstance(r, MonthlyLabel) for r in records):
        raise ValueError("MonthlyLabel records required")
    keys = [(r.asset_id, r.signal_session, r.target_id) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate training label key")
    contracts = {(r.archive_sha256, r.calendar_sha256, r.target_id, r.units, r.price_basis)
                 for r in records}
    if len(contracts) > 1:
        raise ValueError("One source/calendar/target contract required for training labels")
    accepted, excluded = [], []
    for row in sorted(records, key=lambda r: (r.signal_session, r.asset_id)):
        reason = None
        if ((signal_start is not None and row.signal_session < signal_start)
                or (signal_end is not None and row.signal_session > signal_end)):
            reason = "OUTSIDE_SIGNAL_SCOPE"
        elif row.status != "valid":
            reason = "UNRESOLVED_LABEL"
        elif (not _aware(row.label_available_at) or row.return_value is None
              or not math.isfinite(row.return_value)
              or not row.signal_at < row.entry_at < row.exit_at <= row.label_available_at):
            raise ValueError("Resolved label clock/value contract failed")
        elif row.label_available_at > fit_at:
            reason = "IMMATURE_LABEL"
        if reason:
            excluded.append(LabelExclusion(row.asset_id, row.signal_session, reason,
                                           row.status, row.label_available_at))
        else:
            accepted.append(row)
    return LabelSelection(tuple(accepted), tuple(excluded), fit_at,
                          signal_start, signal_end, purpose)


def mature_training_rows(records, fit_at):
    """Select only resolved outcomes known at fit_at; retain exclusion evidence."""
    return _select(records, fit_at, None, None, "training_maturity_only")


def development_selection_rows(records, *, information_cutoff, signal_start, signal_end):
    """Require explicit signal-date scope AND information cutoff for selection.

    An old signal date alone cannot admit an outcome spilling across the cutoff.
    This function does not choose development/holdout boundaries for the caller.
    """
    if signal_start is None or signal_end is None:
        raise ValueError("Explicit development signal-date scope required")
    return _select(records, information_cutoff, signal_start, signal_end,
                   "development_selection")
