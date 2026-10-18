"""Six required price/activity features under the declared source convention.

There is no historical constituent membership or verified monetary liquidity in
this archive. close*volume is a source-value activity proxy; retrospectively
adjusted observations are not an audited point-in-time release vintage.
"""

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from statistics import median, stdev

from quantlab.data.archive import Bar, MarketArchive
from quantlab.data.calendar import XNYSSchedule

FEATURE_IDS = ("F01", "F07", "F09", "F13", "F15", "F19")
FEATURE_UNITS = (
    ("F01", "decimal_price_ratio_minus_one"),
    ("F07", "negative_decimal_price_ratio_minus_one"),
    ("F09", "annualised_log_return_standard_deviation_252"),
    ("F13", "decimal_price_ratio_minus_one"),
    ("F15", "decimal_price_ratio_minus_one"),
    ("F19", "log1p_source_value_proxy_divided_by_one_source_value_unit"),
)
FORMULA_VERSION = "blueprint_v3_required_six_v1"
ACTIVITY_CONVENTION = "median_63_source_close_times_volume_unverified_joint_adjustment"
PRICE_BASIS = "provider_claimed_split_dividend_adjusted_not_independently_verified"


@dataclass(frozen=True)
class FeatureCandidate:
    asset_id: str
    signal_session: date
    signal_at: datetime
    intended_execution_at: datetime
    eligible: bool
    reasons: tuple[str, ...]
    features: tuple[tuple[str, float], ...]
    activity_proxy: float | None
    observed_scheduled_marks: int
    required_scheduled_marks: int
    max_input_available_at: datetime | None
    source_rows: tuple[Bar, ...]
    quality_findings: tuple[tuple[date, tuple[str, ...]], ...]
    archive_sha256: str
    calendar_sha256: str
    formula_version: str = FORMULA_VERSION
    activity_convention: str = ACTIVITY_CONVENTION
    price_basis: str = PRICE_BASIS
    feature_units: tuple[tuple[str, str], ...] = FEATURE_UNITS


@dataclass(frozen=True)
class RankedFeatureRow:
    asset_id: str
    liquidity_rank: int
    activity_proxy: float
    raw: tuple[tuple[str, float], ...]
    normalized: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class CrossSection:
    signal_session: date
    signal_at: datetime
    candidates: tuple[FeatureCandidate, ...]
    selected_asset_ids: tuple[str, ...]
    rows: tuple[RankedFeatureRow, ...]
    selection_reasons: tuple[tuple[str, str], ...]
    top_n: int
    feature_ids: tuple[str, ...] = FEATURE_IDS
    ranking_rule: str = "descending_median63_activity_then_ascending_source_asset_id"
    transformation_rule: str = "within_selected_date_average_rank_centered_singleton_zero"
    normalized_units: str = "dimensionless_centered_cross_sectional_rank"


def _provenance(archive_sha256, schedule):
    if not isinstance(archive_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise ValueError("Declared authenticated source archive SHA256 required")
    if not isinstance(schedule, XNYSSchedule):
        raise ValueError("Actual XNYSSchedule required")


def candidate_from_history(asset_id, bars, schedule, *, as_of, archive_sha256):
    """Calculate one monthly candidate from trusted real source rows.

    This low-level function does not authenticate an archive. It materializes
    only rows dated <=as_of before checking history or calculating features;
    no future endpoint, survival, target or membership condition is inspected.
    Invalid required support produces an excluded candidate, never imputation.
    """
    _provenance(archive_sha256, schedule)
    if not isinstance(asset_id, str) or not re.fullmatch(
        r"stock:[a-z0-9][a-z0-9._^-]*\.us", asset_id
    ):
        raise ValueError("Source-scoped stock asset identity required")
    timing = schedule.session(as_of)
    if schedule.month_end(as_of.year, as_of.month).session != as_of:
        raise ValueError("Final actual exchange session of a complete month required")
    next_open = schedule.next_session(as_of).open_at
    # Source date is the archive's daily label; only after-close availability is assumed.
    past = tuple(b for b in bars if b.session <= as_of)
    if any(not isinstance(b, Bar) or b.asset_id != asset_id for b in past):
        raise ValueError("Identity-aligned source Bar records required")
    if tuple(b.session for b in past) != tuple(sorted({b.session for b in past})):
        raise ValueError("Ordered unique historical asset sessions required")
    index = schedule.sessions.index(as_of)
    reasons = []
    if index < 252:
        reasons.append("INSUFFICIENT_CALENDAR_HISTORY")
    days = schedule.sessions[max(0, index - 252):index + 1]
    indexed = {b.session: b for b in past}
    rows = tuple(indexed[d] for d in days if d in indexed)
    if as_of not in indexed:
        reasons.append("MISSING_LATEST_MARK")
    if len(rows) != 253:
        reasons.append("INSUFFICIENT_EXACT_SCHEDULED_HISTORY")
    if rows and len({(b.source_member, b.source_sha256) for b in rows}) != 1:
        raise ValueError("Conflicting source identities/vintages in historical support")
    if any(not re.fullmatch(r"[0-9a-f]{64}", b.source_sha256)
           or not re.fullmatch(r"[0-9a-f]{64}", b.row_sha256) for b in rows):
        raise ValueError("Source member and row SHA256 provenance required")
    allowed_diagnostics = {"ZERO_VOLUME", "FRACTIONAL_VOLUME", "NONZERO_OPEN_INTEREST"}
    if any(set(b.quality_flags) - allowed_diagnostics or not math.isfinite(b.close) or b.close <= 0
           for b in rows):
        reasons.append("UNUSABLE_REQUIRED_PRICE")
    recent = rows[-63:] if len(rows) == 253 else ()
    if any(not math.isfinite(b.volume) or b.volume < 0 for b in recent):
        reasons.append("UNUSABLE_REQUIRED_VOLUME")
    features, activity = (), None
    if not reasons:
        closes = tuple(b.close for b in rows)
        logs = tuple(math.log(b / a) for a, b in zip(closes[-64:-1], closes[-63:], strict=True))
        activity = float(median(b.close * b.volume for b in recent))
        features = tuple(zip(FEATURE_IDS, (
            closes[-22] / closes[0] - 1,
            -(closes[-1] / closes[-22] - 1),
            math.sqrt(252) * stdev(logs),
            closes[-1] / (math.fsum(closes[-63:]) / 63) - 1,
            closes[-1] / max(closes[-252:]) - 1,
            math.log1p(activity),
        ), strict=True))
        if not all(math.isfinite(v) for _, v in features):
            reasons.append("NONFINITE_REQUIRED_FEATURE")
            features, activity = (), None
    findings = tuple((b.session, b.quality_flags) for b in rows if b.quality_flags)
    available = max((schedule.session(b.session).decision_at for b in rows), default=None)
    return FeatureCandidate(
        asset_id, as_of, timing.decision_at, next_open, not reasons, tuple(reasons),
        features, activity, len(rows), 253, available, rows, findings,
        archive_sha256, schedule.provenance["schedule_sha256"],
    )


def build_candidates(histories, schedule, *, as_of, archive_sha256):
    """Preserve every supplied candidate, including empty/as-yet-unobserved sources."""
    return tuple(candidate_from_history(asset, bars, schedule, as_of=as_of,
                                        archive_sha256=archive_sha256)
                 for asset, bars in sorted(histories.items()))


def read_candidates(source, symbols, schedule, *, as_of, archive_sha256):
    """Read an authenticated archive or an explicitly supplied verified adapter.

    Callable adapters must implement (asset_type, symbol)->tuple[Bar, ...].
    Their snapshot/manifest authentication is the caller's responsibility.
    """
    if isinstance(source, MarketArchive):
        if source.sha256 != archive_sha256:
            raise ValueError("Archive handle differs from declared SHA256")
        loader = source.bars
    elif callable(source):
        loader = source
    else:
        raise ValueError("Authenticated MarketArchive or explicit verified loader required")
    histories = {}
    for symbol in symbols:
        if not isinstance(symbol, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._^-]*", symbol):
            raise ValueError("Source symbol required")
        canonical = symbol.lower().removesuffix(".us")
        asset = "stock:" + canonical + ".us"
        if asset in histories:
            raise ValueError("Duplicate requested source identity")
        histories[asset] = tuple(loader("stock", canonical))
    return build_candidates(histories, schedule, as_of=as_of, archive_sha256=archive_sha256)


def select_cross_section(candidates, *, top_n=500):
    """Top activity candidates, then average-rank transforms of all six features.

    For n>1 use (average_rank-.5)/n-.5; a singleton receives exactly zero.
    Equal activity is broken by source asset ID; feature ties retain equal ranks.
    """
    candidates = tuple(sorted(candidates, key=lambda c: c.asset_id))
    if not candidates or isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("Nonempty candidates and positive integer top_n required")
    if len({c.asset_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identity")
    identity = {(c.signal_session, c.signal_at, c.intended_execution_at, c.archive_sha256,
                 c.calendar_sha256, c.formula_version, c.activity_convention) for c in candidates}
    if len(identity) != 1:
        raise ValueError("One date, source, calendar and feature convention required")
    ready = []
    for c in candidates:
        if c.eligible:
            if (c.reasons or tuple(k for k, _ in c.features) != FEATURE_IDS
                    or not all(math.isfinite(v) for _, v in c.features)
                    or c.activity_proxy is None or not math.isfinite(c.activity_proxy)
                    or c.activity_proxy < 0 or c.max_input_available_at is None
                    or c.max_input_available_at > c.signal_at):
                raise ValueError("Malformed eligible candidate")
            ready.append(c)
    ready.sort(key=lambda c: (-c.activity_proxy, c.asset_id))
    selected = ready[:top_n]
    n = len(selected)
    transformed = {c.asset_id: {} for c in selected}
    for feature in FEATURE_IDS:
        ordered = sorted(selected, key=lambda c: (dict(c.features)[feature], c.asset_id))
        left = 0
        while left < n:
            right = left + 1
            value = dict(ordered[left].features)[feature]
            while right < n and dict(ordered[right].features)[feature] == value:
                right += 1
            rank = (left + 1 + right) / 2
            normalized = (rank - 0.5) / n - 0.5 if n > 1 else 0.0
            for c in ordered[left:right]:
                transformed[c.asset_id][feature] = normalized
            left = right
    ids = tuple(c.asset_id for c in selected)
    rows = tuple(RankedFeatureRow(c.asset_id, i + 1, c.activity_proxy, c.features,
                                 tuple((f, transformed[c.asset_id][f]) for f in FEATURE_IDS))
                 for i, c in enumerate(selected))
    reasons = tuple((c.asset_id, "selected" if c.asset_id in ids else
                     "outside_top_n" if c.eligible else "required_feature_ineligible")
                    for c in sorted(candidates, key=lambda c: c.asset_id))
    first = candidates[0]
    return CrossSection(first.signal_session, first.signal_at, candidates, ids,
                        rows, reasons, top_n)
