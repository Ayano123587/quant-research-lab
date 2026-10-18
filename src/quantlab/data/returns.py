"""Exact scheduled intervals, with unresolved outcomes retained explicitly."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime

from .archive import Bar
from .calendar import XNYSSchedule


@dataclass(frozen=True)
class ReturnInterval:
    asset_id: str
    start_at: datetime
    end_at: datetime
    start_mark: float | None
    end_mark: float | None
    return_value: float | None
    status: str
    return_kind: str
    start_row: tuple[str, int, str] | None
    end_row: tuple[str, int, str] | None
    price_basis: str = "source_OHLC_adjustment_unverified"
    decision_at: datetime | None = None
    label_available_at: datetime | None = None
    quality_flags: tuple[str, ...] = ()


def _index(bars):
    rows = tuple(bars)
    if not rows or len({b.asset_id for b in rows}) != 1:
        raise ValueError("Exactly one nonempty asset history required")
    indexed = {b.session: b for b in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate asset session")
    return rows[0].asset_id, indexed


def _interval(asset, indexed, start, end, field, kind, endpoint):
    left, right = indexed.get(start.session), indexed.get(end.session)
    a = getattr(left, field) if left else None
    b = getattr(right, field) if right else None
    status = "valid"
    if end.session > endpoint:
        status = "right_censored"
    elif left is None or right is None:
        status = "missing_mark"
    elif left.quality_flags or right.quality_flags:
        status = "flagged_source_mark"
    elif not math.isfinite(a) or not math.isfinite(b):
        status = "nonfinite_mark"
    elif a <= 0 or b <= 0:
        status = "nonpositive_mark"

    def identity(row):
        return (row.source_member, row.row_number, row.row_sha256) if row else None

    return ReturnInterval(
        asset,
        getattr(start, field + "_at"),
        getattr(end, field + "_at"),
        a,
        b,
        b / a - 1 if status == "valid" else None,
        status,
        kind,
        identity(left),
        identity(right),
        quality_flags=tuple(
            sorted(
                set((left.quality_flags if left else ()) + (right.quality_flags if right else ()))
            )
        ),
    )


def close_intervals(
    bars: tuple[Bar, ...], schedule: XNYSSchedule, start: date, end: date, *, archive_endpoint: date
) -> tuple[ReturnInterval, ...]:
    asset, indexed = _index(bars)
    schedule.session(start)
    schedule.session(end)
    if start >= end:
        raise ValueError("Interval start must precede end")
    days = [d for d in schedule.sessions if start <= d <= end]
    return tuple(
        _interval(
            asset,
            indexed,
            schedule.session(a),
            schedule.session(b),
            "close",
            "scheduled_close_to_close",
            archive_endpoint,
        )
        for a, b in zip(days, days[1:], strict=False)
    )


def trailing_returns(
    bars: tuple[Bar, ...],
    schedule: XNYSSchedule,
    as_of: date,
    count: int = 252,
    *,
    archive_endpoint: date,
) -> tuple[ReturnInterval, ...]:
    if count < 1:
        raise ValueError("Positive interval count required")
    schedule.session(as_of)
    i = schedule.sessions.index(as_of)
    if i < count:
        raise ValueError("Insufficient scheduled calendar history")
    result = close_intervals(
        bars, schedule, schedule.sessions[i - count], as_of, archive_endpoint=archive_endpoint
    )
    if any(x.status != "valid" for x in result):
        raise ValueError("Insufficient consecutive valid scheduled prices")
    return result


def monthly_next_open_label(
    bars: tuple[Bar, ...], schedule: XNYSSchedule, year: int, month: int, *, archive_endpoint: date
) -> ReturnInterval:
    asset, indexed = _index(bars)
    signal = schedule.month_end(year, month)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    next_signal = schedule.month_end(next_year, next_month)
    result = _interval(
        asset,
        indexed,
        schedule.next_session(signal.session),
        schedule.next_session(next_signal.session),
        "open",
        "monthly_next_open",
        archive_endpoint,
    )
    # Daily OHLC does not establish when the opening print was published.
    # Conservatively expose this label only after that session's close convention.
    available = schedule.next_session(next_signal.session).decision_at
    return replace(
        result,
        decision_at=signal.decision_at,
        label_available_at=available if result.status == "valid" else None,
    )
