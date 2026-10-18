"""XNYS reference schedule; daily-bar availability is an explicit convention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from importlib.metadata import version
from threading import RLock

_HOLIDAY_RANGE_LOCK = RLock()


def _calendar_for_range(start, end):
    """Generate the library's existing rules over the requested historical range.

    CustomBusinessDay asks its holiday calendar for an unbounded holiday list.
    Pandas defaults that list to 1970 onward, omitting earlier regular holidays.
    Configure those public generation bounds while constructing a fresh XNYS
    instance, then restore them. No holiday rules or observations are changed.
    """
    import pandas as pd
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    from pandas.tseries.holiday import AbstractHolidayCalendar

    with _HOLIDAY_RANGE_LOCK:
        previous_start = AbstractHolidayCalendar.start_date
        previous_end = AbstractHolidayCalendar.end_date
        generation_start = min(pd.Timestamp(start), previous_start)
        generation_end = max(pd.Timestamp(end), previous_end)
        try:
            AbstractHolidayCalendar.start_date = generation_start
            AbstractHolidayCalendar.end_date = generation_end
            calendar = XNYSExchangeCalendar(start=start.isoformat(), end=end.isoformat())
        finally:
            AbstractHolidayCalendar.start_date = previous_start
            AbstractHolidayCalendar.end_date = previous_end
    return calendar, generation_start.date(), generation_end.date()


@dataclass(frozen=True)
class SessionTiming:
    session: date
    open_at: datetime
    close_at: datetime
    decision_at: datetime


class XNYSSchedule:
    """Bounded reference schedule with a one-minute after-close decision cutoff.

    The cutoff assumes the daily bar is available then; the archive does not
    establish publication times. Bounds must include intended future executions.
    """

    def __init__(self, start: date, end: date):
        self.library_version = version("exchange-calendars")
        self._calendar, holiday_start, holiday_end = _calendar_for_range(start, end)
        self.sessions = tuple(x.date() for x in self._calendar.sessions)
        self._positions = {day: i for i, day in enumerate(self.sessions)}
        self._timings = {}
        for day in self.sessions:
            key = day.isoformat()
            opening = self._calendar.session_open(key).to_pydatetime()
            closing = self._calendar.session_close(key).to_pydatetime()
            self._timings[day] = SessionTiming(
                day, opening, closing, closing + timedelta(minutes=1)
            )
        rows = [
            (
                d.isoformat(),
                self.session(d).open_at.isoformat(),
                self.session(d).close_at.isoformat(),
            )
            for d in self.sessions
        ]
        self.provenance = {
            "calendar": "XNYS",
            "library": "exchange-calendars",
            "version": self.library_version,
            "pandas_version": version("pandas"),
            "holiday_generation_start": holiday_start.isoformat(),
            "holiday_generation_end": holiday_end.isoformat(),
            "holiday_rule_changes": False,
            "factory": "fresh_XNYS_instance_with_scoped_holiday_generation_bounds",
            "clock_scope": "Library reference hours; historical source publication is unverified",
            "schedule_sha256": hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode()
            ).hexdigest(),
            "availability_convention": "close_plus_one_minute_assumed",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    def session(self, day: date) -> SessionTiming:
        if day not in self._timings:
            raise ValueError(f"Not an included XNYS session: {day}")
        return self._timings[day]

    def next_session(self, day: date) -> SessionTiming:
        self.session(day)
        index = self._positions[day] + 1
        if index >= len(self.sessions):
            raise ValueError("Schedule must extend to intended next session")
        return self.session(self.sessions[index])

    def month_end(self, year: int, month: int) -> SessionTiming:
        from calendar import monthrange

        last = date(year, month, monthrange(year, month)[1])
        if date(year, month, 1) < date.fromisoformat(
            self.provenance["start"]
        ) or last > date.fromisoformat(self.provenance["end"]):
            raise ValueError("Schedule must include the complete calendar month")
        days = [d for d in self.sessions if (d.year, d.month) == (year, month)]
        if not days:
            raise ValueError("Month outside schedule")
        return self.session(days[-1])
