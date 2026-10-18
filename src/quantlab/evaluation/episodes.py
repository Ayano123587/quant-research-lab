"""Descriptive drawdown episodes of a fixed observed reference portfolio.

Eventual troughs and recoveries are hindsight outcomes, never strategy inputs.
Unrecovered episodes retain the final observation and an explicit censor flag.
"""

from __future__ import annotations

import math
from datetime import date


def drawdown_episodes(sessions, nav, *, initial_session, initial_nav, threshold=-0.10):
    dates, values = tuple(sessions), tuple(nav)
    if (
        not dates
        or len(dates) != len(values)
        or type(initial_session) is not date
        or any(type(day) is not date for day in dates)
        or initial_session >= dates[0]
        or any(left >= right for left, right in zip(dates, dates[1:], strict=False))
        or not math.isfinite(initial_nav)
        or initial_nav <= 0
        or not -1 < threshold < 0
        or any(not math.isfinite(value) or value <= 0 for value in values)
    ):
        raise ValueError("Complete ordered observed positive NAV history required")
    peak_nav, peak_day = initial_nav, initial_session
    active = None
    result = []
    for day, value in zip(dates, values, strict=True):
        if active is None:
            if value >= peak_nav:
                peak_nav, peak_day = value, day
            elif value / peak_nav - 1 <= threshold:
                active = {
                    "peak_session": peak_day.isoformat(),
                    "peak_nav": peak_nav,
                    "threshold_crossing_session": day.isoformat(),
                    "trough_session": day.isoformat(),
                    "trough_nav": value,
                }
        else:
            if value < active["trough_nav"]:
                active.update(trough_session=day.isoformat(), trough_nav=value)
            if value >= active["peak_nav"]:
                active.update(
                    end_session=day.isoformat(),
                    recovery_session=day.isoformat(),
                    right_censored=False,
                )
                result.append(active)
                active = None
                peak_nav, peak_day = value, day
    if active is not None:
        active.update(end_session=dates[-1].isoformat(), recovery_session=None, right_censored=True)
        result.append(active)
    for index, episode in enumerate(result, start=1):
        episode.update(
            episode_id=f"reference_drawdown_{index:03d}",
            threshold=threshold,
            maximum_reference_drawdown=episode["trough_nav"] / episode["peak_nav"] - 1,
            hindsight_outcome_only=True,
        )
    return result
