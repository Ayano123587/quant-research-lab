"""Checks against the authenticated supplied archive; no manufactured price rows."""

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.returns import close_intervals, monthly_next_open_label, trailing_returns

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def histories():
    path = Path(
        os.environ.get(
            "QUANTLAB_ARCHIVE",
            str(Path(__file__).resolve().parents[1] / "data" / "raw" / "archive.zip"),
        )
    )
    if not path.exists():
        pytest.skip("Authenticated supplied archive required")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as archive:
        return {symbol: archive.bars("etf", symbol) for symbol in ("SPY", "IEF", "DBC", "BIL")}


@pytest.fixture(scope="module")
def schedule():
    return XNYSSchedule(date(2004, 1, 1), date(2018, 1, 31))


def test_real_holidays_early_close_and_next_scheduled_open(histories, schedule):
    spy = {b.session: b for b in histories["SPY"]}
    friday = date(2017, 11, 3)
    assert friday in spy and date(2017, 11, 6) in spy
    assert schedule.next_session(friday).session == date(2017, 11, 6)
    assert date(2016, 11, 24) not in spy and date(2016, 11, 24) not in schedule.sessions
    early = schedule.session(date(2016, 11, 25))
    assert early.close_at.hour == 18
    assert early.close_at.utcoffset() == timedelta(0)
    assert early.decision_at == early.close_at + timedelta(minutes=1)
    assert schedule.session(date(2016, 11, 23)).close_at.hour == 21
    assert len(schedule.provenance["schedule_sha256"]) == 64


def test_real_1962_histories_and_authored_holiday_range():
    from pandas.tseries.holiday import AbstractHolidayCalendar

    path = Path(
        os.environ.get(
            "QUANTLAB_ARCHIVE",
            str(Path(__file__).resolve().parents[1] / "data" / "raw" / "archive.zip"),
        )
    )
    if not path.exists():
        pytest.skip("Authenticated supplied archive required")
    before = (AbstractHolidayCalendar.start_date, AbstractHolidayCalendar.end_date)
    historical = XNYSSchedule(date(1962, 1, 1), date(1962, 1, 10))
    assert (AbstractHolidayCalendar.start_date, AbstractHolidayCalendar.end_date) == before
    assert date(1962, 1, 1) not in historical.sessions
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as archive:
        for symbol in ("IBM", "GE"):
            bars = archive.bars("stock", symbol)
            assert bars[0].session == historical.sessions[0] == date(1962, 1, 2)
            actual = tuple(bar.session for bar in bars if bar.session <= date(1962, 1, 10))
            assert actual == historical.sessions


def test_actual_252_intervals_need_253_prices(histories, schedule):
    bars = histories["SPY"]
    result = trailing_returns(
        bars, schedule, date(2017, 11, 10), archive_endpoint=date(2017, 11, 10)
    )
    assert len(result) == 252
    assert len({x.start_row for x in result} | {result[-1].end_row}) == 253
    assert all(x.status == "valid" for x in result)
    assert result[-1].end_mark == bars[-1].close
    assert result[-1].return_value == bars[-1].close / bars[-2].close - 1


def test_real_inception_is_missing_history_not_filled(histories, schedule):
    for symbol in ("DBC", "BIL"):
        bars = histories[symbol]
        first = bars[0].session
        prior = schedule.sessions[schedule.sessions.index(first) - 1]
        result = close_intervals(bars, schedule, prior, first, archive_endpoint=date(2017, 11, 10))
        assert result[0].status == "missing_mark"
        assert result[0].start_row is None and result[0].end_row is not None
        with pytest.raises(ValueError, match="consecutive valid"):
            trailing_returns(bars, schedule, first, archive_endpoint=date(2017, 11, 10))


def test_real_monthly_completed_and_truncated_labels(histories, schedule):
    for bars in histories.values():
        completed = monthly_next_open_label(
            bars, schedule, 2017, 9, archive_endpoint=date(2017, 11, 10)
        )
        assert completed.status == "valid"
        assert completed.start_at.date() == date(2017, 10, 2)
        assert completed.end_at.date() == date(2017, 11, 1)
        partial = monthly_next_open_label(
            bars, schedule, 2017, 10, archive_endpoint=date(2017, 11, 10)
        )
        assert partial.status == "right_censored"
        assert partial.end_at.date() == date(2017, 12, 1)
        assert partial.return_value is None and partial.end_mark is None
        final_month = monthly_next_open_label(
            bars, schedule, 2017, 11, archive_endpoint=date(2017, 11, 10)
        )
        assert final_month.status == "right_censored"
        assert final_month.start_at.date() == date(2017, 12, 1)


def test_real_flagged_hlg_price_is_not_a_valid_close_return(schedule):
    path = Path(
        os.environ.get(
            "QUANTLAB_ARCHIVE",
            str(Path(__file__).resolve().parents[1] / "data" / "raw" / "archive.zip"),
        )
    )
    if not path.is_file():
        pytest.skip("Authenticated supplied archive required")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        bars = source.bars("stock", "HLG")
    actual = next(b for b in bars if b.session == date(2017, 8, 10))
    assert actual.low == -1 and actual.close > 0 and actual.quality_flags
    result = close_intervals(
        bars, schedule, date(2017, 8, 9), date(2017, 8, 11), archive_endpoint=date(2017, 11, 10)
    )
    assert len(result) == 2
    assert all(x.status == "flagged_source_mark" and x.return_value is None for x in result)
    assert all(set(actual.quality_flags) <= set(x.quality_flags) for x in result)
    with pytest.raises(ValueError, match="consecutive valid"):
        trailing_returns(
            bars, schedule, date(2017, 8, 11), count=2, archive_endpoint=date(2017, 11, 10)
        )
