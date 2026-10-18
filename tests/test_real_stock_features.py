"""Actual 2010 stock observations verify formulas, availability and exclusions."""

import math
import os
from datetime import date
from pathlib import Path
from statistics import median, stdev

import pytest

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.stocks.features import (
    FEATURE_IDS,
    build_candidates,
    candidate_from_history,
    read_candidates,
    select_cross_section,
)

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def inputs():
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supplied real archive absent; set QUANTLAB_ARCHIVE")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        histories = {}
        for symbol in ("IBM", "GE", "AAPL", "TSLA"):
            bars = source.bars("stock", symbol)
            histories[bars[0].asset_id] = bars
    schedule = XNYSSchedule(date(2008, 1, 1), date(2010, 4, 9))
    return path, histories, schedule


def candidates(inputs):
    _, histories, schedule = inputs
    return build_candidates(histories, schedule, as_of=date(2010, 1, 29),
                            archive_sha256=SUPPLIED_ARCHIVE_SHA256)


def test_real_six_formulas_exact_calendar_and_source_provenance(inputs):
    _, histories, schedule = inputs
    day = date(2010, 1, 29)
    candidate = next(c for c in candidates(inputs) if c.asset_id == "stock:ibm.us")
    assert candidate.eligible and not candidate.reasons
    index = schedule.sessions.index(day)
    days = schedule.sessions[index - 252:index + 1]
    indexed = {b.session: b for b in histories[candidate.asset_id]}
    bars = [indexed[d] for d in days]
    closes = [b.close for b in bars]
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(190, 253)]
    activity = median(b.close * b.volume for b in bars[-63:])
    expected = (
        closes[231] / closes[0] - 1,
        -(closes[252] / closes[231] - 1),
        math.sqrt(252) * stdev(log_returns),
        closes[252] / (sum(closes[190:253]) / 63) - 1,
        closes[252] / max(closes[1:253]) - 1,
        math.log1p(activity),
    )
    assert tuple(k for k, _ in candidate.features) == FEATURE_IDS
    assert tuple(v for _, v in candidate.features) == pytest.approx(expected)
    assert candidate.activity_proxy == pytest.approx(activity)
    assert candidate.source_rows == tuple(bars)
    assert candidate.observed_scheduled_marks == 253
    assert candidate.max_input_available_at == schedule.session(day).decision_at
    assert candidate.intended_execution_at == schedule.next_session(day).open_at
    assert candidate.archive_sha256 == SUPPLIED_ARCHIVE_SHA256
    assert candidate.calendar_sha256 == schedule.provenance["schedule_sha256"]


def test_future_full_history_equals_actual_prefix_and_real_inception_excluded(inputs):
    _, histories, schedule = inputs
    day = date(2010, 1, 29)
    full = candidates(inputs)
    prefix = build_candidates({a: tuple(b for b in bars if b.session <= day)
                               for a, bars in histories.items()}, schedule, as_of=day,
                              archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    assert full == prefix
    tsla = next(c for c in full if c.asset_id == "stock:tsla.us")
    assert histories[tsla.asset_id][0].session > day
    assert not tsla.eligible and tsla.features == () and tsla.source_rows == ()
    assert tsla.reasons == ("MISSING_LATEST_MARK", "INSUFFICIENT_EXACT_SCHEDULED_HISTORY")
    assert tsla.max_input_available_at is None


def test_actual_cross_section_top_activity_ranks_and_singleton(inputs):
    source = candidates(inputs)
    result = select_cross_section(source, top_n=2)
    expected = sorted((c for c in source if c.eligible),
                      key=lambda c: (-c.activity_proxy, c.asset_id))[:2]
    assert result.selected_asset_ids == tuple(c.asset_id for c in expected)
    assert len(result.candidates) == 4
    reversed_result = select_cross_section(tuple(reversed(source)), top_n=2)
    assert result == reversed_result
    assert result.rows == reversed_result.rows
    assert result.selected_asset_ids == reversed_result.selected_asset_ids
    for row in result.rows:
        for feature, actual in row.normalized:
            values = [dict(c.features)[feature] for c in expected]
            own = dict(row.raw)[feature]
            rank = sum(v < own for v in values) + (sum(v == own for v in values) + 1) / 2
            assert actual == pytest.approx((rank - 0.5) / len(values) - 0.5)
    singleton = select_cross_section(source, top_n=1)
    assert all(v == 0 for _, v in singleton.rows[0].normalized)
    assert dict(result.selection_reasons)["stock:tsla.us"] == "required_feature_ineligible"


def test_authenticated_reader_matches_trusted_features_and_month_end_is_required(inputs):
    path, histories, schedule = inputs
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        read = read_candidates(source, ("IBM", "GE", "AAPL", "TSLA"), schedule,
                               as_of=date(2010, 1, 29), archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    assert read == candidates(inputs)
    with pytest.raises(ValueError, match="Final actual exchange session"):
        candidate_from_history("stock:ibm.us", histories["stock:ibm.us"], schedule,
                               as_of=date(2010, 1, 28), archive_sha256=SUPPLIED_ARCHIVE_SHA256)
