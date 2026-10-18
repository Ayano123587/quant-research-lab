"""Controlled policy arithmetic on real 2010 inputs; no performance evaluation."""

import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantlab.allocation.policies import ASSET_IDS, SYMBOLS, allocate
from quantlab.backtest.engine import HistoricalView
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def inputs():
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supplied real archive absent; set QUANTLAB_ARCHIVE")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        histories = tuple((asset, source.bars("etf", symbol))
                          for asset, symbol in zip(ASSET_IDS, SYMBOLS, strict=True))
    schedule = XNYSSchedule(date(2008, 1, 1), date(2010, 4, 9))
    day = date(2010, 1, 29)
    view = HistoricalView(tuple((a, tuple(b for b in bars if b.session <= day))
                                for a, bars in histories), day,
                          schedule.session(day).decision_at, schedule.next_session(day).open_at)
    return histories, schedule, view


def target(inputs, policy):
    _, schedule, view = inputs
    return allocate(view, schedule, policy=policy, archive_sha256=SUPPLIED_ARCHIVE_SHA256)


def test_equal_and_trend_keep_fixed_sleeve_budgets(inputs):
    equal = target(inputs, "strategic_equal")
    trend = target(inputs, "trend_252")
    assert equal.asset_ids == ASSET_IDS
    assert equal.weights == (0.125,) * 8 and equal.cash_weight == 0
    _, schedule, view = inputs
    index = schedule.sessions.index(view.session)
    first = schedule.sessions[index - 252]
    for i, (_asset, bars) in enumerate(view.histories):
        indexed = {b.session: b for b in bars}
        value = indexed[view.session].close / indexed[first].close - 1
        assert trend.trend_returns[i] == pytest.approx(value)
        assert trend.weights[i] == (0.125 if value > 0 else 0)
    assert trend.cash_weight == 1 - sum(trend.weights)
    assert any(trend.active) and not all(trend.active)
    assert len(trend.source_rows) == 253 * 8
    assert all(b.session <= view.session for b in trend.source_rows)
    assert trend.decision.signal_at == view.cutoff
    assert trend.decision.intended_execution_at == view.next_open_at


@pytest.mark.parametrize("policy", ["volatility_63", "trend_252_volatility_63"])
def test_real_covariance_direct_arithmetic_and_no_leverage(inputs, policy):
    result = target(inputs, policy)
    _, schedule, view = inputs
    index = schedule.sessions.index(view.session)
    days = schedule.sessions[index - 63:index + 1]
    columns = []
    for _, bars in view.histories:
        indexed = {b.session: b for b in bars}
        prices = np.array([indexed[d].close for d in days])
        columns.append(prices[1:] / prices[:-1] - 1)
    covariance = np.cov(np.array(columns), ddof=1)
    np.testing.assert_allclose(result.covariance.matrix, covariance, rtol=1e-12, atol=1e-16)
    base = np.array([0.125 if active else 0 for active in result.active])
    vol = math.sqrt(252 * (base @ covariance @ base))
    scale = min(1, 0.10 / vol) if vol else 1
    assert result.scale == pytest.approx(scale)
    np.testing.assert_allclose(result.weights, base * scale)
    assert 0 <= result.scale <= 1
    assert max(result.weights) <= 0.125
    assert result.cash_weight + sum(result.weights) == pytest.approx(1)
    assert result.covariance.asset_ids == ASSET_IDS
    assert result.covariance.units == "decimal_return_squared_per_session"
    assert result.covariance.diagnostics["n_observations"] == 63
    assert result.covariance.max_available_at <= view.cutoff
    assert result.covariance.source_provenance["archive_sha256"] == SUPPLIED_ARCHIVE_SHA256
    assert result.covariance.training_start == schedule.session(days[0]).close_at


def test_history_container_order_does_not_change_asset_alignment(inputs):
    _, schedule, view = inputs
    reversed_view = HistoricalView(tuple(reversed(view.histories)), view.session,
                                   view.cutoff, view.next_open_at)
    result = allocate(reversed_view, schedule, policy="trend_252_volatility_63",
                      archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    original = target(inputs, "trend_252_volatility_63")
    assert result.weights == original.weights
    assert result.covariance.window_hash == original.covariance.window_hash
    assert result.decision.target == original.decision.target


def test_full_future_source_is_rejected_and_calendar_warmup_is_exact(inputs):
    histories, schedule, view = inputs
    future = HistoricalView(histories, view.session, view.cutoff, view.next_open_at)
    with pytest.raises(ValueError, match="FUTURE_SOURCE_ROW"):
        allocate(future, schedule, policy="trend_252", archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    short_calendar = XNYSSchedule(date(2009, 12, 1), date(2010, 4, 9))
    with pytest.raises(ValueError, match="INSUFFICIENT_CALENDAR_HISTORY"):
        allocate(view, short_calendar, policy="trend_252",
                 archive_sha256=SUPPLIED_ARCHIVE_SHA256)
