"""Software invariants on unchanged real records; no performance claims."""

import os
from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from pathlib import Path

import pytest

from quantlab.backtest.engine import Decision, run_history
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def histories():
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supplied real archive absent; set QUANTLAB_ARCHIVE")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        bars = [source.bars("etf", symbol) for symbol in ("SPY", "IEF", "VOO")]
        bars.append(source.bars("stock", "ARE"))
    return {b[0].asset_id: b for b in bars}


def run(histories, callback, **overrides):
    args = dict(
        start=date(2010, 1, 4),
        end=date(2010, 3, 31),
        initial_cash=10000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=callback,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    args.update(overrides)
    return run_history(histories, XNYSSchedule(date(2009, 12, 1), date(2010, 4, 9)), **args)


def test_real_event_ownership_frozen_boundary_and_final_pending(histories):
    asset = next(a for a in histories if a.startswith("etf:spy."))
    seen = []

    def callback(view, state):
        assert all(b.session <= view.session for _, bars in view.histories for b in bars)
        assert not hasattr(view, "schedule") and not hasattr(view, "source")
        with pytest.raises(FrozenInstanceError):
            state.cash = 0
        seen.append(view)
        target = {asset: 1.0}
        decision = Decision(
            "REBALANCE" if len(seen) == 1 else "HOLD",
            target if len(seen) == 1 else None,
            view.cutoff,
            view.next_open_at,
            "fixed software check",
        )
        target.clear()
        return decision

    result = run(histories, callback)
    assert result.failure is None
    assert len(seen) == len(result.daily) == len(result.economic_support)
    first, purchase, drift = result.daily[:3]
    assert first.state.nav == 10000 and first.execution.fee == 0
    assert purchase.overnight.nav_before == purchase.overnight.nav_after == 10000
    assert purchase.execution.fee > 0 and purchase.execution.cash_after == 0
    assert drift.execution.action == "HOLD" and drift.execution.traded_notional == 0
    bars = {b.session: b for b in histories[asset]}
    expected = (
        purchase.execution.nav_after * bars[purchase.session].close / bars[purchase.session].open
    )
    assert purchase.state.nav == pytest.approx(expected)
    assert drift.overnight.nav_after == pytest.approx(
        purchase.state.nav * bars[drift.session].open / bars[purchase.session].close
    )
    assert all(abs(row.reconciliation_residual) < 1e-8 for row in result.daily)
    assert result.pending.intended_execution_at > result.endpoint.valued_at
    assert result.pending_status == "right_censored_unexecuted"


def test_actual_preinception_fill_stops_without_later_fill(histories):
    asset = next(a for a in histories if a.startswith("etf:voo."))
    assert histories[asset][0].session > date(2010, 1, 5)

    def callback(view, state):
        return Decision(
            "REBALANCE",
            {asset: 1},
            view.cutoff,
            view.next_open_at,
            "fixed pre-inception source support check",
        )

    result = run(histories, callback)
    assert result.failure.session == date(2010, 1, 5)
    assert result.failure.unresolved_assets == (asset,)
    assert len(result.daily) == 1 and result.endpoint.nav == 10000
    assert result.endpoint_status == "incomplete_valid_prefix"
    assert result.pending_status == "blocked_unexecuted"


def test_real_prefix_extension_preserves_earlier_decisions(histories):
    def callback(view, state):
        return Decision(
            "HOLD",
            None,
            view.cutoff,
            view.next_open_at,
            str(tuple((a, len(b)) for a, b in view.histories)),
        )

    short = run(histories, callback, end=date(2010, 1, 29))
    longer = run(histories, callback)
    assert short.decisions == longer.decisions[: len(short.decisions)]
    assert short.daily == longer.daily[: len(short.daily)]


def test_real_calendar_rejects_same_close_execution(histories):
    def callback(view, state):
        return Decision("HOLD", None, view.cutoff, view.cutoff, "invalid timing check")

    with pytest.raises(ValueError, match="next scheduled open"):
        run(histories, callback)


def test_actual_archive_endpoint_keeps_unresolved_held_exposure(histories):
    asset = next(a for a in histories if a.startswith("etf:spy."))
    last = histories[asset][-1].session
    schedule = XNYSSchedule(last - timedelta(days=14), last + timedelta(days=14))
    end = schedule.next_session(last).session
    start = schedule.sessions[schedule.sessions.index(last) - 2]

    def callback(view, state):
        return Decision(
            "REBALANCE" if view.session == start else "HOLD",
            {asset: 1} if view.session == start else None,
            view.cutoff,
            view.next_open_at,
            "actual archive endpoint check",
        )

    result = run_history(
        histories,
        schedule,
        start=start,
        end=end,
        initial_cash=10000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=callback,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    assert result.failure.session == end
    assert result.endpoint.valued_at == schedule.session(last).close_at
    assert dict(result.endpoint.values)[asset] > 0
    assert result.failure.unresolved_assets == (asset,)
    assert result.economic_support[-1] == last


def test_actual_flagged_close_preserves_prior_open_purchase(histories):
    """ARE's unchanged 2006-06-06 row has an inconsistent high, not a bad open.

    This event was selected from the source-quality audit, never by its return.
    The end-of-day finding cannot cancel an already scheduled opening purchase.
    """
    asset = next(a for a in histories if a.startswith("stock:are."))
    indexed = {b.session: b for b in histories[asset]}
    bad_day = date(2006, 6, 6)
    assert indexed[bad_day].quality_flags == ("INCONSISTENT_OHLC",)
    assert indexed[bad_day].open > 0
    schedule = XNYSSchedule(date(2006, 5, 30), date(2006, 6, 9))
    start = date(2006, 6, 5)

    def callback(view, state):
        return Decision(
            "REBALANCE",
            {asset: 1},
            view.cutoff,
            view.next_open_at,
            "fixed real source-quality event check",
        )

    result = run_history(
        histories,
        schedule,
        start=start,
        end=bad_day,
        initial_cash=10000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=callback,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    partial = result.incomplete_day
    assert result.failure.event == "close" and result.failure.session == bad_day
    assert result.failure.unresolved_assets == (asset,)
    assert partial.session == bad_day
    assert partial.execution.fee > 0
    assert partial.execution.action == "REBALANCE"
    assert partial.state.valued_at == schedule.session(bad_day).open_at
    assert partial.state.nav == pytest.approx(10000 / 1.001)
    assert dict(partial.state.values)[asset] == partial.state.nav
    assert partial.money_pnl == pytest.approx(-partial.execution.fee)
    assert abs(partial.reconciliation_residual) < 1e-8
    assert result.endpoint == result.failure.last_valid_state == partial.state
    assert result.economic_support == (start,)
    assert result.pending is None
    assert result.pending_status == "executed_before_unresolved_close"
    assert result.endpoint_status == "incomplete_open_marked"


def test_actual_asset_sold_at_open_does_not_require_its_flagged_close(histories):
    asset = next(a for a in histories if a.startswith("stock:are."))
    indexed = {b.session: b for b in histories[asset]}
    assert not indexed[date(2006, 6, 2)].quality_flags
    assert not indexed[date(2006, 6, 5)].quality_flags
    assert indexed[date(2006, 6, 6)].quality_flags
    schedule = XNYSSchedule(date(2006, 5, 30), date(2006, 6, 9))
    start, end = date(2006, 6, 2), date(2006, 6, 6)

    def callback(view, state):
        if view.session == start:
            action, target = "REBALANCE", {asset: 1}
        elif view.session == date(2006, 6, 5):
            action, target = "REBALANCE", {}
        else:
            action, target = "HOLD", None
        return Decision(
            action,
            target,
            view.cutoff,
            view.next_open_at,
            "fixed real source-quality liquidation check",
        )

    result = run_history(
        histories,
        schedule,
        start=start,
        end=end,
        initial_cash=10000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=callback,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    assert result.failure is None and result.incomplete_day is None
    sale = result.daily[-1]
    assert sale.session == end and sale.execution.fee > 0
    assert dict(sale.execution.positions)[asset] == 0
    assert sale.state.cash == sale.state.nav
    assert sale.intraday.asset_pnl == ((asset, 0.0),)
    assert sale.state.valued_at == schedule.session(end).close_at
