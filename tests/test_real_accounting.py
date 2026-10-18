"""Hand reconcile accounting against genuine supplied SPY/IEF records only.

Capital/units/target weights and 10 bps are explicit engineering assumptions,
not empirical trading costs or a claimed strategy. No manufactured or mutated
market rows, future-selected dates, synthetic calendars or profitability tests.
"""

import hashlib
import math
import os
from pathlib import Path

import pytest

from quantlab.backtest.accounting import hold, mark_positions, rebalance
from quantlab.backtest.replay import replay_one_decision
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule

pytestmark = pytest.mark.real_data

ARCHIVE = Path(
    os.environ.get(
        "QUANTLAB_ARCHIVE",
        str(Path(__file__).resolve().parents[1] / "data" / "raw" / "archive.zip"),
    )
)
FEE_BPS = 10.0  # Explicit proportional per-side research assumption, not measured cost.


@pytest.fixture(scope="module")
def observed():
    if not ARCHIVE.is_file():
        pytest.skip("User-supplied real archive required; no replacement data generated")
    with MarketArchive(ARCHIVE, SUPPLIED_ARCHIVE_SHA256) as source:
        spy, ief = source.bars("etf", "SPY"), source.bars("etf", "IEF")
        start = max(spy[0].session, ief[0].session)
        # Actual library calendar is used even for this small development trace.
        schedule = XNYSSchedule(start, min(spy[-1].session, ief[-1].session))
        histories = {
            spy[0].asset_id: {b.session: b for b in spy},
            ief[0].asset_id: {b.session: b for b in ief},
        }
        signal = next(s for s in schedule.sessions[:-1] if all(s in h for h in histories.values()))
        execution = schedule.next_session(signal).session
        rows = {a: (h[signal], h[execution]) for a, h in histories.items()}
        assert all(not bar.quality_flags for pair in rows.values() for bar in pair)
        yield source, schedule, rows


def test_real_rows_have_independent_raw_byte_provenance(observed):
    source, schedule, rows = observed
    for asset, pair in rows.items():
        symbol = "SPY" if asset == "etf:spy.us" else "IEF"
        member, raw = source.source_bytes("etf", symbol)
        for bar in pair:
            assert bar.source_member == member
            assert bar.source_sha256 == hashlib.sha256(raw).hexdigest()
            assert (
                bar.row_sha256
                == hashlib.sha256(raw.splitlines(keepends=True)[bar.row_number - 1]).hexdigest()
            )
    assert (
        schedule.next_session(next(iter(rows.values()))[0].session).session
        == next(iter(rows.values()))[1].session
    )


def test_cash_purchase_reserves_fee_using_real_source_cash_value(observed):
    source, schedule, rows = observed
    asset = "etf:spy.us"
    cash = rows[asset][0].close  # Declared cash funding equal to one actual closing share value.
    result = rebalance({}, cash, {asset: 1.0}, FEE_BPS)
    rate = FEE_BPS / 10000
    assert result.nav_after == pytest.approx(cash / (1 + rate))
    assert result.fee == pytest.approx(result.positions[asset] * rate)
    assert result.cash_after == 0
    assert result.nav_after + result.fee == pytest.approx(cash)
    assert result.traded_notional == pytest.approx(result.positions[asset])


def test_full_real_asset_replacement_preserves_old_new_union(observed):
    source, schedule, rows = observed
    old, new = "etf:spy.us", "etf:ief.us"
    value = rows[old][1].open  # One actually observed SPY unit already owned at this open.
    result = rebalance({old: value}, 0.0, {new: 1.0}, FEE_BPS)
    rate = FEE_BPS / 10000
    expected = value * (1 - rate) / (1 + rate)
    assert result.nav_after == pytest.approx(expected)
    legs = {leg.asset_id: leg for leg in result.legs}
    assert set(legs) == {old, new}
    assert legs[old].signed_notional == -value
    assert legs[new].signed_notional == pytest.approx(expected)
    assert result.fee == pytest.approx(rate * (value + expected))
    assert result.cash_after >= 0


def test_real_drift_hold_and_resubmitted_target_are_distinct(observed):
    source, schedule, rows = observed
    closes = {a: pair[0].close for a, pair in rows.items()}
    opens = {a: pair[1].open for a, pair in rows.items()}
    initial = dict(closes)  # One observed share of each ETF, with no invented mark.
    target = {a: value / math.fsum(initial.values()) for a, value in initial.items()}
    marked = mark_positions(initial, 0.0, closes, opens, cash_return=0.0)
    assert marked.positions == pytest.approx(opens)
    held = hold(marked.positions, marked.cash)
    restored = rebalance(marked.positions, marked.cash, target, FEE_BPS)
    assert held.action == "HOLD" and held.fee == held.traded_notional == 0
    assert held.positions == marked.positions
    # This is observed event drift, never a profitability assertion.
    expected_notional = math.fsum(abs(target[a] * restored.nav_after - opens[a]) for a in target)
    assert restored.traded_notional == pytest.approx(expected_notional)
    assert restored.nav_after + restored.fee == pytest.approx(marked.nav_after)
    assert restored.fee == pytest.approx(expected_notional * FEE_BPS / 10000)


def test_real_liquidation_fees_and_cash_reconcile(observed):
    source, schedule, rows = observed
    values = {a: pair[1].open for a, pair in rows.items()}
    result = rebalance(values, 0.0, {}, FEE_BPS)
    notional = math.fsum(values.values())
    assert result.traded_notional == pytest.approx(notional)
    assert result.fee == pytest.approx(notional * FEE_BPS / 10000)
    assert result.cash_after == pytest.approx(notional - result.fee)
    assert all(v == 0 for v in result.positions.values())


def test_missing_real_held_mark_refuses_instead_of_dropping_position(observed):
    source, schedule, rows = observed
    closes = {a: pair[0].close for a, pair in rows.items()}
    opens = {a: pair[1].open for a, pair in rows.items()}
    unavailable = {a: price for a, price in opens.items() if a != "etf:ief.us"}
    with pytest.raises(ValueError, match="UNRESOLVED_HELD_ASSET"):
        mark_positions(closes, 0.0, closes, unavailable, cash_return=0.0)


def test_real_one_decision_full_event_hand_reconciliation(observed):
    source, schedule, rows = observed
    old, new = "etf:spy.us", "etf:ief.us"
    result = replay_one_decision(
        source,
        schedule,
        initial_units={old: 1.0},
        initial_cash=0.0,
        target_weights={new: 1.0},
        fee_bps=FEE_BPS,
        action="REBALANCE",
        cash_policy="explicit_zero_interest_assumption",
    )
    rate = FEE_BPS / 10000
    old_close, old_open = rows[old][0].close, rows[old][1].open
    expected_after = old_open * (1 - rate) / (1 + rate)
    expected_final = expected_after * rows[new][1].close / rows[new][1].open
    assert result["archive_sha256"] == SUPPLIED_ARCHIVE_SHA256
    assert result["execution"]["nav_before"] == pytest.approx(old_open)
    assert result["execution"]["nav_after"] == pytest.approx(expected_after)
    assert result["intraday_posttrade_owners"]["nav_after"] == pytest.approx(expected_final)
    assert dict(result["overnight_existing_owners"]["asset_pnl"])[old] == pytest.approx(
        old_open - old_close
    )
    assert result["reconciliation_residual"] == pytest.approx(0.0, abs=1e-10)
    assert result["future_outcomes_used_by_replay_for_target"] is False
    assert result["benchmark_or_profitability_claim"] is False
    assert result["signal_rows"] and result["execution_rows"]


def test_real_replay_hold_never_resubmits_weights(observed):
    source, schedule, rows = observed
    asset = "etf:spy.us"
    result = replay_one_decision(
        source,
        schedule,
        initial_units={asset: 1.0},
        initial_cash=0.0,
        target_weights=None,
        fee_bps=FEE_BPS,
        action="HOLD",
        cash_policy="explicit_zero_interest_assumption",
    )
    assert result["execution"]["traded_notional"] == result["execution"]["fee"] == 0
    assert result["intraday_posttrade_owners"]["nav_after"] == pytest.approx(rows[asset][1].close)
    with pytest.raises(ValueError, match="HOLD"):
        replay_one_decision(
            source,
            schedule,
            initial_units={asset: 1.0},
            initial_cash=0.0,
            target_weights={asset: 1.0},
            fee_bps=FEE_BPS,
            action="HOLD",
            cash_policy="explicit_zero_interest_assumption",
        )
