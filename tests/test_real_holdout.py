"""Release mechanics on unchanged source records, never final strategy outcomes."""

import os
from datetime import date
from pathlib import Path

import pytest

from quantlab.backtest.engine import Decision
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity
from quantlab.data.returns import monthly_next_open_label
from quantlab.evaluation.holdout import (
    Arm,
    LabelClock,
    LoadedHistory,
    calendar_window,
    file_sha,
    mature_training_keys,
    risk_probe_clock,
    run_development_engineering_window,
    run_shared_holdout,
    source_closure,
)

pytestmark = pytest.mark.real_data
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual():
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", PROJECT / "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supply the actual archive with QUANTLAB_ARCHIVE")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as archive:
        bars = [archive.bars("etf", symbol) for symbol in ("SPY", "IEF")]
    schedule = XNYSSchedule(date(2009, 12, 1), date(2010, 5, 10))
    identity = SourceIdentity(
        "original_archive",
        schedule.provenance["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    return LoadedHistory({r[0].asset_id: r for r in bars}, identity), schedule


def test_actual_cash_close_next_open_fee_and_right_censor(actual):
    loaded, schedule = actual
    start, end = date(2010, 1, 29), date(2010, 2, 3)
    observed = []

    def decide(view, state):
        assert all(bar.session <= view.session for _, rows in view.histories for bar in rows)
        observed.append((view.session, state.cash, state.nav))
        # An explicit equal-weight engineering assumption, not generated prices.
        target = {asset: 0.5 for asset in loaded.histories}
        return Decision(
            "REBALANCE" if view.session == start else "HOLD",
            target if view.session == start else None,
            view.cutoff,
            view.next_open_at,
            "Actual source, fixed equal-weight accounting check",
        )

    arm = Arm("real_equal_pair", "etf", file_sha(PROJECT / "configs/research.toml"), None, decide)
    result = run_development_engineering_window(
        loaded,
        schedule,
        initial_signal=start,
        terminal_close=end,
        arm=arm,
        initial_cash=100000.0,
        fee_bps=10.0,
    )
    assert result.failure is None and result.incomplete_day is None
    window = calendar_window(schedule, start, end)
    assert tuple(row.session for row in result.daily[1:]) == window.measured_sessions
    assert result.daily[0].state.cash == result.daily[0].state.nav == 100000
    assert result.daily[0].execution.fee == 0
    assert result.daily[1].session == schedule.next_session(start).session == date(2010, 2, 1)
    first = result.daily[1]
    # Full initial funding: NAV_after = NAV_before / (1 + one-side cost).
    assert first.execution.fee == pytest.approx(100000 - 100000 / 1.001, abs=1e-8)
    assert first.overnight.asset_pnl == ()
    assert all(abs(row.reconciliation_residual) < 1e-7 for row in result.daily)
    assert all(row.execution.fee == 0 for row in result.daily[2:])
    assert result.endpoint.valued_at == schedule.session(end).close_at
    assert set(dict(result.endpoint.values)) == set(loaded.histories)
    assert result.endpoint_status == "requested_close_marked"
    assert result.pending_status == "right_censored_unexecuted"
    assert result.pending.intended_execution_at == schedule.next_session(end).open_at
    assert observed[0] == (start, 100000.0, 100000.0)


def test_actual_monthly_label_maturity_before_value_loading(actual):
    loaded, schedule = actual
    rows = []
    for asset, bars in loaded.histories.items():
        outcome = monthly_next_open_label(
            bars, schedule, 2010, 1, archive_endpoint=date(2017, 11, 10)
        )
        assert outcome.status == "valid"
        rows.append(
            LabelClock(
                asset,
                date(2010, 1, 29),
                outcome.label_available_at,
                outcome.status,
                bars[0].source_sha256,
            )
        )
    early = mature_training_keys(
        rows,
        signal_session=date(2010, 2, 26),
        decision_at=schedule.session(date(2010, 2, 26)).decision_at,
    )
    assert early.keys == () and dict(early.excluded_status_counts) == {"not_mature_at_fit": 2}
    later = mature_training_keys(
        rows,
        signal_session=date(2010, 3, 31),
        decision_at=schedule.session(date(2010, 3, 31)).decision_at,
    )
    assert later.keys == tuple(
        (row.signal_session, row.asset_id) for row in sorted(rows, key=lambda r: r.key)
    )
    assert later.sample_weights == (1.0, 1.0)
    assert later.maximum_available_at == schedule.session(date(2010, 3, 1)).decision_at


def test_actual_calendar_proxy_uses_22_marks_without_prices(actual):
    _, schedule = actual
    execution = date(2010, 2, 1)
    probe = risk_probe_clock(schedule, execution, source_endpoint=date(2010, 2, 3))
    assert len(probe["closing_sessions"]) == 22 and probe["intervals"] == 21
    assert probe["closing_sessions"][0] == execution
    assert (
        probe["closing_sessions"]
        == schedule.sessions[
            schedule.sessions.index(execution) : schedule.sessions.index(execution) + 22
        ]
    )
    assert probe["status"] == "right_censored" and not probe["outcome_values_read"]


def test_existing_development_metadata_cannot_open_holdout_loader(actual, tmp_path):
    _, schedule = actual
    policy = PROJECT / "configs/execution-projection.json"
    called = []

    def forbidden(*args):
        called.append(True)
        raise AssertionError("Holdout loader must not run")

    arm = Arm("unreleased", "stocks", file_sha(policy), forbidden, forbidden)
    with pytest.raises(ValueError, match="Actual complete shared release"):
        run_shared_holdout(
            policy,
            file_sha(policy),
            project=PROJECT,
            inputs=PROJECT,
            registry={arm.arm_id: arm},
            schedule=schedule,
            dispatch_ticket_path=policy,
            expected_dispatch_ticket_sha256=file_sha(policy),
        )
    assert called == [] and not (tmp_path / "not-consumed.json").exists()


def test_engineering_cannot_bypass_final_gate(actual):
    loaded, schedule = actual
    arm = Arm("none", "etf", SUPPLIED_ARCHIVE_SHA256, None, None)
    with pytest.raises(ValueError, match="refuses shared final period"):
        run_development_engineering_window(
            loaded,
            schedule,
            initial_signal=date(2014, 12, 31),
            terminal_close=date(2015, 1, 2),
            arm=arm,
            initial_cash=100000.0,
            fee_bps=10.0,
        )


def test_actual_source_inventory_includes_transitive_engine_and_initializers():
    closure = source_closure(PROJECT, ["src/quantlab/evaluation/holdout.py"])
    assert {
        "src/quantlab/evaluation/holdout.py",
        "src/quantlab/backtest/engine.py",
        "src/quantlab/backtest/accounting.py",
        "src/quantlab/data/calendar.py",
        "src/quantlab/data/archive.py",
        "src/quantlab/data/provenance.py",
        "src/quantlab/__init__.py",
        "src/quantlab/evaluation/__init__.py",
    } <= set(closure)
    assert all(file_sha(PROJECT / name) == digest for name, digest in closure.items())
