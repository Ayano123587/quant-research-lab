"""Actual-source engineering arithmetic; never a full study, cloud artifact, or selection."""

import csv
import json
import math
import os
import time
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantlab.backtest.engine import Decision, run_history
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive, file_sha256
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import identity_from_source
from quantlab.data.yahoo import YahooETFRelease
from quantlab.risk import economic_study as economics
from quantlab.risk import optimization, study
from quantlab.risk.baseline_execution import project_baseline
from quantlab.risk.covariance import estimate_covariance
from quantlab.risk.signal_weights import project_signal_weights

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"
RELEASE_SHA = "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3"


def test_actual_published_signal_close_ratios_preserve_funding_with_ulp_correction():
    report = ROOT / "reports/etf_controls_v3"
    assert file_sha256(report / "portfolio_daily.csv") == (
        "9db31fd99b8ce12a008f2979136bb32c044fb269d06099c3f935fc06c6e1ca08"
    )
    with (report / "portfolio_daily.csv").open(newline="") as stream:
        row = next(row for row in csv.DictReader(stream)
                   if row["policy"] == "trend_252" and row["session"] == "2011-05-31")
    values = tuple(("etf:" + key.removeprefix("value_").lower() + ".us", float(value))
                   for key, value in row.items() if key.startswith("value_"))
    state = SimpleNamespace(values=values, nav=float(row["close_nav"]), cash=float(row["cash"]))
    audit = project_signal_weights(state)
    assert audit.raw_sum == 1.0000000000000002
    assert audit.applied_sum <= 1
    assert audit.raw_values == values and audit.cash == 0
    assert 0 < audit.total_l1_adjustment <= 4 * math.ulp(1.0)
    assert all(change <= 0 for _, change in audit.adjustments)
    assert sum(change != 0 for _, change in audit.adjustments) == 1
    # Invalid accounting state rejection; actual observed market values stay untouched.
    with pytest.raises(ValueError, match="funded holdings"):
        project_signal_weights(SimpleNamespace(values=values, nav=state.nav, cash=-1.0))
    with pytest.raises(ValueError, match="NAV reconciliation"):
        project_signal_weights(SimpleNamespace(values=values, nav=state.nav - 100, cash=0))


def test_actual_original_source_gap_is_terminal_unsupported_not_complete(tmp_path):
    archive = Path(os.environ.get("QUANTLAB_ARCHIVE", ROOT / "data/raw/archive.zip"))
    if not archive.exists():
        pytest.skip("Supplied original archive required; no fabricated gap fixture")
    with MarketArchive(archive, SUPPLIED_ARCHIVE_SHA256) as source:
        bars = source.bars("etf", "SPY")
    schedule = XNYSSchedule(date(2011, 1, 1), date(2011, 3, 1))
    asset = bars[0].asset_id

    def callback(view, holdings):
        first = view.session == date(2011, 1, 31)
        return Decision("REBALANCE" if first else "HOLD", {asset: 1.0} if first else None,
                        view.cutoff, view.next_open_at, "actual source-gap engineering check")

    engine = run_history(
        {asset: bars}, schedule, start=date(2011, 1, 31), end=date(2011, 2, 18),
        initial_cash=100000.0, fee_bps=10.0, cash_policy="explicit_zero_interest_assumption",
        decision_callback=callback, archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    assert engine.failure is not None
    assert engine.economic_support[-1] == date(2011, 2, 16)
    terminal = economics._save_path(
        tmp_path, "actual_gap_engineering_only", {"complete": False, "engine": engine}, {}
    )
    assert terminal["terminal"] and not terminal["complete"]
    assert terminal["full_period_metrics"] is None
    assert (tmp_path / "UNSUPPORTED").is_file()
    assert (tmp_path / "partial-owned-ledger.json").is_file()
    assert not (tmp_path / "COMPLETE").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_actual_registered_full_family_and_host_guard_before_input_reads(tmp_path, monkeypatch):
    spec = economics.load_spec(ROOT / "configs/risk-economic-development.toml")
    forecast = study.load_spec(ROOT / "configs/risk-development.toml")
    arms = economics.arms(forecast)
    assert len(arms) == len({a.arm_id for a in arms}) == 26
    assert len(arms) * len(economics.FEES) * len(economics.PANELS) == 208
    assert spec["solver_failure"] == "HOLD_actual_previous_positions_and_cash_initially"
    with monkeypatch.context() as host:
        host.chdir(tmp_path)
        actual_is_dir = Path.is_dir
        host.setattr(
            Path,
            "is_dir",
            lambda path: False
            if path in (Path("/kaggle/working"), Path("/kaggle/input"))
            else actual_is_dir(path),
        )
        with pytest.raises(ValueError, match="reviewed Kaggle"):
            economics.run_economic_study(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                tmp_path / "forbidden",
                frozen_solver_policy=None,
                expected_solver_policy_id=None,
                projection_path=None,
                expected_projection_sha256=None,
                service_deadline_monotonic=None,
            )
    assert not (tmp_path / "forbidden").exists()


@pytest.fixture(scope="module")
def actual():
    root = Path(os.environ.get("QUANTLAB_YAHOO_RELEASE", PRIVATE / "yahoo-etf-price-index-v1"))
    if not (root / "COMPLETE").is_file():
        pytest.skip("Approved actual complete Yahoo input release required")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    source = YahooETFRelease(root, RELEASE_SHA, schedule)
    identity = identity_from_source(source, schedule)
    histories = {
        asset: source.bars("etf", asset.split(":")[1].removesuffix(".us"))
        for asset in source.asset_ids
    }
    estimates, windows = {}, {}
    for day in (date(2010, 6, 30), date(2010, 7, 30)):
        start = schedule.sessions[schedule.sessions.index(day) - 252]
        candidates = []
        for asset, history in histories.items():
            days = tuple(b.session for b in history)
            candidate, reasons = study.asset_window(
                asset,
                history[bisect_left(days, start) : bisect_right(days, day)],
                schedule,
                signal_session=day,
            )
            assert not reasons
            candidates.append(candidate)
        window = study.return_window(
            candidates, schedule, 63, signal_session=day, source_identity=identity
        )
        windows[day] = window
        estimates[day] = {"sample_ddof1": estimate_covariance(window)}
    policy_path = ROOT / "configs/execution-projection.json"
    solver = optimization.SolverPolicy(
        tuple((name, version(name)) for name in optimization.RUNTIME_PACKAGES), datetime.now(UTC)
    )
    return (
        schedule,
        identity,
        histories,
        estimates,
        windows,
        policy_path,
        file_sha256(policy_path),
        solver,
    )


@pytest.mark.real_data
def test_rehydrate_actual_computed_covariance_without_inventing_a_cloud_receipt(actual):
    _, _, _, estimates, windows, *_ = actual
    day = date(2010, 6, 30)
    estimate, window = estimates[day]["sample_ddof1"], windows[day]
    recipe = study.recipes(study.load_spec(ROOT / "configs/risk-development.toml"))[0]
    # This is serialization of an actual scoped mathematical calculation, not a
    # fabricated accepted cloud run/COMPLETE or invented market observations.
    record = {
        "status": estimate.status,
        "units": estimate.units,
        "input_sha256": estimate.window_hash,
        "asset_ids": list(estimate.asset_ids),
        "recipe": json.loads(economics._raw(asdict(recipe))),
        "signal_session": str(day),
        "observations": 63,
        "matrix": estimate.matrix.tolist(),
        "location": estimate.location.tolist(),
        "parameters": estimate.parameters,
        "diagnostics": estimate.diagnostics,
        "configuration_hash": estimate.configuration_hash,
        "runtime": estimate.runtime,
        **{
            name: getattr(estimate, name).isoformat()
            for name in ("training_start", "training_end", "max_available_at", "fitted_at")
        },
    }
    restored = economics.rehydrate_estimate(record, window, recipe)
    np.testing.assert_array_equal(restored.matrix, estimate.matrix)
    assert restored.source_provenance == estimate.source_provenance
    assert not restored.matrix.flags.writeable and not restored.location.flags.writeable


@pytest.mark.real_data
@pytest.mark.parametrize("method", ["equal_weight", "inverse_volatility"])
def test_real_baseline_projection_preserves_method_and_actual_variance(actual, method):
    _, _, _, estimates, _, path, sha, _ = actual
    estimate = estimates[date(2010, 6, 30)]["sample_ddof1"]
    solution = (
        optimization.equal_weight(estimate.asset_ids, 0.35)
        if method == "equal_weight"
        else optimization.capped_inverse_volatility(estimate, 0.35)
    )
    projection = project_baseline(solution, estimate, policy_path=path, expected_policy_sha256=sha)
    assert projection.method == solution.method
    assert projection.solver_attempts == () and solution.attempts == ()
    assert projection.raw_weights == solution.weights
    assert 0 <= projection.cash_remainder_weight <= 1e-8
    assert projection.total_l1_adjustment <= 1e-8
    assert all(0 <= w <= 0.35 for w in projection.applied_weights)
    np.testing.assert_allclose(
        projection.applied_daily_variance,
        np.asarray(projection.applied_weights)
        @ estimate.matrix
        @ np.asarray(projection.applied_weights),
        rtol=1e-12,
        atol=1e-18,
    )


@pytest.mark.real_data
@pytest.mark.parametrize(
    "method", ["equal_weight", "capped_inverse_volatility", "minimum_variance"]
)
def test_real_two_signal_ledger_own_holdings_and_money_attribution(actual, method):
    schedule, identity, histories, estimates, _, path, sha, solver = actual
    arm = economics.Arm(
        method,
        None if method == "minimum_variance" else method,
        "sample_ddof1",
        0.01 if method == "minimum_variance" else 0.0,
    )
    # Only June30â€“August2,2010, to exercise two actual month-end signal states.
    # Metrics are software diagnostics; no path is saved as full research support.
    result = economics.run_path(
        histories,
        schedule,
        arm=arm,
        estimates=estimates,
        cap=0.35,
        source_identity=identity,
        fee_bps=10.0,
        solver_policy=solver,
        projection_path=path,
        projection_sha=sha,
        start_signal=date(2010, 6, 30),
        end=date(2010, 8, 2),
    )
    assert result["complete"] is True
    assert len(result["targets"]) == 2
    assert result["targets"][0]["signal_close_weights"] == {}
    assert result["targets"][1]["signal_close_weights"]
    assert result["engine"].source_identity == identity and result["engine"].archive_sha256 is None
    assert result["attribution"]["sessions"] == len(result["measured"])
    assert abs(result["attribution"]["reconciliation_residual"]) < 1e-7
    if method == "minimum_variance":
        assert result["targets"][1]["raw_solution"]["turnover_eta"] == 0.01
        assert result["targets"][1]["raw_solution"]["signal_close_weights"]


def test_actual_expired_clock_refuses_before_running_any_ledger(actual):
    schedule, identity, histories, estimates, _, path, sha, solver = actual
    with pytest.raises(ValueError, match="Future finite path deadline"):
        economics.run_path(
            histories,
            schedule,
            arm=economics.Arm("equal_weight", "equal_weight", "sample_ddof1", 0.0),
            estimates=estimates,
            cap=0.35,
            source_identity=identity,
            fee_bps=10.0,
            solver_policy=solver,
            projection_path=path,
            projection_sha=sha,
            start_signal=date(2010, 6, 30),
            end=date(2010, 7, 9),
            deadline=time.monotonic() - 1,
        )
