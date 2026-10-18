"""Optimizer arithmetic on unchanged authenticated ETF returns, never fake matrices."""

import hashlib
import math
import os
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.returns import trailing_returns
from quantlab.risk.covariance import ReturnWindow, estimate_covariance
from quantlab.risk.optimization import (
    RUNTIME_PACKAGES,
    SolverPolicy,
    capped_inverse_volatility,
    equal_weight,
    minimum_variance,
)

pytestmark = pytest.mark.real_data
CAP = 0.35


@pytest.fixture(scope="module")
def observed():
    path = Path(
        os.environ.get(
            "QUANTLAB_ARCHIVE", str(Path(__file__).resolve().parents[1] / "data/raw/archive.zip")
        )
    )
    if not path.is_file():
        pytest.skip("Supplied actual archive required; no replacement prices or matrices")
    policy = SolverPolicy(
        tuple((name, version(name)) for name in RUNTIME_PACKAGES), datetime.now(UTC)
    )
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        symbols = ("SPY", "IEF", "EFA", "TLT", "GLD", "DBC")
        histories = [source.bars("etf", symbol) for symbol in symbols]
        bil = source.bars("etf", "BIL")
    schedule = XNYSSchedule(date(2008, 1, 1), date(2010, 12, 31))
    signal = schedule.month_end(2010, 1)
    endpoint = min(rows[-1].session for rows in histories)
    ids = tuple(rows[0].asset_id for rows in histories)
    intervals = [
        trailing_returns(rows, schedule, signal.session, 126, archive_endpoint=endpoint)
        for rows in histories
    ]
    estimates = {}
    for count in (2, 63, 126):
        columns = [values[-count:] for values in intervals]
        values = np.array([[column[i].return_value for column in columns] for i in range(count)])
        times = columns[0]
        provenance = hashlib.sha256(
            repr([[v.start_row for v in col] + [col[-1].end_row] for col in columns]).encode()
        ).hexdigest()
        window = ReturnWindow(
            values,
            ids,
            tuple(v.start_at for v in times),
            tuple(v.end_at for v in times),
            tuple(schedule.session(v.end_at.date()).decision_at for v in times),
            signal.decision_at,
            SUPPLIED_ARCHIVE_SHA256,
            "source_close_to_close_adjustment_unverified",
            "fixed_six_real_ETFs_optimization_check",
            "all_requested_past_intervals_valid",
            {
                "source_rows_sha256": provenance,
                "schedule_sha256": schedule.provenance["schedule_sha256"],
            },
        )
        estimates[count] = estimate_covariance(window, "sample_ddof1")
    closes = {
        rows[0].asset_id: next(b.close for b in rows if b.session == signal.session)
        for rows in histories
    }
    bil_close = next(b.close for b in bil if b.session == signal.session)
    # One actual closing unit per asset plus an explicit cash reserve equal to
    # one observed BIL share value. No market observation is manufactured.
    total = math.fsum([*closes.values(), bil_close, bil_close])
    old = {a: value / total for a, value in closes.items()} | {bil[0].asset_id: bil_close / total}
    yield dict(estimates=estimates, policy=policy, ids=ids, old=old)


def solve(d, count=126, eta=0.0):
    return minimum_variance(
        d["estimates"][count], CAP, d["old"], eta, frozen_solver_policy=d["policy"]
    )


def test_equal_weight_caps_are_not_silently_redistributed(observed):
    d = observed
    result = equal_weight(d["ids"], CAP)
    assert result.weights == (1 / len(d["ids"]),) * len(d["ids"])
    assert result.status == "valid_raw_target" and result.attempts == ()
    # These are explicit policy constraints on the same actual asset universe.
    with pytest.raises(ValueError, match="INFEASIBLE_CAPS"):
        equal_weight(d["ids"], 0.1)
    caps = {asset: 1.0 for asset in d["ids"]}
    caps[d["ids"][0]] = 0.1
    with pytest.raises(ValueError, match="Equal weights violate"):
        equal_weight(d["ids"], caps)


def test_actual_inverse_volatility_cap_waterfill_independent_reference(observed):
    d = observed
    estimate = d["estimates"][126]
    result = capped_inverse_volatility(estimate, CAP)
    inv = [1 / math.sqrt(float(estimate.matrix[i, i])) for i in range(len(d["ids"]))]
    # Independent scalar multiplier root: sum(min(cap,k/sigma))=1.
    low, high = 0.0, 1.0
    while math.fsum(min(CAP, high * v) for v in inv) < 1:
        high *= 2
    for _ in range(100):
        mid = (low + high) / 2
        if math.fsum(min(CAP, mid * v) for v in inv) > 1:
            high = mid
        else:
            low = mid
    reference = [min(CAP, (low + high) / 2 * v) for v in inv]
    assert result.weights == pytest.approx(reference, rel=1e-13, abs=1e-15)
    assert abs(math.fsum(result.weights) - 1) < 1e-12
    assert all(0 <= w <= CAP for w in result.weights)
    assert "not_risk_parity" in result.method
    assert result.raw_weights_modified is False


@pytest.mark.parametrize("count", [63, 126])
def test_real_minimum_variance_matches_independent_scipy_problem(observed, count):
    d = observed
    estimate = d["estimates"][count]
    result = solve(d, count)
    assert result.status == "valid_raw_target"
    assert result.weights == result.attempts[-1].raw_weights
    matrix = estimate.matrix
    scale = 1 / float(np.diag(matrix).mean())
    equal = np.array(equal_weight(d["ids"], CAP).weights)
    reference = minimize(
        lambda w: float(w @ matrix @ w) * scale,
        equal,
        jac=lambda w: 2 * matrix @ w * scale,
        method="SLSQP",
        bounds=[(0.0, CAP)] * len(equal),
        constraints={
            "type": "eq",
            "fun": lambda w: float(w.sum() - 1),
            "jac": lambda w: np.ones(len(w)),
        },
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert reference.success, reference.message
    assert abs(math.fsum(result.weights) - 1) <= 1e-8
    assert min(result.weights) >= -1e-8 and max(result.weights) <= CAP + 1e-8
    assert result.daily_variance <= float(equal @ matrix @ equal) + 1e-8
    assert result.daily_variance == pytest.approx(
        float(reference.x @ matrix @ reference.x), rel=2e-5, abs=1e-10
    )
    assert result.daily_penalty == 0 and result.daily_objective == result.daily_variance
    assert (
        result.covariance_provenance["source_provenance"]["source_snapshot_id"]
        == SUPPLIED_ARCHIVE_SHA256
    )
    for attempt in result.attempts:
        assert attempt.status == "optimal" or attempt.validation_status == "rejected"
        assert dict(attempt.versions) == dict(d["policy"].expected_versions)
        assert attempt.iterations is not None and attempt.wall_seconds >= 0
    assert [a.solver for a in result.attempts] in [["CLARABEL"], ["CLARABEL", "OSQP"]]
    assert result.raw_weights_modified is False


def test_actual_two_return_singular_psd_needs_no_ridge(observed):
    d = observed
    estimate = d["estimates"][2]
    before = estimate.matrix.copy()
    assert estimate.diagnostics["singular"] and estimate.diagnostics["rank"] <= 1
    result = solve(d, 2)
    assert result.status == "valid_raw_target"
    np.testing.assert_array_equal(estimate.matrix, before)
    assert result.covariance_provenance["diagnostics"]["matrix_repair"] == "none"
    assert result.daily_variance >= -1e-12
    assert abs(math.fsum(result.weights) - 1) <= 1e-8


def test_actual_signal_close_union_turnover_penalty_and_scaling(observed):
    d = observed
    result = solve(d, eta=1.0)
    assert result.status == "valid_raw_target"
    outside = dict(result.outside_universe_liquidation)
    assert outside == {"etf:bil.us": d["old"]["etf:bil.us"]}
    l1 = math.fsum(
        abs(w - d["old"].get(a, 0.0)) for a, w in zip(result.asset_ids, result.weights, strict=True)
    ) + math.fsum(outside.values())
    daily_scale = float(np.diag(d["estimates"][126].matrix).mean())
    assert result.estimated_signal_l1_turnover == pytest.approx(l1)
    assert result.daily_penalty == pytest.approx(daily_scale * l1)
    assert result.daily_objective == pytest.approx(result.daily_variance + result.daily_penalty)
    assert result.objective_scaling_factor == pytest.approx(1 / daily_scale)
    assert result.attempts[-1].reported_scaled_objective == pytest.approx(
        result.daily_objective * result.objective_scaling_factor, abs=1e-8
    )
    assert "strict budget/sign" in result.execution_projection


def test_declared_retry_actual_osqp_after_injected_primary_software_failure(observed, monkeypatch):
    """Software fault only; the real63-return matrix and objective remain unchanged."""
    import cvxpy as cp

    d = observed
    estimate = d["estimates"][63]
    original_matrix = estimate.matrix.copy()
    original = cp.Problem.solve
    calls = []

    def fail_primary_only(problem, *args, **kwargs):
        calls.append((id(problem), id(problem.objective), kwargs["solver"]))
        if kwargs["solver"] == "CLARABEL":
            raise cp.error.SolverError(
                "Explicit test-only injected primary software failure; not a natural solver failure"
            )
        assert kwargs["solver"] == "OSQP"
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_primary_only)
    result = solve(d, count=63)
    assert [a.solver for a in result.attempts] == ["CLARABEL", "OSQP"]
    assert result.attempts[0].status == "solver_exception"
    assert "injected primary software failure" in result.attempts[0].reason
    assert result.attempts[0].validation_status == "rejected"
    assert result.attempts[1].status == "optimal"
    assert result.attempts[1].validation_status == "accepted"
    assert result.status == "valid_raw_target"
    assert calls[0][:2] == calls[1][:2]  # Identical CVXPY problem and objective.
    assert all(dict(a.versions) == dict(d["policy"].expected_versions) for a in result.attempts)
    assert dict(result.attempts[1].settings) == d["policy"].settings("OSQP")
    np.testing.assert_array_equal(estimate.matrix, original_matrix)
    variance = math.fsum(
        result.weights[i] * float(estimate.matrix[i, j]) * result.weights[j]
        for i in range(len(result.weights))
        for j in range(len(result.weights))
    )
    assert result.daily_variance == variance and result.daily_objective == variance
    residuals = dict(result.residuals)
    assert abs(math.fsum(result.weights) - 1) <= 1e-8
    assert min(result.weights) >= -1e-8 and max(result.weights) <= CAP + 1e-8
    assert residuals["objective_residual_scaled"] <= 1e-8
    assert result.raw_weights_modified is False
