"""Registered roundoff handling on actual raw real-ETF covariance solves only."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_real_optimization import observed as source_observed
from test_real_optimization import solve  # noqa: F401 - real-source pytest fixture

from quantlab.backtest.accounting import rebalance
from quantlab.risk.execution import POLICY_RULES, project_for_execution
from quantlab.risk.optimization import equal_weight

pytestmark = pytest.mark.real_data
POLICY_PATH = Path(__file__).resolve().parents[1] / "configs/execution-projection.json"
POLICY_SHA256 = "7a732bd76731370b65c805fe8cef5c8e8eaaac49faeff9b641157951e4487685"


@pytest.fixture(scope="module")
def real_observed():
    # Reuse the unchanged actual-archive loading fixture, never fabricate prices.
    yield from source_observed.__wrapped__()


def project(result, estimate):
    return project_for_execution(
        result, estimate, policy_path=POLICY_PATH, expected_policy_sha256=POLICY_SHA256
    )


def test_actual_root_registered_policy_identity():
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == POLICY_SHA256
    policy = json.loads(POLICY_PATH.read_bytes())
    assert set(policy) == set(POLICY_RULES) | {"registered_at"}
    assert all(policy[key] == value for key, value in POLICY_RULES.items())


@pytest.mark.parametrize("count", [63, 126])
def test_actual_solver_roundoff_retained_and_strict_accounting_accepts(real_observed, count):
    d = real_observed
    estimate = d["estimates"][count]
    result = solve(d, count)
    raw = result.weights
    matrix = estimate.matrix.copy()
    audit = project(result, estimate)
    assert result.weights == raw == audit.raw_weights == result.attempts[-1].raw_weights
    assert audit.policy_id == "numerical_feasibility_projection_v1"
    assert audit.registered_at <= audit.created_at
    assert audit.total_l1_adjustment <= 1e-8
    assert 0 <= audit.cash_remainder_weight <= 1e-8
    assert all(0 <= w <= c for w, c in zip(audit.applied_weights, result.caps, strict=True))
    assert math.fsum(audit.applied_weights) <= 1
    assert sum(abs(a.change) for a in audit.adjustments) == pytest.approx(audit.total_l1_adjustment)
    if math.fsum(raw) > 1:
        candidates = [i for i, value in enumerate(raw) if value > 0]
        largest = min(candidates, key=lambda i: (-raw[i], result.asset_ids[i]))
        assert audit.adjustments[largest].change < 0
        assert all(row.change == 0 for i, row in enumerate(audit.adjustments) if i != largest)
    else:
        # No budget top-up is permitted, even though full investment is the model target.
        assert all(w <= r for w, r in zip(audit.applied_weights, raw, strict=True))
    np.testing.assert_array_equal(estimate.matrix, matrix)
    assert audit.matrix_modified is False
    assert audit.raw_daily_variance == result.daily_variance
    q = audit.applied_weights
    independent = math.fsum(
        q[i] * float(matrix[i, j]) * q[j] for i in range(len(q)) for j in range(len(q))
    )
    assert audit.applied_daily_variance == independent
    # A declared initial research capital assumption, not a fabricated market price.
    ledger = rebalance({}, 100000.0, audit.targets, 10.0)
    assert ledger.cash_after >= 0 and ledger.nav_after + ledger.fee == pytest.approx(100000.0)
    assert ledger.cash_after == pytest.approx(ledger.nav_after * audit.cash_remainder_weight)
    assert audit.to_record()["covariance_provenance"]["matrix_sha256"] == audit.matrix_sha256


def test_actual_penalty_union_recomputed_before_after_projection(real_observed):
    d = real_observed
    estimate = d["estimates"][126]
    result = solve(d, eta=1.0)
    audit = project(result, estimate)
    outside = dict(result.outside_universe_liquidation)
    coefficient = float(np.diag(estimate.matrix).mean())
    for weights, variance, penalty, objective in [
        (
            audit.raw_weights,
            audit.raw_daily_variance,
            audit.raw_daily_penalty,
            audit.raw_daily_objective,
        ),
        (
            audit.applied_weights,
            audit.applied_daily_variance,
            audit.applied_daily_penalty,
            audit.applied_daily_objective,
        ),
    ]:
        l1 = math.fsum(
            abs(w - d["old"].get(a, 0.0)) for a, w in zip(result.asset_ids, weights, strict=True)
        ) + math.fsum(outside.values())
        assert penalty == pytest.approx(result.turnover_eta * coefficient * l1)
        assert objective == variance + penalty
    assert audit.outside_universe_liquidation == result.outside_universe_liquidation
    assert audit.objective_change == audit.applied_daily_objective - audit.raw_daily_objective
    assert audit.raw_solution_configuration_hash == result.configuration_hash


def test_actual_other_covariance_or_arbitrary_target_refused(real_observed):
    d = real_observed
    result = solve(d, 63)
    with pytest.raises(ValueError, match="provenance differs"):
        project(result, d["estimates"][126])
    with pytest.raises(ValueError, match="Accepted actual"):
        project(result.weights, d["estimates"][63])
    with pytest.raises(ValueError, match="Accepted actual"):
        project(equal_weight(d["ids"], 0.35), d["estimates"][63])
