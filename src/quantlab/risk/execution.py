"""Registered, audited numerical feasibility projection of raw optimizer targets.

This adapter handles roundoff only. It preserves raw solver evidence, never
changes covariance entries or the investment objective, never tops up a tiny
underbudget target, and never substitutes a different model or cash strategy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantlab.risk import optimization
from quantlab.risk.covariance import UNITS, CovarianceEstimate
from quantlab.risk.optimization import PortfolioSolution

POLICY_RULES = {
    "schema": "risk_execution_projection_policy_v1",
    "policy_id": "numerical_feasibility_projection_v1",
    "tolerance": 1e-8,
    "bound_policy": "clip_only_within_registered_tolerance",
    "overbudget_policy": "reduce_largest_positive_tie_asset_id_then_nextafter_toward_zero",
    "underbudget_policy": "retain_cash_no_topup",
    "matrix_policy": "unchanged_no_ridge",
    "raw_weight_policy": "retain_original_unchanged",
}


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _policy(path, expected_sha, created_at):
    raw = Path(path).read_bytes()
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        raise ValueError("Externally authenticated registered projection policy required")
    record = json.loads(raw)
    if set(record) != set(POLICY_RULES) | {"registered_at"} or any(
        record[k] != v for k, v in POLICY_RULES.items()
    ):
        raise ValueError("Unregistered numerical projection rules")
    registered = datetime.fromisoformat(record["registered_at"])
    if registered.tzinfo is None or registered.utcoffset() is None or registered > created_at:
        raise ValueError("Actual prior projection policy registration required")
    return record, registered


def _covariance_binding(solution, estimate):
    if (
        not isinstance(solution, PortfolioSolution)
        or not isinstance(estimate, CovarianceEstimate)
        or solution.status != "valid_raw_target"
        or solution.method != "minimum_variance"
        or solution.weights is None
        or estimate.status != "valid"
        or solution.units != UNITS
        or estimate.units != UNITS
        or solution.asset_ids != estimate.asset_ids
    ):
        raise ValueError(
            "Accepted actual minimum-variance solution and exact daily covariance required"
        )
    ids = solution.asset_ids
    if (
        len(set(ids)) != len(ids)
        or len(solution.weights) != len(ids)
        or len(solution.caps) != len(ids)
        or estimate.matrix.shape != (len(ids), len(ids))
        or not np.isfinite(estimate.matrix).all()
    ):
        raise ValueError("Exact ordered covariance/cap/raw-weight dimensions required")
    provenance = solution.covariance_provenance
    expected = {
        "window_hash": estimate.window_hash,
        "configuration_hash": estimate.configuration_hash,
        "method": estimate.method,
        "parameters": estimate.parameters,
        "source_provenance": estimate.source_provenance,
        "runtime": estimate.runtime,
        "signal_at": estimate.signal_at.isoformat(),
        "max_input_available_at": estimate.max_available_at.isoformat(),
        "training_start": estimate.training_start.isoformat(),
        "training_end": estimate.training_end.isoformat(),
        "units": estimate.units,
        "matrix_sha256": hashlib.sha256(
            estimate.matrix.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
    }
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError("Raw solution covariance/source/configuration provenance differs")
    attempts = solution.attempts
    if (
        [a.solver for a in attempts] not in [["CLARABEL"], ["CLARABEL", "OSQP"]]
        or attempts[-1].status != "optimal"
        or attempts[-1].validation_status != "accepted"
        or attempts[-1].raw_weights != solution.weights
        or any(a.validation_status != "rejected" for a in attempts[:-1])
    ):
        raise ValueError("Accepted raw solver attempt provenance required")
    versions = {
        name: provenance.get("optimization_runtime", {}).get(name)
        for name in optimization.RUNTIME_PACKAGES
    }
    if (
        any(not value for value in versions.values())
        or any(dict(a.versions) != versions for a in attempts)
        or provenance.get("optimization_runtime", {}).get("implementation_sha256")
        != _digest(optimization.__file__)
        or not provenance.get("solver_policy_id")
    ):
        raise ValueError("Raw optimizer implementation/runtime/policy provenance differs")
    configuration = {
        "assets": ids,
        "caps": solution.caps,
        "eta": solution.turnover_eta,
        "signal_close_weights": list(solution.signal_close_weights),
        "provenance": provenance,
        "objective_scale": solution.objective_scaling_factor,
    }
    if _hash(configuration) != solution.configuration_hash:
        raise ValueError("Raw optimizer input configuration differs")
    return expected["matrix_sha256"]


def _terms(weights, solution, estimate):
    old = dict(solution.signal_close_weights)
    if (
        len(old) != len(solution.signal_close_weights)
        or any(not math.isfinite(w) or w < 0 for w in old.values())
        or math.fsum(old.values()) > 1
    ):
        raise ValueError("Original signal-close weights invalid")
    outside = tuple(sorted((a, w) for a, w in old.items() if a not in solution.asset_ids))
    if outside != solution.outside_universe_liquidation:
        raise ValueError("Original union liquidation weights differ")
    coefficient = float(np.mean(np.diag(estimate.matrix)))
    if coefficient != solution.penalty_scale_mean_daily_variance:
        raise ValueError("Original mean daily variance penalty scale differs")
    eta = solution.turnover_eta
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("Original turnover eta invalid")
    variance = math.fsum(
        weights[i] * float(estimate.matrix[i, j]) * weights[j]
        for i in range(len(weights))
        for j in range(len(weights))
    )
    l1 = math.fsum(
        abs(w - old.get(a, 0.0)) for a, w in zip(solution.asset_ids, weights, strict=True)
    ) + math.fsum(w for _, w in outside)
    penalty = eta * coefficient * l1
    objective = variance + penalty
    if not all(math.isfinite(v) for v in (variance, l1, penalty, objective)):
        raise ValueError("Nonfinite objective replay")
    return variance, l1, penalty, objective


@dataclass(frozen=True)
class WeightAdjustment:
    asset_id: str
    cap: float
    raw: float
    applied: float
    change: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionProjection:
    policy_id: str
    policy_sha256: str
    registered_at: str
    created_at: str
    asset_ids: tuple[str, ...]
    caps: tuple[float, ...]
    raw_weights: tuple[float, ...]
    applied_weights: tuple[float, ...]
    adjustments: tuple[WeightAdjustment, ...]
    total_l1_adjustment: float
    maximum_absolute_change: float
    raw_weight_sum: float
    applied_weight_sum: float
    cash_remainder_weight: float
    raw_daily_variance: float
    applied_daily_variance: float
    variance_change: float
    raw_signal_l1_turnover: float
    applied_signal_l1_turnover: float
    raw_daily_penalty: float
    applied_daily_penalty: float
    penalty_change: float
    raw_daily_objective: float
    applied_daily_objective: float
    objective_change: float
    turnover_eta: float
    outside_universe_liquidation: tuple[tuple[str, float], ...]
    raw_solution_configuration_hash: str
    projection_configuration_hash: str
    matrix_sha256: str
    covariance_provenance_json: str
    source_implementation_sha256: str
    weight_basis: str = "postcost_NAV_with_registered_tiny_cash_remainder"
    matrix_modified: bool = False
    role: str = "declared_numerical_roundoff_handling_not_a_performance_improvement"

    @property
    def targets(self):
        return dict(zip(self.asset_ids, self.applied_weights, strict=True))

    def to_record(self):
        result = asdict(self)
        result["covariance_provenance"] = json.loads(result.pop("covariance_provenance_json"))
        return result


def project_for_execution(solution, estimate, *, policy_path, expected_policy_sha256):
    """Apply only the externally authenticated prospective roundoff policy."""
    created = datetime.now(UTC)
    policy, registered = _policy(policy_path, expected_policy_sha256, created)
    matrix_sha = _covariance_binding(solution, estimate)
    first_attempt = datetime.fromisoformat(solution.attempts[0].started_at)
    if first_attempt.tzinfo is None or registered > first_attempt:
        raise ValueError("Projection policy must be registered before this raw solve")
    tolerance = policy["tolerance"]
    raw, caps = solution.weights, solution.caps
    if (
        not all(math.isfinite(w) for w in (*raw, *caps))
        or any(not 0 <= c <= 1 for c in caps)
        or math.fsum(caps) < 1
    ):
        raise ValueError("Finite feasible original raw target/caps required")
    recomputed = {
        "raw_weight_sum": math.fsum(raw),
        "sum_residual": abs(math.fsum(raw) - 1),
        "nonnegative_residual": max(0.0, -min(raw)),
        "cap_residual": max(0.0, max(w - c for w, c in zip(raw, caps, strict=True))),
    }
    for recorded in (dict(solution.residuals), dict(solution.attempts[-1].residuals)):
        if (
            not recorded
            or any(
                v is None
                or not math.isfinite(v)
                or (k != "raw_weight_sum" and not 0 <= v <= tolerance)
                for k, v in recorded.items()
            )
            or any(recorded.get(k) != v for k, v in recomputed.items())
        ):
            raise ValueError("Raw independently checked residual provenance differs or is material")
    before = _terms(raw, solution, estimate)
    for actual, saved in zip(
        before,
        (
            solution.daily_variance,
            solution.estimated_signal_l1_turnover,
            solution.daily_penalty,
            solution.daily_objective,
        ),
        strict=True,
    ):
        if actual != saved:
            raise ValueError("Original raw objective/union terms changed")
    applied, reasons = list(raw), [[] for _ in raw]
    for i, (weight, cap) in enumerate(zip(raw, caps, strict=True)):
        if weight < 0:
            if -weight > tolerance:
                raise ValueError("Material negative raw weight")
            applied[i] = 0.0
            reasons[i].append("clip_tiny_negative_within_registered_tolerance")
        elif weight > cap:
            if weight - cap > tolerance:
                raise ValueError("Material raw cap overshoot")
            applied[i] = cap
            reasons[i].append("clip_tiny_cap_overshoot_within_registered_tolerance")
    if math.fsum(applied) > 1:
        candidates = [i for i, w in enumerate(applied) if w > 0]
        if not candidates:
            raise ValueError("No positive weight available for roundoff reduction")
        i = min(candidates, key=lambda j: (-applied[j], solution.asset_ids[j]))
        replacement = applied[i] - (math.fsum(applied) - 1)
        if replacement < 0:
            raise ValueError("Budget correction exceeds largest positive position")
        applied[i] = replacement
        reasons[i].append("reduce_largest_positive_tie_asset_id_by_budget_excess")
        steps = 0
        while math.fsum(applied) > 1:
            if steps >= 1024 or applied[i] <= 0:
                raise ValueError("Roundoff reduction did not reach an exact nonborrowed budget")
            applied[i] = math.nextafter(applied[i], 0.0)
            reasons[i].append("nextafter_toward_zero_for_exact_budget")
            steps += 1
    adjustments = tuple(
        WeightAdjustment(asset, cap, old, new, new - old, tuple(why))
        for asset, cap, old, new, why in zip(
            solution.asset_ids, caps, raw, applied, reasons, strict=True
        )
    )
    l1 = math.fsum(abs(row.change) for row in adjustments)
    maximum = max(abs(row.change) for row in adjustments)
    total = math.fsum(applied)
    cash = 1 - total
    if (
        any(not math.isfinite(w) or w < 0 or w > cap for w, cap in zip(applied, caps, strict=True))
        or total > 1
        or not 0 <= cash <= tolerance
        or l1 > tolerance
    ):
        raise ValueError("Projection exceeds registered roundoff allowance; strict refusal")
    after = _terms(applied, solution, estimate)
    if (
        hashlib.sha256(estimate.matrix.astype("<f8", copy=False).tobytes()).hexdigest()
        != matrix_sha
    ):
        raise ValueError("Covariance changed during numerical projection")
    config = _hash(
        {
            "policy_sha256": expected_policy_sha256,
            "raw_solution_configuration_hash": solution.configuration_hash,
            "raw_weights": raw,
            "caps": caps,
            "matrix_sha256": matrix_sha,
            "implementation_sha256": _digest(__file__),
        }
    )
    return ExecutionProjection(
        policy["policy_id"],
        expected_policy_sha256,
        registered.isoformat(),
        created.isoformat(),
        solution.asset_ids,
        caps,
        raw,
        tuple(applied),
        adjustments,
        l1,
        maximum,
        recomputed["raw_weight_sum"],
        total,
        cash,
        before[0],
        after[0],
        after[0] - before[0],
        before[1],
        after[1],
        before[2],
        after[2],
        after[2] - before[2],
        before[3],
        after[3],
        after[3] - before[3],
        solution.turnover_eta,
        solution.outside_universe_liquidation,
        solution.configuration_hash,
        config,
        matrix_sha,
        _json(solution.covariance_provenance),
        _digest(__file__),
    )
