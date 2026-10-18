"""Declared full-investment covariance targets; no cash or model fallback.

Raw solver weights are retained without clipping or renormalization. A numerical
solution within this module's tolerance may still be refused by strict downstream
money accounting. Execution projection, if ever studied, needs its own registered
raw/applied-weight and recomputed-risk audit; none is performed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from quantlab.risk.covariance import UNITS, CovarianceEstimate

FEASIBILITY_TOLERANCE = 1e-8
SOLVER_TOLERANCE = 1e-9
RUNTIME_PACKAGES = ("cvxpy", "clarabel", "osqp", "numpy", "scipy")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _ids(ids):
    if not isinstance(ids, (tuple, list)):
        raise ValueError("Explicit ordered asset IDs required")
    result = tuple(ids)
    if (
        not result
        or any(not isinstance(a, str) or not a for a in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("Unique nonempty asset IDs required")
    return result


def _caps(ids, caps):
    if isinstance(caps, Mapping):
        if tuple(caps) != ids:
            raise ValueError("Cap asset order must exactly match covariance order")
        raw = tuple(caps.values())
    elif isinstance(caps, (int, float)) and not isinstance(caps, bool):
        raw = (caps,) * len(ids)
    else:
        raise ValueError("Scalar cap or exact-order asset-keyed caps required")
    if any(isinstance(v, bool) for v in raw):
        raise ValueError("Caps must be numeric")
    values = tuple(float(v) for v in raw)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or math.fsum(values) < 1:
        raise ValueError("INFEASIBLE_CAPS: long-only full-risky budget requires sum(caps)>=1")
    return values


def _covariance(estimate):
    if (
        not isinstance(estimate, CovarianceEstimate)
        or estimate.status != "valid"
        or estimate.units != UNITS
    ):
        raise ValueError("Valid daily-unit CovarianceEstimate required")
    ids = _ids(estimate.asset_ids)
    matrix = estimate.matrix
    if matrix.shape != (len(ids), len(ids)) or not np.isfinite(matrix).all():
        raise ValueError("Finite matrix must match explicit asset order")
    norm = float(np.linalg.norm(matrix, ord=2))
    tolerance = max(1e-15, norm * 1e-10)
    minimum = float(np.linalg.eigvalsh(matrix)[0])
    asymmetry = float(np.linalg.norm(matrix - matrix.T, ord=2))
    if minimum < -tolerance or asymmetry > tolerance or np.any(np.diag(matrix) < 0):
        raise ValueError("Invalid covariance; no added ridge, eigenvalue clip or symmetry repair")
    if (
        estimate.max_available_at > estimate.signal_at
        or estimate.training_end > estimate.max_available_at
    ):
        raise ValueError("Covariance uses inputs unavailable at signal close")
    return (
        ids,
        matrix,
        {
            "minimum_eigenvalue": minimum,
            "symmetry_residual": asymmetry,
            "psd_tolerance": tolerance,
            "matrix_repair": "none",
        },
    )


def _residuals(weights, caps):
    if len(weights) != len(caps) or not all(math.isfinite(w) for w in weights):
        return None
    return {
        "raw_weight_sum": math.fsum(weights),
        "sum_residual": abs(math.fsum(weights) - 1),
        "nonnegative_residual": max(0.0, -min(weights)),
        "cap_residual": max(0.0, max(w - c for w, c in zip(weights, caps, strict=True))),
    }


def _variance(weights, matrix):
    return math.fsum(
        weights[i] * float(matrix[i, j]) * weights[j]
        for i in range(len(weights))
        for j in range(len(weights))
    )


@dataclass(frozen=True)
class SolverPolicy:
    """Actual audit freeze of one strategy; not historical preregistration."""

    expected_versions: tuple[tuple[str, str], ...]
    frozen_at: datetime
    primary: str = "CLARABEL"
    retry: str = "OSQP"
    feasibility_tolerance: float = FEASIBILITY_TOLERANCE
    solver_tolerance: float = SOLVER_TOLERANCE

    def __post_init__(self):
        pairs = (
            tuple(self.expected_versions.items())
            if isinstance(self.expected_versions, Mapping)
            else tuple(self.expected_versions)
        )
        if (
            len(dict(pairs)) != len(pairs)
            or set(dict(pairs)) != set(RUNTIME_PACKAGES)
            or any(not isinstance(v, str) or not v for _, v in pairs)
        ):
            raise ValueError("Explicit locked solver/numeric runtime versions required")
        if (
            self.primary != "CLARABEL"
            or self.retry != "OSQP"
            or self.feasibility_tolerance != FEASIBILITY_TOLERANCE
            or self.solver_tolerance != SOLVER_TOLERANCE
            or self.frozen_at.tzinfo is None
            or self.frozen_at.utcoffset() is None
        ):
            raise ValueError("Only the declared CLARABEL then one OSQP policy is supported")
        object.__setattr__(self, "expected_versions", tuple(sorted(pairs)))

    def settings(self, solver):
        if solver == "CLARABEL":
            return {
                "tol_gap_abs": SOLVER_TOLERANCE,
                "tol_gap_rel": SOLVER_TOLERANCE,
                "tol_feas": SOLVER_TOLERANCE,
                "max_iter": 1000,
            }
        if solver == "OSQP":
            return {
                "eps_abs": SOLVER_TOLERANCE,
                "eps_rel": SOLVER_TOLERANCE,
                "max_iter": 100000,
                "polishing": True,
                "adaptive_rho": True,
            }
        raise ValueError("Undeclared solver")

    @property
    def policy_id(self):
        return _hash(
            {
                "versions": self.expected_versions,
                "actual_frozen_at": self.frozen_at.isoformat(),
                "primary": self.primary,
                "retry_once": self.retry,
                "settings": {s: self.settings(s) for s in (self.primary, self.retry)},
                "raw_residual_tolerance": self.feasibility_tolerance,
                "accepted_status": "optimal_only_no_inaccurate",
            }
        )


@dataclass(frozen=True)
class SolverAttempt:
    solver: str
    status: str
    settings: tuple[tuple[str, float | int | bool], ...]
    started_at: str
    finished_at: str
    wall_seconds: float
    iterations: int | None
    solve_seconds: float | None
    setup_seconds: float | None
    reported_scaled_objective: float | None
    raw_weights: tuple[float, ...] | None
    residuals: tuple[tuple[str, float | None], ...]
    validation_status: str
    reason: str
    versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PortfolioSolution:
    method: str
    status: str
    reason: str
    asset_ids: tuple[str, ...]
    caps: tuple[float, ...]
    weights: tuple[float, ...] | None
    daily_variance: float | None
    daily_penalty: float | None
    daily_objective: float | None
    turnover_eta: float
    penalty_scale_mean_daily_variance: float | None
    signal_close_weights: tuple[tuple[str, float], ...]
    outside_universe_liquidation: tuple[tuple[str, float], ...]
    estimated_signal_l1_turnover: float | None
    objective_scaling_factor: float
    residuals: tuple[tuple[str, float], ...]
    attempts: tuple[SolverAttempt, ...]
    covariance_provenance: dict
    configuration_hash: str
    raw_weights_modified: bool = False
    execution_projection: str = "none; downstream strict budget/sign rules may refuse raw residuals"
    weight_basis: str = "full_risky_budget_1_of_postcost_NAV"
    units: str = UNITS

    def to_record(self):
        return asdict(self)


def _provenance(estimate, diagnostics):
    if estimate is None:
        return {"covariance": "not_applicable"}
    return {
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
        "diagnostics": diagnostics,
        "matrix_sha256": hashlib.sha256(
            estimate.matrix.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
    }


def _baseline(method, ids, caps, weights, estimate=None, diagnostics=None):
    residuals = _residuals(weights, caps)
    if (
        residuals is None
        or max(residuals[k] for k in ("sum_residual", "nonnegative_residual", "cap_residual"))
        > FEASIBILITY_TOLERANCE
    ):
        raise ValueError("Baseline raw weights violate the full-budget/cap contract")
    provenance = _provenance(estimate, diagnostics)
    config = _hash({"method": method, "assets": ids, "caps": caps, "provenance": provenance})
    variance = _variance(weights, estimate.matrix) if estimate is not None else None
    return PortfolioSolution(
        method,
        "valid_raw_target",
        "No normalization or execution projection",
        ids,
        caps,
        weights,
        variance,
        None,
        variance,
        0.0,
        None,
        (),
        (),
        None,
        1.0,
        tuple(sorted(residuals.items())),
        (),
        provenance,
        config,
    )


def equal_weight(ids, caps):
    ids = _ids(ids)
    bounds = _caps(ids, caps)
    weights = (1 / len(ids),) * len(ids)
    if any(w > cap for w, cap in zip(weights, bounds, strict=True)):
        raise ValueError(
            "Equal weights violate a cap; do not relabel capped redistribution as equal weight"
        )
    return _baseline("equal_weight", ids, bounds, weights)


def capped_inverse_volatility(estimate, caps):
    ids, matrix, diagnostics = _covariance(estimate)
    if estimate.method != "sample_ddof1":
        raise ValueError("Baseline requires the fixed sample_ddof1 reference covariance")
    bounds = _caps(ids, caps)
    diagonal = tuple(float(matrix[i, i]) for i in range(len(ids)))
    if any(value <= 0 for value in diagonal):
        raise ValueError("Inverse volatility undefined for nonpositive variance; no epsilon")
    inverse = tuple(1 / math.sqrt(value) for value in diagonal)
    weights, free = [0.0] * len(ids), set(range(len(ids)))
    while free:
        remaining = 1 - math.fsum(weights)
        denominator = math.fsum(inverse[i] for i in sorted(free))
        proposed = {i: remaining * inverse[i] / denominator for i in sorted(free)}
        capped = [i for i in sorted(free) if proposed[i] > bounds[i]]
        if not capped:
            for i in sorted(free):
                weights[i] = proposed[i]
            break
        for i in capped:
            weights[i] = bounds[i]
            free.remove(i)
    return _baseline(
        "capped_inverse_volatility_not_risk_parity",
        ids,
        bounds,
        tuple(weights),
        estimate,
        diagnostics,
    )


def minimum_variance(
    estimate, caps, signal_close_weights_by_asset, turnover_eta=0.0, *, frozen_solver_policy
):
    ids, matrix, diagnostics = _covariance(estimate)
    bounds = _caps(ids, caps)
    policy = frozen_solver_policy
    if not isinstance(policy, SolverPolicy) or policy.frozen_at > datetime.now(UTC):
        raise ValueError("Actual prior frozen solver policy required")
    actual_versions = tuple(sorted((name, version(name)) for name in RUNTIME_PACKAGES))
    if actual_versions != policy.expected_versions:
        raise ValueError("Locked optimization runtime differs from frozen solver policy")
    if not isinstance(signal_close_weights_by_asset, Mapping):
        raise ValueError("Explicit signal-close asset-keyed weights required")
    old = {}
    for asset, weight in signal_close_weights_by_asset.items():
        if not isinstance(asset, str) or not asset or isinstance(weight, bool):
            raise ValueError("Valid signal-close weights required")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Finite long-only signal-close weights required")
        old[asset] = weight
    if math.fsum(old.values()) > 1:
        raise ValueError("Signal-close risky weights cannot borrow")
    if (
        isinstance(turnover_eta, bool)
        or not math.isfinite(float(turnover_eta))
        or float(turnover_eta) < 0
    ):
        raise ValueError("Finite nonnegative predeclared turnover_eta required")
    eta = float(turnover_eta)
    diagonal_mean = float(np.mean(np.diag(matrix)))
    if diagonal_mean < 0 or not math.isfinite(diagonal_mean):
        raise ValueError("Invalid mean daily variance; no objective-scale epsilon")
    scale = 1 / diagonal_mean if diagonal_mean > 0 else 1.0
    if not math.isfinite(scale):
        raise ValueError("Unrepresentable variance scaling; no clipping")
    old_inside = np.array([old.get(a, 0.0) for a in ids])
    outside = tuple(sorted((a, w) for a, w in old.items() if a not in ids))
    liquidation = math.fsum(w for _, w in outside)
    provenance = _provenance(estimate, diagnostics)
    provenance["optimization_runtime"] = dict(actual_versions) | {
        "python": platform.python_version(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    provenance["solver_policy_id"] = policy.policy_id
    provenance["actual_solver_policy_frozen_at"] = policy.frozen_at.isoformat()
    config = _hash(
        {
            "assets": ids,
            "caps": bounds,
            "eta": eta,
            "signal_close_weights": sorted(old.items()),
            "provenance": provenance,
            "objective_scale": scale,
        }
    )
    import cvxpy as cp

    variable = cp.Variable(len(ids))
    # psd_wrap declares the independently verified PSD-with-tolerance input; it
    # does not alter its entries, add a ridge or repair its eigenvalues.
    variance_expression = cp.quad_form(variable, cp.psd_wrap(matrix))
    l1_expression = cp.norm1(variable - old_inside) + liquidation
    objective = cp.Minimize(scale * (variance_expression + eta * diagonal_mean * l1_expression))
    problem = cp.Problem(
        objective, [cp.sum(variable) == 1, variable >= 0, variable <= np.array(bounds)]
    )
    attempts = []
    for solver in (policy.primary, policy.retry):
        started, clock = datetime.now(UTC), time.perf_counter()
        reason, status, reported, weights, residuals = "", "not_run", None, None, {}
        stats = None
        try:
            problem.solve(solver=solver, warm_start=False, verbose=False, **policy.settings(solver))
            status, stats = str(problem.status), problem.solver_stats
            reported = (
                float(problem.value)
                if problem.value is not None and math.isfinite(problem.value)
                else None
            )
            weights = (
                tuple(float(w) for w in variable.value) if variable.value is not None else None
            )
            candidate = _residuals(weights, bounds) if weights is not None else None
            if candidate is not None:
                residuals.update(candidate)
                variance = _variance(weights, matrix)
                l1 = (
                    math.fsum(abs(w - old.get(a, 0.0)) for a, w in zip(ids, weights, strict=True))
                    + liquidation
                )
                penalty = eta * diagonal_mean * l1
                raw_objective = variance + penalty
                residuals["objective_residual_daily_units"] = (
                    abs(raw_objective - reported / scale) if reported is not None else None
                )
                residuals["objective_residual_scaled"] = (
                    abs(raw_objective * scale - reported) if reported is not None else None
                )
            accepted = (
                status == "optimal"
                and candidate is not None
                and reported is not None
                and all(
                    value is not None and math.isfinite(value) and value <= FEASIBILITY_TOLERANCE
                    for name, value in residuals.items()
                    if name != "raw_weight_sum"
                )
            )
            reason = (
                "Raw constraints/objective independently verified"
                if accepted
                else "Non-optimal/inaccurate status or raw feasibility/objective residual refused"
            )
        except (cp.error.SolverError, ValueError, ArithmeticError) as error:
            accepted = False
            reason = type(error).__name__ + ": " + str(error)[:300]
            status = "solver_exception"
        finished = datetime.now(UTC)
        attempts.append(
            SolverAttempt(
                solver,
                status,
                tuple(sorted(policy.settings(solver).items())),
                started.isoformat(),
                finished.isoformat(),
                time.perf_counter() - clock,
                stats.num_iters if stats else None,
                stats.solve_time if stats else None,
                stats.setup_time if stats else None,
                reported,
                weights,
                tuple(sorted(residuals.items())),
                "accepted" if accepted else "rejected",
                reason,
                actual_versions,
            )
        )
        if accepted:
            return PortfolioSolution(
                "minimum_variance",
                "valid_raw_target",
                reason,
                ids,
                bounds,
                weights,
                variance,
                penalty,
                raw_objective,
                eta,
                diagonal_mean,
                tuple(sorted(old.items())),
                outside,
                l1,
                scale,
                tuple(sorted(residuals.items())),
                tuple(attempts),
                provenance,
                config,
            )
    return PortfolioSolution(
        "minimum_variance",
        "solver_failed",
        "Both declared attempts refused; no target/cash/model fallback",
        ids,
        bounds,
        None,
        None,
        None,
        None,
        eta,
        diagonal_mean,
        tuple(sorted(old.items())),
        outside,
        None,
        scale,
        (),
        tuple(attempts),
        provenance,
        config,
    )
