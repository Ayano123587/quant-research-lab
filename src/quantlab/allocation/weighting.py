"""Strategic sleeve weights before any trend mask or volatility scaler."""

import math
from dataclasses import dataclass

import numpy as np

from quantlab.allocation.policies import ASSET_IDS
from quantlab.risk.covariance import CovarianceEstimate


@dataclass(frozen=True)
class StrategicWeights:
    method: str
    asset_ids: tuple[str, ...]
    weights: tuple[float, ...]
    cap: float
    risk_shares: tuple[float, ...]
    risk_share_max_absolute_deviation_from_equal: float
    sum_residual: float
    cap_residual: float
    funding_roundoff_reduction: float
    cap_roundoff_reduction: float
    solver: dict
    covariance_window_sha256: str


def strategic_weights(estimate, method, *, cap=0.25):
    """Use the same observed covariance and report attained, possibly unequal risk shares.

    The capped risk-budget method solves a convex log-barrier problem with linear
    normalized position caps. Caps can prevent equal risk contributions. No active
    trend subset is accepted here: strategic budgets cover all original sleeves.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize, nnls

    methods = {"equal_capital", "capped_inverse_volatility", "capped_risk_budget"}
    if (
        not isinstance(estimate, CovarianceEstimate)
        or tuple(estimate.asset_ids) != ASSET_IDS
        or estimate.units != "decimal_return_squared_per_session"
    ):
        raise ValueError("Typed actual daily covariance in exact ETF sleeve order required")
    assets = tuple(estimate.asset_ids)
    matrix = np.array(estimate.matrix, dtype=float, copy=True)
    n = len(assets)
    if (
        method not in methods
        or n != 8
        or len(set(assets)) != n
        or cap != 0.25
        or matrix.shape != (n, n)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-16, rtol=1e-12)
    ):
        raise ValueError("Eight aligned sleeves, declared method and fixed25pct cap required")
    eigenvalues = np.linalg.eigvalsh(matrix)
    diagonal = np.diag(matrix)
    if diagonal.min() <= 0 or eigenvalues.min() <= eigenvalues.max() * 1e-12:
        raise ValueError(
            "Strictly positive definite observed covariance required; no invented ridge"
        )
    solver = {"name": "closed_form", "success": True}
    if method == "equal_capital":
        weights = np.full(n, 1 / n)
    elif method == "capped_inverse_volatility":
        raw = 1 / np.sqrt(diagonal)
        weights = np.zeros(n)
        free = np.ones(n, dtype=bool)
        remaining = 1.0
        for _ in range(n):
            proposed = remaining * raw[free] / math.fsum(raw[free])
            if (proposed <= cap).all():
                weights[free] = proposed
                break
            indices = np.flatnonzero(free)
            capped = indices[proposed > cap]
            weights[capped] = cap
            free[capped] = False
            remaining = 1 - math.fsum(weights)
        else:
            raise ValueError("Capped inverse volatility did not terminate")
    else:
        # Scaling the covariance avoids ill-conditioned daily-return units. A
        # positive scalar does not change normalized risk-budget weights.
        scaled = matrix / float(diagonal.mean())
        equal = np.full(n, 1 / n)
        initial = equal / np.sqrt(equal @ scaled @ equal)

        def objective(value):
            return float(0.5 * value @ scaled @ value - np.log(value).mean())

        def gradient(value):
            return scaled @ value - 1 / (n * value)

        cap_constraint = cap * np.ones((n, n)) - np.eye(n)
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(np.full(n, 1e-12), np.full(n, np.inf)),
            constraints=LinearConstraint(cap_constraint, 0, np.inf),
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if (
            not result.success
            or not np.isfinite(result.x).all()
            or not math.isfinite(float(result.fun))
        ):
            raise ValueError("Capped risk-budget solver failed; no substitute allocation")

        def certificate(value):
            slack = cap_constraint @ value
            active = slack <= 1e-7 * max(1.0, float(np.linalg.norm(value)))
            if value.min() <= 1e-10:
                raise ValueError("Risk-budget solution hit numerical positivity bound")
            grad = gradient(value)
            if active.any():
                multipliers, _ = nnls(cap_constraint[active].T, grad)
                stationarity = grad - cap_constraint[active].T @ multipliers
                complementarity = float(np.max(np.abs(multipliers * slack[active])))
            else:
                multipliers = np.array([])
                stationarity, complementarity = grad, 0.0
            residual = float(np.max(np.abs(stationarity)))
            normalized = value / math.fsum(value)
            if (
                not np.isfinite(grad).all()
                or not math.isfinite(complementarity)
                or not math.isfinite(residual)
                or normalized.min() < 0
                or abs(math.fsum(normalized) - 1) > 1e-12
                or max(0.0, float(normalized.max()) - cap) > 1e-12
            ):
                raise ValueError("Risk-budget iterate feasibility or finite certificate failed")
            return slack, active, grad, multipliers, stationarity, complementarity, residual

        def attempt_record(value, ftol, residual, complementarity):
            return {
                "success": bool(value.success),
                "status": int(value.status),
                "message": str(value.message),
                "iterations": int(value.nit),
                "objective": float(value.fun),
                "ftol": ftol,
                "maxiter": 2000,
                "stationarity_residual": residual,
                "complementarity_residual": complementarity,
            }

        (slack, active, grad, multipliers, stationarity, complementarity, stationarity_residual) = (
            certificate(result.x)
        )
        initial_attempt = attempt_record(result, 1e-12, stationarity_residual, complementarity)
        refinement_attempt = None
        final_ftol = 1e-12
        if stationarity_residual > 1e-6 or complementarity > 1e-7:
            # One prospective numerical refinement, only after the successful,
            # finite, positive and feasible original iterate fails its unchanged
            # KKT certificate. Objective, covariance, constraints and gates remain
            # identical. A failed refinement never substitutes another allocation.
            result = minimize(
                objective,
                result.x.copy(),
                jac=gradient,
                method="SLSQP",
                bounds=Bounds(np.full(n, 1e-12), np.full(n, np.inf)),
                constraints=LinearConstraint(cap_constraint, 0, np.inf),
                options={"ftol": 1e-14, "maxiter": 2000},
            )
            if (
                not result.success
                or not np.isfinite(result.x).all()
                or not math.isfinite(float(result.fun))
                or float(result.fun) > initial_attempt["objective"]
            ):
                raise ValueError(
                    "Risk-budget numerical refinement failed; no substitute allocation"
                )
            (
                slack,
                active,
                grad,
                multipliers,
                stationarity,
                complementarity,
                stationarity_residual,
            ) = certificate(result.x)
            refinement_attempt = attempt_record(
                result, 1e-14, stationarity_residual, complementarity
            )
            final_ftol = 1e-14
        if stationarity_residual > 1e-6 or complementarity > 1e-7:
            raise ValueError("Risk-budget optimality residual failed")
        weights = result.x / math.fsum(result.x)
        solver = {
            "name": "scipy_SLSQP_convex_log_budget",
            "success": bool(result.success),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "covariance_positive_scale": float(diagonal.mean()),
            "linear_cap_constraint_minimum": float((cap_constraint @ result.x).min()),
            "ftol": final_ftol,
            "maxiter": 2000,
            "initial_attempt": initial_attempt,
            "refinement_attempt": refinement_attempt,
            "refinement_policy": (
                "one_successful_feasible_original_KKT_failure_warmstart_same_problem_ftol1e-14"
            ),
            "stationarity_residual": stationarity_residual,
            "complementarity_residual": complementarity,
            "equal_contributions_not_guaranteed_under_caps": True,
        }
    sum_residual = math.fsum(weights) - 1
    cap_residual = max(0.0, float(weights.max()) - cap)
    if weights.min() < 0 or abs(sum_residual) > 1e-12 or cap_residual > 1e-12:
        raise ValueError("Strategic sleeve feasibility failed")
    solver["raw_normalized_cap_residual"] = cap_residual
    solver["raw_normalized_sum_residual"] = sum_residual
    solver["representation_policy"] = (
        "reduce_only_cap_excess_at_most1e-12_per_sleeve_then_sum_excess_at_most1e-12; "
        "no_redistribution_or_topup"
    )
    # Record one-sided solver representation excess; never redistribute it.
    cap_reduction = math.fsum(max(0.0, float(w) - cap) for w in weights)
    weights = np.minimum(weights, cap)
    cap_residual = max(0.0, float(weights.max()) - cap)
    sum_residual = math.fsum(weights) - 1
    # Decimal weights must also be executable without borrowing. Reduce only a
    # positive representation excess, never scale up a budget or repair infeasibility.
    reduction = 0.0
    if sum_residual > 0:
        index = min(range(n), key=lambda i: (-weights[i], assets[i]))
        original = float(weights[index])
        weights[index] -= sum_residual
        for _ in range(16):
            if math.fsum(weights) <= 1:
                break
            weights[index] = math.nextafter(float(weights[index]), 0.0)
        else:
            raise ValueError("Strategic budget representation could not be funded")
        reduction = original - float(weights[index])
        if reduction > 1e-12:
            raise ValueError("Material strategic budget reduction forbidden")
        sum_residual = math.fsum(weights) - 1
    variance = float(weights @ matrix @ weights)
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Positive observed strategic portfolio variance required")
    shares = weights * (matrix @ weights) / variance
    if not np.isfinite(shares).all() or abs(math.fsum(shares) - 1) > 1e-12:
        raise ValueError("Attained risk contribution reconciliation failed")
    return StrategicWeights(
        method,
        assets,
        tuple(map(float, weights)),
        cap,
        tuple(map(float, shares)),
        float(np.max(np.abs(shares - 1 / n))),
        sum_residual,
        cap_residual,
        reduction,
        cap_reduction,
        solver,
        estimate.window_hash,
    )
