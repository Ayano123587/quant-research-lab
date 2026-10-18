"""Prospective roundoff-only execution adapter for the two genuine baselines.

This is separate from minimum-variance solver projection. Baselines retain their
own method names and have no fabricated optimizer attempts or optimality claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from quantlab.risk import execution, optimization


@dataclass(frozen=True)
class BaselineProjection:
    method: str
    asset_ids: tuple[str, ...]
    caps: tuple[float, ...]
    raw_weights: tuple[float, ...]
    applied_weights: tuple[float, ...]
    adjustments: tuple[tuple[str, float, tuple[str, ...]], ...]
    raw_daily_variance: float
    applied_daily_variance: float
    raw_daily_objective: float
    applied_daily_objective: float
    cash_remainder_weight: float
    total_l1_adjustment: float
    matrix_sha256: str
    raw_solution_configuration_hash: str
    policy_sha256: str
    registered_at: str
    created_at: str
    solver_attempts: tuple = ()
    role: str = "baseline_roundoff_only_no_solver_or_matrix_repair"

    @property
    def targets(self):
        return tuple(zip(self.asset_ids, self.applied_weights, strict=True))

    def to_record(self):
        return asdict(self)


def project_baseline(solution, reference, *, policy_path, expected_policy_sha256):
    """Validate the actual deterministic baseline, then apply registered tiny bounds."""
    created = datetime.now(UTC)
    policy, registered = execution._policy(policy_path, expected_policy_sha256, created)
    ids, matrix, _ = optimization._covariance(reference)
    if reference.method != "sample_ddof1":
        raise ValueError("Both baseline diagnostics require the fixed actual sample reference")
    caps_by_asset = dict(zip(ids, solution.caps, strict=True))
    if solution.method == "equal_weight":
        actual = optimization.equal_weight(ids, caps_by_asset)
    elif solution.method == "capped_inverse_volatility_not_risk_parity":
        actual = optimization.capped_inverse_volatility(reference, caps_by_asset)
    else:
        raise ValueError("Only explicit equal-weight/inverse-volatility baselines accepted")
    if (
        solution.to_record() != actual.to_record()
        or solution.attempts
        or solution.status != "valid_raw_target"
        or solution.asset_ids != ids
    ):
        raise ValueError("Baseline method/order/cap/raw provenance differs from actual calculation")
    tolerance, raw, caps = policy["tolerance"], solution.weights, solution.caps
    matrix_sha = hashlib.sha256(matrix.astype("<f8", copy=False).tobytes()).hexdigest()
    applied, reasons = list(raw), [[] for _ in raw]
    residuals = optimization._residuals(raw, caps)
    if any(
        residuals[key] > tolerance
        for key in ("sum_residual", "nonnegative_residual", "cap_residual")
    ):
        raise ValueError("Material baseline violation; no economic fallback")
    for i, (weight, cap) in enumerate(zip(raw, caps, strict=True)):
        if weight < 0:
            applied[i] = 0.0
            reasons[i].append("clip_tiny_negative_within_registered_tolerance")
        elif weight > cap:
            applied[i] = cap
            reasons[i].append("clip_tiny_cap_overshoot_within_registered_tolerance")
    if math.fsum(applied) > 1:
        i = min((j for j, w in enumerate(applied) if w > 0), key=lambda j: (-applied[j], ids[j]))
        applied[i] -= math.fsum(applied) - 1
        reasons[i].append("reduce_largest_positive_tie_asset_id_by_budget_excess")
        steps = 0
        while math.fsum(applied) > 1:
            if steps >= 1024 or applied[i] <= 0:
                raise ValueError("Baseline budget roundoff could not be resolved")
            applied[i] = math.nextafter(applied[i], 0)
            reasons[i].append("nextafter_toward_zero_for_exact_budget")
            steps += 1
    changes = tuple(new - old for old, new in zip(raw, applied, strict=True))
    l1, cash = math.fsum(map(abs, changes)), 1 - math.fsum(applied)
    if (
        l1 > tolerance
        or not 0 <= cash <= tolerance
        or any(
            not math.isfinite(w) or not 0 <= w <= cap for w, cap in zip(applied, caps, strict=True)
        )
    ):
        raise ValueError("Baseline projection exceeds registered roundoff allowance")
    before, after = optimization._variance(raw, matrix), optimization._variance(applied, matrix)
    if hashlib.sha256(matrix.astype("<f8", copy=False).tobytes()).hexdigest() != matrix_sha:
        raise ValueError("Baseline covariance changed during projection")
    return BaselineProjection(
        solution.method,
        ids,
        caps,
        raw,
        tuple(applied),
        tuple(
            (asset, change, tuple(why))
            for asset, change, why in zip(ids, changes, reasons, strict=True)
        ),
        before,
        after,
        before,
        after,
        cash,
        l1,
        matrix_sha,
        solution.configuration_hash,
        expected_policy_sha256,
        registered.isoformat(),
        created.isoformat(),
    )
