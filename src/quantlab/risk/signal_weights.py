"""Prospective, audited representation correction for actual funded holdings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

POLICY = {
    "policy_id": "funded_signal_close_ratio_roundoff_v1",
    "registered_at": "2026-10-17T14:54:16+00:00",
    "maximum_l1_adjustment": 1e-8,
    "nav_reconciliation_ulps": 8,
    "correction": "reduce_largest_positive_tie_asset_id_then_nextafter_no_topup",
    "scope": "old_weights_optimizer_turnover_penalty_basis_only_actual_holdings_unchanged",
}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SignalWeights:
    raw_values: tuple[tuple[str, float], ...]
    nav: float
    cash: float
    nav_reconciliation_residual: float
    raw_weights: tuple[tuple[str, float], ...]
    raw_sum: float
    applied_weights: tuple[tuple[str, float], ...]
    applied_sum: float
    adjustments: tuple[tuple[str, float], ...]
    total_l1_adjustment: float
    policy_sha256: str

    def to_record(self):
        return asdict(self)


def project_signal_weights(holdings):
    """Never normalize a material funding error or change actual economic positions."""
    values = tuple(holdings.values)
    nav, cash = holdings.nav, holdings.cash
    if (
        not math.isfinite(nav) or nav <= 0 or not math.isfinite(cash) or cash < 0
        or len({asset for asset, _ in values}) != len(values)
        or any(not math.isfinite(value) or value < 0 for _, value in values)
    ):
        raise ValueError("Actual finite nonnegative funded holdings required")
    residual = math.fsum([cash, *(value for _, value in values)]) - nav
    if abs(residual) > POLICY["nav_reconciliation_ulps"] * math.ulp(nav):
        raise ValueError("Actual holdings fail NAV reconciliation; no representation correction")
    raw = tuple((asset, value / nav) for asset, value in values)
    raw_sum = math.fsum(weight for _, weight in raw)
    applied = dict(raw)
    if raw_sum > 1:
        if raw_sum - 1 > POLICY["maximum_l1_adjustment"]:
            raise ValueError("Material risky funding excess cannot be corrected")
        asset = min(applied, key=lambda key: (-applied[key], key))
        applied[asset] -= raw_sum - 1
        steps = 0
        while math.fsum(applied.values()) > 1:
            if steps >= 1024 or applied[asset] <= 0:
                raise ValueError("Funded ratio roundoff could not be resolved")
            applied[asset] = math.nextafter(applied[asset], 0)
            steps += 1
    changes = tuple((asset, applied[asset] - weight) for asset, weight in raw)
    l1 = math.fsum(abs(change) for _, change in changes)
    if l1 > POLICY["maximum_l1_adjustment"] or any(change > 0 for _, change in changes):
        raise ValueError("Signal weight correction exceeded one-sided registered bound")
    return SignalWeights(values, nav, cash, residual, raw, raw_sum, tuple(applied.items()),
                         math.fsum(applied.values()), changes, l1, POLICY_SHA256)
