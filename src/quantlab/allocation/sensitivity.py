"""One-change ETF risk and target experiments on the existing causal signal view."""

import math
from dataclasses import dataclass

from quantlab.allocation.policies import ASSET_IDS, allocate
from quantlab.backtest.engine import Decision


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    covariance_method: str
    annual_target: float
    weighting: str = "equal_capital"


# A comparison graph, not a covariance-by-target Cartesian search. The reference
# is repeated unchanged so its actual ledgers can be checked against the earlier run.
SCENARIOS = (
    Scenario("reference_sample_target10", "sample_ddof1", 0.10),
    Scenario("target05_only", "sample_ddof1", 0.05),
    Scenario("target15_only", "sample_ddof1", 0.15),
    Scenario("covariance_ledoit_wolf_only", "ledoit_wolf", 0.10),
    Scenario("covariance_ewma63_only", "ewma", 0.10),
    Scenario("weight_inverse_vol_only", "sample_ddof1", 0.10, "capped_inverse_volatility"),
    Scenario("weight_risk_budget_only", "sample_ddof1", 0.10, "capped_risk_budget"),
)


@dataclass(frozen=True)
class SensitivityTarget:
    scenario: Scenario
    strategic_weights: tuple[float, ...]
    active: tuple[bool, ...]
    forecast_before: float
    forecast_after: float
    scale: float
    weights: tuple[float, ...]
    cash_weight: float
    status: str
    source_target: object
    decision: Decision
    strategic_certificate: object | None = None


def sensitivity_target(view, schedule, source_identity, scenario):
    """Build target using only past prices; preserve trend masks and cash budgets."""
    if scenario not in SCENARIOS:
        raise ValueError("Declared one-change scenario required")
    original = allocate(
        view,
        schedule,
        policy="trend_252_volatility_63",
        source_identity=source_identity,
        covariance_method=scenario.covariance_method,
    )
    strategic = (1 / len(ASSET_IDS),) * len(ASSET_IDS)
    certificate = None
    if scenario.weighting != "equal_capital":
        from quantlab.allocation.weighting import strategic_weights

        certificate = strategic_weights(original.covariance, scenario.weighting)
        strategic = certificate.weights
        active_budget = tuple(
            budget if active else 0.0
            for budget, active in zip(strategic, original.active, strict=True)
        )
        before = math.sqrt(252 * original.covariance.portfolio_variance(active_budget, ASSET_IDS))
    else:
        before = original.annual_volatility_before_scaling
    if before is None or not math.isfinite(before) or before < 0:
        raise ValueError("Observed finite nonnegative signal risk required")
    scale = min(1.0, scenario.annual_target / before) if before > 0 else 1.0
    weights = tuple(
        budget * scale if active else 0.0
        for budget, active in zip(strategic, original.active, strict=True)
    )
    cash = 1 - math.fsum(weights)
    if cash < 0 or not all(math.isfinite(x) and x >= 0 for x in weights):
        raise ValueError("No-leverage funded target required")
    after = before * scale
    if after > scenario.annual_target + 1e-14:
        raise ValueError("Forecast risk cap violated")
    status = (
        "all_cash"
        if not any(original.active)
        else "zero_variance_no_leverage"
        if before == 0
        else "scaled_down"
        if scale < 1
        else "unscaled_below_target"
    )
    decision = Decision(
        "REBALANCE",
        tuple(zip(ASSET_IDS, weights, strict=True)),
        view.cutoff,
        view.next_open_at,
        scenario.scenario_id,
    )
    if scenario == SCENARIOS[0] and (
        weights != original.weights
        or scale != original.scale
        or cash != original.cash_weight
        or after != original.annual_volatility_after_scaling
    ):
        raise ValueError("Reference target differs from frozen four-control policy")
    return SensitivityTarget(
        scenario,
        strategic,
        original.active,
        before,
        after,
        scale,
        weights,
        cash,
        status,
        original,
        decision,
        certificate,
    )
