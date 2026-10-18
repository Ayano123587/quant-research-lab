"""Fee-funded long-only money accounting at one declared valuation timestamp.

Inputs are already marked money values, not forecasts or desired historical
weights. Full fills and one proportional charge on each traded side are explicit
research assumptions; this is not an exchange execution or corporate-action model.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass


def _money(values, label):
    if not isinstance(values, Mapping):
        raise ValueError(label + " requires an asset-keyed mapping")
    result = {}
    for asset, value in values.items():
        if not isinstance(asset, str) or not asset:
            raise ValueError("Nonempty asset IDs required")
        if isinstance(value, bool):
            raise ValueError(label + " must contain finite nonnegative amounts")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(label + " must contain finite nonnegative amounts")
        result[asset] = value
    return result


def _cash(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("Cash must be finite and nonnegative")
    return float(value)


def _total(values, cash):
    value = math.fsum([*values.values(), cash])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Finite positive portfolio NAV required")
    return value


@dataclass(frozen=True)
class Trade:
    asset_id: str
    value_before: float
    target_weight: float | None
    value_after: float
    signed_notional: float
    fee: float


@dataclass(frozen=True)
class Execution:
    action: str
    nav_before: float
    nav_after: float
    cash_before: float
    cash_after: float
    fee_bps: float
    fee: float
    traded_notional: float
    full_turnover: float
    legs: tuple[Trade, ...]
    reconciliation_residual: float

    @property
    def positions(self):
        return {leg.asset_id: leg.value_after for leg in self.legs}


def rebalance(marked_values, cash, target_weights, fee_bps):
    """Solve V+ + c*sum_old_union_target(abs(w_i*V+ - x_i)) = V-.

    Risky weights are fractions of post-fee NAV. Their remainder stays in cash.
    Missing target assets mean liquidation, not disappearance. Turnover is the
    full absolute buy-plus-sell notional divided by drifted pretrade NAV.
    """
    values = _money(marked_values, "Marked holdings")
    weights = _money(target_weights, "Target weights")
    weight_sum = math.fsum(weights.values())
    if weight_sum > 1:
        raise ValueError("Long-only target weights must sum to at most one")
    cash = _cash(cash)
    before = _total(values, cash)
    if (
        isinstance(fee_bps, bool)
        or not math.isfinite(float(fee_bps))
        or not 0 <= float(fee_bps) < 10000
    ):
        raise ValueError("Fee bps must lie in [0, 10000)")
    rate = float(fee_bps) / 10000
    assets = sorted(values.keys() | weights.keys())

    def notional(nav):
        return math.fsum(abs(weights.get(a, 0.0) * nav - values.get(a, 0.0)) for a in assets)

    if rate == 0 or notional(before) == 0:
        after = before
    else:
        low, high = 0.0, before
        for _ in range(100):
            middle = (low + high) / 2
            if middle + rate * notional(middle) > before:
                high = middle
            else:
                low = middle
        after = (low + high) / 2
    legs = tuple(
        Trade(
            a,
            values.get(a, 0.0),
            weights.get(a, 0.0),
            weights.get(a, 0.0) * after,
            weights.get(a, 0.0) * after - values.get(a, 0.0),
            rate * abs(weights.get(a, 0.0) * after - values.get(a, 0.0)),
        )
        for a in assets
    )
    fee = math.fsum(leg.fee for leg in legs)
    traded = math.fsum(abs(leg.signed_notional) for leg in legs)
    new_cash = (1 - weight_sum) * after
    residual = math.fsum([after, fee, -before])
    tolerance = max(1.0, before) * 2e-12
    if (
        not all(math.isfinite(x) for x in (after, fee, traded, new_cash))
        or new_cash < 0
        or abs(residual) > tolerance
        or abs(math.fsum([*(leg.value_after for leg in legs), new_cash, -after])) > tolerance
        or abs(math.fsum([cash, -math.fsum(leg.signed_notional for leg in legs), -fee, -new_cash]))
        > tolerance
    ):
        raise ValueError("Money-level fee funding reconciliation failed")
    return Execution(
        "REBALANCE",
        before,
        after,
        cash,
        new_cash,
        float(fee_bps),
        fee,
        traded,
        traded / before,
        legs,
        residual,
    )


def hold(marked_values, cash):
    """Explicit HOLD preserves actual drifted holdings and never reapplies targets."""
    values, cash = _money(marked_values, "Marked holdings"), _cash(cash)
    nav = _total(values, cash)
    legs = tuple(Trade(a, v, None, v, 0.0, 0.0) for a, v in sorted(values.items()))
    return Execution("HOLD", nav, nav, cash, cash, 0.0, 0.0, 0.0, 0.0, legs, 0.0)


@dataclass(frozen=True)
class MarkedPortfolio:
    values: tuple[tuple[str, float], ...]
    cash: float
    nav_before: float
    nav_after: float
    asset_pnl: tuple[tuple[str, float], ...]
    cash_pnl: float
    reconciliation_residual: float

    @property
    def positions(self):
        return dict(self.values)


def mark_positions(values, cash, start_prices, end_prices, cash_return):
    """Value existing owners over a specified interval using positive real marks.

    A missing held-asset mark refuses the whole interval. Zero holdings require
    no economic mark. Cash return must be supplied as an explicit assumption or
    independently sourced rate; it is never inferred from these price files.
    """
    values, cash = _money(values, "Held values"), _cash(cash)
    before = _total(values, cash)
    if not isinstance(start_prices, Mapping) or not isinstance(end_prices, Mapping):
        raise ValueError("Asset-keyed start/end marks required")
    if (
        isinstance(cash_return, bool)
        or not math.isfinite(float(cash_return))
        or float(cash_return) < -1
    ):
        raise ValueError("Finite cash return of at least -1 required")
    marked, contributions = {}, {}
    for asset, value in values.items():
        if value == 0:
            marked[asset], contributions[asset] = 0.0, 0.0
            continue
        if asset not in start_prices or asset not in end_prices:
            raise ValueError("UNRESOLVED_HELD_ASSET: missing interval mark for " + asset)
        a, b = float(start_prices[asset]), float(end_prices[asset])
        if not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0:
            raise ValueError("UNRESOLVED_HELD_ASSET: invalid interval mark for " + asset)
        marked[asset] = value * (b / a)
        contributions[asset] = marked[asset] - value
    new_cash = cash * (1 + float(cash_return))
    after = math.fsum([*marked.values(), new_cash])
    cash_pnl = new_cash - cash
    residual = math.fsum([after, -before, -math.fsum(contributions.values()), -cash_pnl])
    if not math.isfinite(after) or abs(residual) > max(1.0, before) * 2e-12:
        raise ValueError("Marked-ownership reconciliation failed")
    return MarkedPortfolio(
        tuple(sorted(marked.items())),
        new_cash,
        before,
        after,
        tuple(sorted(contributions.items())),
        cash_pnl,
        residual,
    )
