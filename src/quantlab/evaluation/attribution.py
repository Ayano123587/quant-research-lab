"""Additive monetary contributions from the actual event ledger.

Dollar contributions reconcile; sums of standalone compounded asset returns do
not. Overnight P&L belongs to positions before the opening trade, while intraday
P&L belongs to positions after that trade. Fees are a separate negative component.
"""

from __future__ import annotations

import math
from collections import defaultdict

from quantlab.backtest.engine import DailyRecord


def monetary_contributions(records, *, initial_nav):
    """Attribute a complete consecutive ledger prefix in money and initial-NAV units.

    This calculation consumes saved ownership events; it does not infer positions
    from final weights, recompute prices, or impute missing outcomes. A caller must
    separately establish full requested economic support before publishing a full
    study result. A reconciled prefix alone cannot establish full-period coverage.
    """
    rows = tuple(records)
    if (
        not rows
        or isinstance(initial_nav, bool)
        or not math.isfinite(initial_nav)
        or initial_nav <= 0
        or any(not isinstance(row, DailyRecord) for row in rows)
        or any(a.session >= b.session for a, b in zip(rows, rows[1:], strict=False))
    ):
        raise ValueError("Ordered actual daily records and positive finite initial NAV required")
    previous = float(initial_nav)
    by_asset = defaultdict(list)
    cash_parts, fee_parts, daily = [], [], []
    for row in rows:
        tolerance = max(1.0, abs(previous)) * 4e-12
        if abs(row.overnight.nav_before - previous) > tolerance:
            raise ValueError("Ledger ownership chain has an omitted or inconsistent NAV interval")
        asset_parts = defaultdict(list)
        for event in (row.overnight, row.intraday):
            for asset, pnl in event.asset_pnl:
                if not isinstance(asset, str) or not asset or not math.isfinite(pnl):
                    raise ValueError("Finite identified asset P&L required")
                asset_parts[asset].append(pnl)
        assets = {asset: math.fsum(values) for asset, values in sorted(asset_parts.items())}
        cash = math.fsum((row.overnight.cash_pnl, row.intraday.cash_pnl))
        fee = -row.execution.fee
        delta = row.state.nav - previous
        residual = delta - math.fsum((*assets.values(), cash, fee))
        if (
            not all(math.isfinite(v) for v in (cash, fee, delta, residual))
            or fee > 0
            or abs(residual) > tolerance
            or abs(row.money_pnl - delta) > tolerance
        ):
            raise ValueError("Actual monetary ownership components fail NAV reconciliation")
        for asset, pnl in assets.items():
            by_asset[asset].append(pnl)
        cash_parts.append(cash)
        fee_parts.append(fee)
        daily.append(
            {
                "session": row.session.isoformat(),
                "asset_pnl": assets,
                "cash_pnl": cash,
                "fee_pnl": fee,
                "net_pnl": delta,
                "reconciliation_residual": residual,
            }
        )
        previous = row.state.nav
    assets = {asset: math.fsum(parts) for asset, parts in sorted(by_asset.items())}
    cash, fees = math.fsum(cash_parts), math.fsum(fee_parts)
    net = previous - initial_nav
    residual = net - math.fsum((*assets.values(), cash, fees))
    if abs(residual) > max(1.0, initial_nav, abs(previous)) * 4e-12 * len(rows):
        raise ValueError("Cumulative monetary attribution does not reconcile")
    return {
        "schema": "event_owned_monetary_attribution_v1",
        "basis": "additive_money_and_initial_NAV_units_not_compounded_return_attribution",
        "initial_nav": initial_nav,
        "final_nav": previous,
        "sessions": len(rows),
        "first_session": rows[0].session.isoformat(),
        "last_session": rows[-1].session.isoformat(),
        "asset_pnl": assets,
        "cash_pnl": cash,
        "fee_pnl": fees,
        "net_pnl": net,
        "initial_nav_contributions": {
            **{a: p / initial_nav for a, p in assets.items()},
            "cash": cash / initial_nav,
            "fees": fees / initial_nav,
        },
        "reconciliation_residual": residual,
        "daily": daily,
        "requested_period_completeness": "caller_must_verify_requested_support_separately",
    }
