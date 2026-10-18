"""Fee-funded cost sensitivity and monetary attribution of fixed ETF controls.

Every scenario executes the same signal targets through the shared ledger.
Costs change actual funded holdings and subsequent wealth, rather than being
subtracted from an already compounded annual return. These scenarios are not a
search over realistic execution costs or a reason to select a more favorable fee.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, date, datetime
from pathlib import Path

from quantlab.allocation import study
from quantlab.allocation.policies import ASSET_IDS, POLICIES
from quantlab.evaluation import attribution

COSTS_BPS = (0.0, 5.0, 10.0, 25.0)


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _sources():
    return {
        **study._sources(),
        "allocation/robustness.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation/attribution.py": hashlib.sha256(
            Path(attribution.__file__).read_bytes()
        ).hexdigest(),
    }


def _target_records(targets):
    return [
        {
            "policy": target.policy,
            "signal_at": target.decision.signal_at.isoformat(),
            "intended_execution_at": target.decision.intended_execution_at.isoformat(),
            "asset_ids": list(target.asset_ids),
            "weights": list(target.weights),
            "cash_weight": target.cash_weight,
            "active": list(target.active),
            "scale": target.scale,
            "forecast_annual_volatility_before": target.annual_volatility_before_scaling,
            "forecast_annual_volatility_after": target.annual_volatility_after_scaling,
            "source_rows_sha256": hashlib.sha256(
                json.dumps(
                    [bar.row_sha256 for bar in target.source_rows], separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        for target in targets
    ]


def compare_costs(
    histories,
    schedule,
    *,
    source_identity,
    start_signal,
    end,
    service_deadline_monotonic=None,
):
    """Run all four fees on identical complete observed support, or fail closed.

    This low-level function supports short real-data engineering checks. The
    publication runner below separately requires the full development window.
    A service deadline may stop between cost scenarios without selecting results
    from the partial set. The cloud service must also enforce a child timeout.
    """
    if service_deadline_monotonic is not None and (
        isinstance(service_deadline_monotonic, bool)
        or not math.isfinite(service_deadline_monotonic)
        or service_deadline_monotonic <= time.monotonic()
    ):
        raise ValueError("A future finite service deadline is required")
    output, reference_targets, reference_support = {}, None, None
    for cost in COSTS_BPS:
        if (
            service_deadline_monotonic is not None
            and time.monotonic() >= service_deadline_monotonic
        ):
            raise TimeoutError("Incomplete cost family; no complete sensitivity result")
        results, targets = study.compare_controls(
            histories,
            schedule,
            start_signal=start_signal,
            end=end,
            source_identity=source_identity,
            fee_bps=cost,
        )
        target_rows = {policy: _target_records(targets[policy]) for policy in POLICIES}
        support = tuple(row.session for row in results[POLICIES[0]]["measured"])
        if reference_targets is None:
            reference_targets, reference_support = target_rows, support
        elif target_rows != reference_targets or support != reference_support:
            raise ValueError("Fee sensitivity must retain identical signals, targets and support")
        scenarios = {}
        for policy in POLICIES:
            result = results[policy]
            owned = attribution.monetary_contributions(result["measured"], initial_nav=100_000.0)
            if owned["sessions"] != len(support):
                raise ValueError("Attribution support differs from the economic comparison")
            scenarios[policy] = {**result, "targets": target_rows[policy], "attribution": owned}
        output[cost] = scenarios
    return output


def run_cost_study(histories, schedule, *, source_identity, output, service_deadline_monotonic):
    """Persist all 16 full-period ledgers, target evidence and owned P&L.

    Run in the authenticated cloud environment, never as a host-side training or
    full backtest shortcut. Caller authenticates the original input partitions.
    """
    if (
        isinstance(service_deadline_monotonic, bool)
        or not isinstance(service_deadline_monotonic, (int, float))
        or not math.isfinite(service_deadline_monotonic)
        or service_deadline_monotonic <= time.monotonic()
    ):
        raise ValueError("Future finite cloud service deadline required before reading inputs")
    if (
        source_identity.source_type != "yahoo_etf_release"
        or source_identity.price_basis != "provider_adjusted_close_ratio_price_index_research_proxy"
    ):
        raise ValueError("Approved distinct actual Yahoo ETF research-price-index source required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = _sources()
    start, end = date(2007, 2, 28), date(2014, 12, 31)
    contract = {
        "schema": "etf_actual_cost_sensitivity_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_identity": source_identity.as_provenance(),
        "source_code_sha256": sources,
        "start_signal": start.isoformat(),
        "end_close": end.isoformat(),
        "cost_bps_per_traded_side": list(COSTS_BPS),
        "initial_cash": 100_000.0,
        "policies": list(POLICIES),
        "cash": "explicit_zero_interest_assumption",
        "scenario_selection": "none_all_four_assumed_fees_reported",
        "interpretation": "research_price_index_proxy_not_observed_execution_cost_or_auction_fills",
        "holdout_evaluated": False,
    }
    _write(output / "contract.json", contract)
    try:
        results = compare_costs(
            histories,
            schedule,
            source_identity=source_identity,
            start_signal=start,
            end=end,
            service_deadline_monotonic=service_deadline_monotonic,
        )
        import pandas as pd

        metrics = {}
        for cost, policies in results.items():
            key = f"cost-{cost:g}bps"
            metrics[key] = {}
            for policy, result in policies.items():
                rows = result["measured"]
                if (
                    len(rows) != 1975
                    or rows[0].session != date(2007, 3, 1)
                    or rows[-1].session != end
                ):
                    raise ValueError("Entire 1,975-session measured support required for every arm")
                prefix = key + "-" + policy
                frame = pd.DataFrame(
                    [
                        {
                            "session": row.session,
                            "close_nav": row.state.nav,
                            "cash": row.state.cash,
                            "fee": row.execution.fee,
                            "pretrade_nav": row.execution.nav_before,
                            "traded_notional": row.execution.traded_notional,
                            "full_turnover": row.execution.full_turnover,
                            "money_pnl": row.money_pnl,
                            "reconciliation_residual": row.reconciliation_residual,
                            **{
                                asset: dict(row.state.values).get(asset, 0.0) for asset in ASSET_IDS
                            },
                        }
                        for row in rows
                    ]
                )
                frame.to_parquet(output / (prefix + ".parquet"), index=False)
                _write(output / (prefix + "-targets.json"), result["targets"])
                _write(output / (prefix + "-attribution.json"), result["attribution"])
                metrics[key][policy] = result["metrics"]
        _write(output / "metrics.json", metrics)
        if _sources() != sources:
            raise ValueError("Scientific source changed during the study")
        if time.monotonic() >= service_deadline_monotonic:
            raise TimeoutError("Service deadline reached before complete result verification")
        inventory = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file()
        }
        _write(
            output / "manifest.json",
            {
                "complete": True,
                "contract": contract,
                "outputs_sha256": inventory,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        (output / "COMPLETE").write_text(
            hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() + "\n"
        )
        return metrics
    except BaseException as error:
        _write(output / "FAILURE.json", {"complete": False, "error_type": type(error).__name__})
        raise
