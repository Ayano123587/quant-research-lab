"""Controlled development-period ETF comparison using the shared actual ledger.

This runner refuses final-period strategy evaluation. Final evaluation requires
a separate joint recipe freeze, including the stock and covariance studies.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

from quantlab.allocation.policies import ASSET_IDS, POLICIES, SYMBOLS, allocate
from quantlab.backtest.engine import Decision, run_history
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import identity_from_source, resolve_source_identity


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources():
    package = Path(__file__).resolve().parents[1]
    names = (
        "allocation/study.py",
        "allocation/policies.py",
        "backtest/engine.py",
        "backtest/accounting.py",
        "data/archive.py",
        "data/calendar.py",
        "data/snapshot.py",
        "data/ingest.py",
        "data/yahoo.py",
        "data/provenance.py",
        "risk/covariance.py",
    )
    return {name: _sha(package / name) for name in names}


def _metrics(records, initial_cash):
    if len(records) < 2:
        raise ValueError("At least two measured sessions required")
    previous = float(initial_cash)
    peak = previous
    returns, drawdowns = [], []
    for record in records:
        nav = record.state.nav
        returns.append(nav / previous - 1)
        peak = max(peak, nav)
        drawdowns.append(nav / peak - 1)
        previous = nav
    volatility = stdev(returns) * math.sqrt(252)
    return {
        "measured_sessions": len(records),
        "net_return": previous / initial_cash - 1,
        "annual_return_252_session_convention": (previous / initial_cash) ** (252 / len(records))
        - 1,
        "annual_volatility_sample_ddof1": volatility,
        "annual_sharpe_zero_cash_reference": mean(returns) * 252 / volatility
        if volatility > 0
        else None,
        "maximum_close_nav_drawdown": min(drawdowns),
        "drawdown_basis": "measured_session_closes_plus_initial_cash_no_intraday_extrema",
        "total_fee_money": math.fsum(r.execution.fee for r in records),
        "total_full_buy_plus_sell_turnover": math.fsum(r.execution.full_turnover for r in records),
        "mean_close_risky_weight": mean(
            math.fsum(v for _, v in r.state.values) / r.state.nav for r in records
        ),
        "maximum_money_reconciliation_residual": max(
            abs(r.reconciliation_residual) for r in records
        ),
    }


def compare_controls(
    histories,
    schedule,
    *,
    start_signal,
    end,
    archive_sha256=None,
    source_identity=None,
    initial_cash=100000.0,
    fee_bps=10.0,
):
    """Low-level real-history development comparison, without implicit selection.

    Callers authenticate histories. Start is an explicit monthly decision date,
    with cash initialized at its close; measurement starts at the next open.
    Every policy uses the same dates, source prices, cost and funding conventions.
    No partial valid prefix receives a complete-comparison metric.
    """
    if not isinstance(schedule, XNYSSchedule):
        raise ValueError("Actual XNYS reference schedule required")
    source_identity = resolve_source_identity(
        schedule.provenance["schedule_sha256"],
        source_identity=source_identity,
        archive_sha256=archive_sha256,
    )
    if start_signal >= date(2015, 1, 1) or end >= date(2015, 1, 1):
        raise ValueError("Development runner cannot score the shared final period")
    if start_signal >= end:
        raise ValueError("Development end must follow the initial monthly signal")
    if schedule.month_end(start_signal.year, start_signal.month).session != start_signal:
        raise ValueError("Start must be the complete month-end decision session")
    if set(histories) != set(ASSET_IDS):
        raise ValueError("Exactly the eight prespecified source sleeves required")
    results, targets = {}, {}
    for policy in POLICIES:
        saved = []

        def decide(view, _state, policy=policy, saved=saved):
            if schedule.month_end(view.session.year, view.session.month).session != view.session:
                return Decision("HOLD", None, view.cutoff, view.next_open_at, "between_month_ends")
            target = allocate(view, schedule, policy=policy, source_identity=source_identity)
            saved.append(target)
            return target.decision

        result = run_history(
            histories,
            schedule,
            start=start_signal,
            end=end,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            cash_policy="explicit_zero_interest_assumption",
            decision_callback=decide,
            source_identity=source_identity,
        )
        if result.failure or result.incomplete_day:
            raise ValueError("Incomplete economic support; no complete ETF comparison permitted")
        if result.daily[0].state.nav != initial_cash or result.daily[0].execution.fee != 0:
            raise ValueError("Initial decision close must contain cash only")
        measured = result.daily[1:]
        expected_first = schedule.next_session(start_signal).session
        if not measured or measured[0].session != expected_first:
            raise ValueError("Initial execution measurement boundary differs")
        results[policy] = {
            "engine": result,
            "measured": measured,
            "metrics": _metrics(measured, initial_cash),
        }
        targets[policy] = tuple(saved)
    supports = {tuple(r.session for r in result["measured"]) for result in results.values()}
    if len(supports) != 1:
        raise ValueError("All four policies require identical economic observation support")
    return results, targets


def run_development(
    snapshot, schedule, output, *, start_signal, end, initial_cash=100000.0, fee_bps=10.0
):
    """Write immutable, authenticated four-control development research outputs."""
    source_identity = identity_from_source(snapshot, schedule)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = _sources()
    contract = {
        "phase": "development_only_not_final_selection_or_test",
        "source_manifest_sha256": snapshot.manifest_sha256,
        "source_snapshot_id": source_identity.source_id,
        "archive_sha256": source_identity.archive_sha256,
        "source_identity": source_identity.as_dict(),
        "calendar": schedule.provenance,
        "source_code_sha256": sources,
        "start_signal": str(start_signal),
        "end_close": str(end),
        "measurement_start": str(schedule.next_session(start_signal).session),
        "initial_cash": initial_cash,
        "fee_bps_per_traded_side": fee_bps,
        "cash": "explicit_zero_interest_assumption",
        "symbols": SYMBOLS,
        "policies": POLICIES,
        "terminal": "observed_close_mark_no_forced_liquidation_no_pending_trade_charge",
        "runtime": {
            name: version(name)
            for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "exchange-calendars")
        },
    }

    def write_json(name, value):
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with (output / name).open("xb") as stream:
            stream.write(raw)

    write_json("contract.json", contract)
    try:
        histories = {
            asset: snapshot.bars("etf", symbol)
            for asset, symbol in zip(ASSET_IDS, SYMBOLS, strict=True)
        }
        results, targets = compare_controls(
            histories,
            schedule,
            start_signal=start_signal,
            end=end,
            source_identity=source_identity,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
        )
        import pyarrow as pa
        import pyarrow.parquet as pq

        for policy, result in results.items():
            rows = []
            for record in result["measured"]:
                rows.append(
                    {
                        "session": record.session,
                        "close_nav": record.state.nav,
                        "cash": record.state.cash,
                        "fee": record.execution.fee,
                        "pretrade_nav": record.execution.nav_before,
                        "traded_notional": record.execution.traded_notional,
                        "full_turnover": record.execution.full_turnover,
                        "money_pnl": record.money_pnl,
                        "reconciliation_residual": record.reconciliation_residual,
                        **dict(record.state.values),
                    }
                )
            pq.write_table(pa.Table.from_pylist(rows), output / (policy + ".parquet"))
            decisions = []
            for target in targets[policy]:
                identities = [
                    (b.asset_id, str(b.session), b.source_sha256, b.row_sha256)
                    for b in target.source_rows
                ]
                decisions.append(
                    {
                        "signal_at": target.decision.signal_at.isoformat(),
                        "intended_execution_at": target.decision.intended_execution_at.isoformat(),
                        "action": target.decision.action,
                        "reason": target.decision.reason,
                        "weights": dict(target.decision.target),
                        "active_sleeves": dict(zip(target.asset_ids, target.active, strict=True)),
                        "max_input_available_at": target.max_input_available_at.isoformat(),
                        "annual_volatility_before_scaling": target.annual_volatility_before_scaling,
                        "annual_volatility_after_scaling": target.annual_volatility_after_scaling,
                        "volatility_status": target.volatility_status,
                        "covariance_method": target.covariance.method
                        if target.covariance
                        else None,
                        "covariance_configuration_hash": target.covariance.configuration_hash
                        if target.covariance
                        else None,
                        "cash_weight": target.cash_weight,
                        "trend_returns": target.trend_returns,
                        "scale": target.scale,
                        "daily_covariance_window_hash": target.covariance.window_hash
                        if target.covariance
                        else None,
                        "input_row_identity_sha256": hashlib.sha256(
                            json.dumps(identities, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "input_rows": len(identities),
                        "source_identity": target.source_identity.as_dict(),
                        "execution_within_requested_period": (
                            target.decision.intended_execution_at.date() <= end
                        ),
                    }
                )
            write_json(policy + "-monthly-targets.json", decisions)
            write_json(
                policy + "-decision-events.json",
                [
                    {
                        "action": d.action,
                        "reason": d.reason,
                        "signal_at": d.signal_at.isoformat(),
                        "intended_execution_at": d.intended_execution_at.isoformat(),
                        "target_weights": dict(d.target) if d.target is not None else None,
                        "execution_within_requested_period": d.intended_execution_at.date() <= end,
                    }
                    for d in result["engine"].decisions
                ],
            )
        if _sources() != sources:
            raise ValueError("Research source changed during execution")
        for asset in ASSET_IDS:
            snapshot.rows(asset)  # Re-authenticate each actual input partition after execution.
        if identity_from_source(snapshot, schedule) != source_identity:
            raise ValueError("Authenticated source identity changed during study")
        write_json("metrics.json", {p: r["metrics"] for p, r in results.items()})
        outputs = {p.name: _sha(p) for p in output.iterdir() if p.is_file()}
        manifest = {
            "status": "complete",
            "complete": True,
            "contract": contract,
            "outputs_sha256": outputs,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        write_json("manifest.json", manifest)
        (output / "COMPLETE").write_bytes((_sha(output / "manifest.json") + "\n").encode())
        return manifest
    except BaseException as error:
        write_json("FAILURE.json", {"status": "failed", "error_type": type(error).__name__})
        raise
