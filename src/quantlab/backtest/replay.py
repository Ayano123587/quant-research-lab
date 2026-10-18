"""One fixed-target decision replay using only authenticated real ETF source rows.

This is an accounting trace, not a strategy backtest or a profitability result.
The source's modern OHLC adjustment/publication conventions remain unverified;
opens and closes are declared research proxies, never claimed auction fills.
"""

import math
from dataclasses import asdict
from datetime import date

from quantlab.backtest.accounting import hold, mark_positions, rebalance
from quantlab.data.archive import MarketArchive

SYMBOLS = ("SPY", "IEF")


def _timing(schedule, day):
    timing = schedule.session(day)
    if timing.session != day:
        raise ValueError("Calendar session identity differs")
    for value in (timing.open_at, timing.close_at, timing.decision_at):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timezone-aware actual exchange timing required")
    if not timing.open_at < timing.close_at <= timing.decision_at:
        raise ValueError("Actual exchange event ordering differs")
    return timing


def _row(bar):
    return {
        "asset_id": bar.asset_id,
        "session": bar.session.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "open_interest": bar.open_interest,
        "source_member": bar.source_member,
        "source_sha256": bar.source_sha256,
        "row_number": bar.row_number,
        "row_sha256": bar.row_sha256,
        "quality_flags": list(bar.quality_flags),
    }


def replay_one_decision(
    source,
    schedule,
    *,
    initial_units,
    initial_cash,
    target_weights,
    fee_bps,
    action,
    cash_policy,
    decision_session=None,
):
    """Replay one externally fixed allocation at the next scheduled XNYS open.

    initial_units and initial_cash are declared starting-capital assumptions,
    not estimated market observations. Target weights and costs are supplied
    externally, not inferred or selected using this event's eventual prices.
    """
    if not isinstance(source, MarketArchive):
        raise ValueError("Authenticated real MarketArchive handle required")
    if cash_policy != "explicit_zero_interest_assumption":
        raise ValueError("Declare the zero-interest cash convention explicitly")
    if (
        isinstance(fee_bps, bool)
        or not math.isfinite(float(fee_bps))
        or not 0 <= float(fee_bps) < 10000
    ):
        raise ValueError("Explicit finite one-way cost assumption in [0, 10000) bps required")
    if action not in {"HOLD", "REBALANCE"}:
        raise ValueError("Explicit HOLD or REBALANCE required")
    if action == "HOLD" and target_weights is not None:
        raise ValueError("HOLD must not resubmit target weights")
    if action == "REBALANCE" and target_weights is None:
        raise ValueError("REBALANCE requires an externally fixed target")
    sessions = tuple(schedule.sessions)
    if (
        not sessions
        or any(type(s) is not date for s in sessions)
        or sessions != tuple(sorted(set(sessions)))
    ):
        raise ValueError("Ordered unique actual exchange sessions required")
    provenance = schedule.provenance
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Actual exchange schedule provenance required")
    histories = {}
    for symbol in SYMBOLS:
        bars = source.bars("etf", symbol)
        if not bars:
            raise ValueError("Missing real requested ETF source")
        histories[bars[0].asset_id] = {bar.session: bar for bar in bars}
    assets = set(histories)
    if not hasattr(initial_units, "items") or set(initial_units) - assets:
        raise ValueError("Starting units must identify the requested real ETFs")
    if any(isinstance(v, bool) for v in initial_units.values()):
        raise ValueError("Initial units must be numeric amounts, not booleans")
    units = {a: float(v) for a, v in initial_units.items()}
    if any(not math.isfinite(v) or v < 0 for v in units.values()):
        raise ValueError("Finite nonnegative initial units required")
    if target_weights is not None and (
        not hasattr(target_weights, "items") or set(target_weights) - assets
    ):
        raise ValueError("Fixed target must identify the requested real ETFs")
    if decision_session is None:
        # Availability-only rule; no prices, profits or future outcome scores select the event.
        candidates = [
            s for s in sessions[:-1] if all(s in history for history in histories.values())
        ]
        if not candidates:
            raise ValueError("No observed common decision session in supplied calendar")
        decision_session = candidates[0]
        selection = (
            "earliest supplied XNYS session with both actual signal rows and a scheduled successor"
        )
    else:
        selection = "externally fixed decision session"
    if decision_session not in sessions:
        raise ValueError("Decision session absent from actual calendar")
    position = sessions.index(decision_session)
    if position + 1 >= len(sessions):
        raise ValueError("No complete scheduled execution interval in calendar")
    execution_session = sessions[position + 1]
    signal_timing, execution_timing = (
        _timing(schedule, decision_session),
        _timing(schedule, execution_session),
    )
    if schedule.next_session(decision_session).session != execution_session:
        raise ValueError("Actual scheduled successor differs")
    if signal_timing.decision_at >= execution_timing.open_at:
        raise ValueError("Fixed decision must precede scheduled execution")
    signal, execution = {}, {}
    for asset, history in histories.items():
        if decision_session not in history:
            raise ValueError("Missing fixed signal row: " + asset)
        if execution_session not in history:
            raise ValueError("MISSING_EXECUTION_MARK: no next-observed-row substitution: " + asset)
        signal[asset], execution[asset] = (
            history[decision_session],
            history[execution_session],
        )
        if signal[asset].quality_flags or execution[asset].quality_flags:
            raise ValueError("SOURCE_QUALITY_FLAG: cannot use flagged event rows: " + asset)
    close_marks = {a: b.close for a, b in signal.items()}
    open_marks = {a: b.open for a, b in execution.items()}
    final_marks = {a: b.close for a, b in execution.items()}
    old_values = {a: value * close_marks[a] for a, value in units.items()}
    overnight = mark_positions(old_values, initial_cash, close_marks, open_marks, cash_return=0.0)
    trade = (
        hold(overnight.positions, overnight.cash)
        if action == "HOLD"
        else rebalance(overnight.positions, overnight.cash, target_weights, fee_bps)
    )
    intraday = mark_positions(
        trade.positions, trade.cash_after, open_marks, final_marks, cash_return=0.0
    )
    total_pnl = intraday.nav_after - overnight.nav_before
    explained = math.fsum(
        [
            *(v for _, v in overnight.asset_pnl),
            overnight.cash_pnl,
            -trade.fee,
            *(v for _, v in intraday.asset_pnl),
            intraday.cash_pnl,
        ]
    )
    residual = total_pnl - explained
    if abs(residual) > max(1.0, overnight.nav_before) * 4e-12:
        raise ValueError("Full event ownership/fee reconciliation failed")
    return {
        "schema": "real_etf_one_decision_accounting_v1",
        "scope": "Single observed event accounting replay; not a strategy performance result",
        "archive_sha256": source.sha256,
        "calendar": provenance,
        "selection_rule": selection,
        "signal_session": decision_session.isoformat(),
        "execution_session": execution_session.isoformat(),
        "signal_at": signal_timing.decision_at.isoformat(),
        "execution_at": execution_timing.open_at.isoformat(),
        "valuation_at": execution_timing.close_at.isoformat(),
        "availability_assumption": (
            "Completed daily bar available at the schedule's declared after-close decision; "
            "provider publication vintage is not audited"
        ),
        "price_assumption": (
            "Same-source OHLC research proxies; adjustment consistency, distributions "
            "and auction execution not established"
        ),
        "assumptions": {
            "initial_units": units,
            "initial_cash": float(initial_cash),
            "target_weights": (dict(target_weights) if target_weights is not None else None),
            "one_way_fee_bps": float(fee_bps),
            "cash_policy": cash_policy,
            "fill_policy": "full fill of fixed postcost-NAV target at next scheduled opening proxy",
            "action": action,
        },
        "signal_rows": [_row(signal[a]) for a in sorted(assets)],
        "execution_rows": [_row(execution[a]) for a in sorted(assets)],
        "overnight_existing_owners": asdict(overnight),
        "execution": asdict(trade),
        "intraday_posttrade_owners": asdict(intraday),
        "total_money_pnl": total_pnl,
        "explained_money_pnl": explained,
        "reconciliation_residual": residual,
        "future_outcomes_used_by_replay_for_target": False,
        "target_origin": (
            "Externally supplied fixed assumption; historical target derivation is not audited"
        ),
        "model_training": "not_applicable",
        "benchmark_or_profitability_claim": False,
    }
