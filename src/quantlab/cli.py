"""Source audit, immutable normalization and real-event accounting checks."""

import argparse
import json
from datetime import date
from pathlib import Path

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Verify a real source and report its coverage")
    inspect.add_argument("--archive", required=True)
    inspect.add_argument("--archive-sha256", default=SUPPLIED_ARCHIVE_SHA256)
    inspect.add_argument("--asset-type", choices=("stock", "etf"), required=True)
    inspect.add_argument("--symbol", required=True)
    audit = commands.add_parser("audit", help="Audit every supplied archive member")
    audit.add_argument("--archive", required=True)
    audit.add_argument("--archive-sha256", default=SUPPLIED_ARCHIVE_SHA256)
    audit.add_argument("--output", required=True)
    normalize = commands.add_parser("normalize", help="Preserve the full archive as Parquet")
    normalize.add_argument("--archive", required=True)
    normalize.add_argument("--archive-sha256", default=SUPPLIED_ARCHIVE_SHA256)
    normalize.add_argument("--audit-summary", required=True)
    normalize.add_argument("--audit-summary-sha256", required=True)
    normalize.add_argument("--file-summaries", required=True)
    normalize.add_argument("--file-summaries-sha256", required=True)
    normalize.add_argument("--output", required=True)
    replay = commands.add_parser("replay", help="Reconcile one real ETF decision")
    replay.add_argument("--archive", required=True)
    replay.add_argument("--archive-sha256", default=SUPPLIED_ARCHIVE_SHA256)
    replay.add_argument("--decision-session", type=date.fromisoformat, required=True)
    replay.add_argument("--initial-cash", type=float, default=100000.0)
    replay.add_argument("--fee-bps", type=float, default=10.0)
    replay.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        from quantlab.data.audit import audit_archive

        result = audit_archive(args.archive, args.output, args.archive_sha256)
        print(json.dumps({"files": result["primary_files"], "rows": result["primary_rows"]}))
        return
    if args.command == "normalize":
        from quantlab.data.calendar import XNYSSchedule
        from quantlab.data.ingest import ingest_archive

        schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
        result = ingest_archive(
            args.archive,
            args.output,
            archive_sha256=args.archive_sha256,
            audit_summary_path=args.audit_summary,
            audit_summary_sha256=args.audit_summary_sha256,
            file_summaries_path=args.file_summaries,
            file_summaries_sha256=args.file_summaries_sha256,
            schedule=schedule,
        )
        print(json.dumps({"complete": result["complete"], "counts": result["counts"]}))
        return
    if args.command == "replay":
        from quantlab.backtest.replay import replay_one_decision
        from quantlab.data.calendar import XNYSSchedule
        from quantlab.data.returns import trailing_returns

        output = Path(args.output)
        if output.exists():
            parser.error("Use a fresh replay output path")
        schedule = XNYSSchedule(date(2005, 1, 1), date(2018, 12, 31))
        signals, targets = [], {}
        with MarketArchive(args.archive, args.archive_sha256) as archive:
            for symbol in ("SPY", "IEF"):
                bars = archive.bars("etf", symbol)
                history = trailing_returns(
                    bars,
                    schedule,
                    args.decision_session,
                    252,
                    archive_endpoint=bars[-1].session,
                )
                by_date = {bar.session: bar for bar in bars}
                trend = (
                    by_date[args.decision_session].close / by_date[history[0].start_at.date()].close
                    - 1
                )
                targets[bars[0].asset_id] = 0.5 if trend > 0 else 0.0
                signals.append({"asset_id": bars[0].asset_id, "trend_252": trend})
            result = replay_one_decision(
                archive,
                schedule,
                initial_units={},
                initial_cash=args.initial_cash,
                target_weights=targets,
                fee_bps=args.fee_bps,
                action="REBALANCE",
                cash_policy="explicit_zero_interest_assumption",
                decision_session=args.decision_session,
            )
        result["signal_derivation"] = signals
        result["target_origin"] = "Past-only 252-session positive-trend mask, fixed half budgets"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "reconciliation_residual": result["reconciliation_residual"],
                }
            )
        )
        return
    with MarketArchive(args.archive, args.archive_sha256) as archive:
        bars = archive.bars(args.asset_type, args.symbol)
        result = {
            "archive_sha256": archive.sha256,
            "rows": len(bars),
            "asset_id": bars[0].asset_id if bars else None,
            "first_session": str(bars[0].session) if bars else None,
            "last_session": str(bars[-1].session) if bars else None,
            "source_sha256": bars[0].source_sha256 if bars else None,
            "quality_findings": sum(bool(bar.quality_flags) for bar in bars),
        }
    print(json.dumps(result, indent=2, allow_nan=False))
