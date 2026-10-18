"""Execute authenticated saved stock rankings on the shared real-price ledger.

This is development research with opening-price proxies, full fills and declared
proportional fees. Missing held marks stop a path; no terminal price, delisting
return, corporate action or cash substitute is invented.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import re
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

from quantlab.backtest.engine import Decision, run_history
from quantlab.data.provenance import identity_from_source
from quantlab.stocks.development import registered_candidates
from quantlab.stocks.prepare import SOURCE_MANIFEST_SHA256, verify_prepared_panel

PREPARED_SHA = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
ARCHIVE_SHA = "d9317c8fb2d63b9b00db5f933b6c9639d2bf7ea3b918169bb5cec5903dce85a1"


def encoded(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=lambda v: v.isoformat(),
        )
        + "\n"
    ).encode()


def content_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_inventory(root):
    return [
        {
            "path": p.relative_to(root).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": file_sha(p),
        }
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and p.name not in ("manifest.json", "COMPLETE")
    ]


def economic_source_closure():
    """Bind every local static import, including imports inside function bodies."""
    src = Path(__file__).resolve().parents[2]
    pending, found = [Path(__file__).resolve()], {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(src).as_posix()
        if relative in found:
            continue
        found[relative] = file_sha(path)
        for parent in path.parents:
            if parent == src:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer != path:
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = path.relative_to(src).with_suffix("").parts[:-1]
                    base = ".".join(
                        (*parts[: len(parts) - node.level + 1], base)
                    ).strip(".")
                modules = [base, *(base + "." + a.name for a in node.names)]
            for module in modules:
                if not module.startswith("quantlab"):
                    continue
                target = src.joinpath(*module.split("."))
                for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                    if candidate.is_file():
                        pending.append(candidate)
    return dict(sorted(found.items()))


def economic_runtime():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in (
                "numpy",
                "pyarrow",
                "scipy",
                "scikit-learn",
                "pandas",
                "exchange-calendars",
            )
        },
    }


def completion_budget_guard(manifest, service_deadline_monotonic):
    """Postflight time is part of the service budget, including inventory hashing."""
    if time.monotonic() >= service_deadline_monotonic:
        manifest.update(
            complete=False,
            status="INCOMPLETE_SERVICE_BUDGET",
            metrics=None,
            continuation="no_automatic_resume_completed_paths_preserved",
        )
    return manifest["complete"]


def safe_file(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("Unsafe or missing authenticated stock artifact")
    return path


def aware(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Explicit aware historical prediction timestamp required")
    return result


@dataclass(frozen=True)
class PredictionStudy:
    root: Path
    manifest_sha256: str
    manifest: dict
    artifacts: dict
    fit_records: dict
    candidates: tuple
    signal_sessions: tuple[date, ...]


@dataclass(frozen=True)
class PredictionSignal:
    candidate_id: str
    session: date
    signal_at: datetime
    execution_at: datetime
    scores: tuple[tuple[str, float], ...]
    fit_id: str
    prediction_file_sha256: str


def authenticate_prediction_study(root, expected_manifest_sha256, schedule):
    """Hash all 96 saved prediction partitions before Arrow sees their values.

    Authenticate the original completed source identity; intentionally do not
    demand that today's later rank-target implementation equal the older fit.
    No learned model payload or outcome-value column is read.
    """
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("External actual study manifest SHA required")
    if file_sha(safe_file(root, "manifest.json")) != expected_manifest_sha256:
        raise ValueError("Prediction study manifest differs")
    if (root / "COMPLETE").read_bytes() != (expected_manifest_sha256 + "\n").encode():
        raise ValueError("Actual completed prediction study required")
    if (root / "FAILURE.json").exists():
        raise ValueError("Failed prediction study is not an economic input")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        manifest.get("schema") != "stock_prequential_development_v1"
        or manifest.get("complete") is not True
        or manifest.get("status") != "complete"
        or manifest.get("errors") != []
        or manifest.get("final_holdout_evaluation") is not False
        or manifest.get("source_prepared_manifest_sha256") != PREPARED_SHA
        or content_sha(manifest["source_closure_sha256"])
        != manifest["source_code_sha256"]
    ):
        raise ValueError("Full actual development prediction/source contract required")
    candidates = registered_candidates(manifest["spec"])
    if len(candidates) != 16 or manifest["candidate_ids"] != [
        c.candidate_id for c in candidates
    ]:
        raise ValueError("All sixteen fixed candidates required")
    sessions = tuple(
        schedule.month_end(y, m).session
        for y in range(2007, 2015)
        for m in range(1, 13)
    )
    artifacts = {x["path"]: x for x in manifest["artifacts"]}
    if len(artifacts) != len(manifest["artifacts"]):
        raise ValueError("Duplicate original artifact path")
    expected = {"predictions/" + d.isoformat() + ".parquet" for d in sessions}
    if {n for n in artifacts if n.startswith("predictions/")} != expected:
        raise ValueError("Exactly all 96 monthly prediction partitions required")
    for name in sorted(expected | {"fit-index.json"}):
        path = safe_file(root, name)
        item = artifacts[name]
        if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
            raise ValueError("Saved prediction/fit metadata raw bytes differ")
    fits = json.loads((root / "fit-index.json").read_bytes())
    indexed = {
        (date.fromisoformat(r["signal_session"]), r["candidate_id"]): r for r in fits
    }
    if (
        len(indexed) != 1536
        or len(fits) != 1536
        or set(indexed)
        != {(day, c.candidate_id) for day in sessions for c in candidates}
    ):
        raise ValueError("All 1536 completed candidate/date fit receipts required")
    for day in sessions:
        cutoff = schedule.session(day).decision_at
        for candidate in candidates:
            record = indexed[day, candidate.candidate_id]
            if record["status"] not in ("fitted", "fixed_control_no_fit"):
                raise ValueError(
                    "Only actual completed fitted/control predictions admitted"
                )
            expected_fit = content_sha(
                {
                    "candidate": asdict(candidate),
                    "training": record["training_input_sha256"],
                    "signal_session": day,
                    "spec_sha256": manifest["spec_sha256"],
                    "source_code_sha256": manifest["source_code_sha256"],
                    "runtime": manifest["runtime"],
                }
            )
            if (
                record["fit_id"] != expected_fit
                or aware(record["information_cutoff"]) != cutoff
                or encoded(record["parameters"]) != encoded(candidate.parameters)
                or record["max_label_available_at"] is not None
                and aware(record["max_label_available_at"]) > cutoff
                or any(date.fromisoformat(d) >= day for d, _ in record["date_counts"])
            ):
                raise ValueError("Saved fit identity or historical maturity differs")
    return PredictionStudy(
        root,
        expected_manifest_sha256,
        manifest,
        artifacts,
        indexed,
        candidates,
        sessions,
    )


def read_prediction_signal(study, day, schedule, prepared_root):
    """Read one authenticated date's features and scores, never label values."""
    import pyarrow.parquet as pq

    if day not in study.signal_sessions:
        raise ValueError("Unregistered prediction signal date")
    relative = "predictions/" + day.isoformat() + ".parquet"
    path = safe_file(study.root, relative)
    if file_sha(path) != study.artifacts[relative]["sha256"]:
        raise ValueError("Prediction bytes changed before parsing")
    prepared_manifest = json.loads((Path(prepared_root) / "manifest.json").read_bytes())
    if file_sha(Path(prepared_root) / "manifest.json") != PREPARED_SHA:
        raise ValueError("Actual full prepared panel manifest differs")
    feature_name = "features/" + day.isoformat() + ".parquet"
    item = next(x for x in prepared_manifest["outputs"] if x["path"] == feature_name)
    feature_path = safe_file(prepared_root, feature_name)
    if (
        feature_path.stat().st_size != item["bytes"]
        or file_sha(feature_path) != item["sha256"]
    ):
        raise ValueError("Actual prepared feature bytes differ")
    features = (
        pq.ParquetFile(feature_path)
        .read(
            columns=[
                "asset_id",
                "signal_session",
                "signal_at",
                "intended_execution_at",
                "max_input_available_at",
                "input_rows_sha256",
            ]
        )
        .to_pylist()
    )
    cutoff = schedule.session(day).decision_at
    execution = schedule.next_session(day).open_at
    indexed = {r["asset_id"]: r for r in features}
    if len(indexed) != len(features) or any(
        r["signal_session"] != day
        or r["signal_at"] != cutoff
        or r["intended_execution_at"] != execution
        or r["max_input_available_at"] > cutoff
        for r in features
    ):
        raise ValueError("Saved feature population/clock differs")
    rows = pq.ParquetFile(path).read().to_pylist()
    result = {}
    for candidate in study.candidates:
        record = study.fit_records[day, candidate.candidate_id]
        selected = [r for r in rows if r["candidate_id"] == candidate.candidate_id]
        ids = [r["asset_id"] for r in selected]
        expected_ids = (
            set(indexed)
            if record["status"] in ("fixed_control_no_fit", "fitted")
            else set()
        )
        if len(set(ids)) != len(ids) or set(ids) != expected_ids:
            raise ValueError(
                "Prediction population does not match actual feature support"
            )
        for row in selected:
            if (
                row["fit_id"] != record["fit_id"]
                or row["signal_session"] != day
                or row["signal_at"] != cutoff
                or not math.isfinite(row["score"])
                or row["feature_input_sha256"]
                != indexed[row["asset_id"]]["input_rows_sha256"]
                or not re.fullmatch(r"stock:[a-z0-9][a-z0-9._^-]*\.us", row["asset_id"])
            ):
                raise ValueError("Prediction fit/clock/source provenance differs")
        result[candidate.candidate_id] = PredictionSignal(
            candidate.candidate_id,
            day,
            cutoff,
            execution,
            tuple((r["asset_id"], float(r["score"])) for r in selected),
            record["fit_id"],
            study.artifacts[relative]["sha256"],
        )
    if len(rows) != sum(len(s.scores) for s in result.values()):
        raise ValueError("Unknown candidate rows in saved predictions")
    return result


def signal_decision(signal, positions=50, cap=0.03):
    """Saved scores are ordinal only; never interpret them as expected returns."""
    if positions != 50 or cap != 0.03:
        raise ValueError("Registered top50/equal-weight/cap policy required")
    if len(signal.scores) < positions:
        return Decision(
            "HOLD",
            None,
            signal.signal_at,
            signal.execution_at,
            "insufficient_saved_prediction_support_preserve_positions_and_cash",
        )
    selected = sorted(signal.scores, key=lambda pair: (-pair[1], pair[0]))[:positions]
    return Decision(
        "REBALANCE",
        tuple((asset, 1 / positions) for asset, _ in selected),
        signal.signal_at,
        signal.execution_at,
        "saved_ordinal_top50_source_id_ties",
    )


def execute_candidate_window(
    histories,
    schedule,
    signals,
    *,
    start_signal,
    end,
    initial_cash,
    fee_bps,
    source_identity,
    budget_callback=None,
):
    """Real-history engineering entrypoint; full study uses all fixed cost paths."""

    def decide(view, _state):
        if budget_callback is not None:
            budget_callback(view, _state)
        signal = signals.get(view.session)
        if signal is None:
            if (
                schedule.month_end(view.session.year, view.session.month).session
                == view.session
            ):
                raise ValueError("Missing registered monthly saved prediction signal")
            return Decision(
                "HOLD", None, view.cutoff, view.next_open_at, "between_signal_dates"
            )
        if signal.signal_at != view.cutoff or signal.execution_at != view.next_open_at:
            raise ValueError("Saved score decision clock changed")
        return signal_decision(signal)

    return run_history(
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


def full_metrics(records, initial_cash):
    if len(records) < 2 or not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Complete positive-capital multi-session ledger required")
    previous = peak = initial_cash
    returns, drawdowns = [], []
    for record in records:
        nav = record.state.nav
        returns.append(nav / previous - 1)
        peak = max(peak, nav)
        drawdowns.append(nav / peak - 1)
        previous = nav
    volatility = stdev(returns) * math.sqrt(252)
    return {
        "sessions": len(records),
        "net_return": previous / initial_cash - 1,
        "annual_return_252_session_convention": (previous / initial_cash)
        ** (252 / len(records))
        - 1,
        "annual_volatility_sample_ddof1": volatility,
        "annual_sharpe_zero_cash_reference": (
            mean(returns) * 252 / volatility if volatility else None
        ),
        "maximum_close_nav_drawdown": min(drawdowns),
        "fee_money": math.fsum(r.execution.fee for r in records),
        "full_buy_plus_sell_turnover": math.fsum(
            r.execution.full_turnover for r in records
        ),
        "maximum_money_reconciliation_residual": max(
            abs(r.reconciliation_residual) for r in records
        ),
    }


def run_economic_study(
    study_root,
    snapshot,
    prepared_root,
    schedule,
    config_path,
    output,
    *,
    service_deadline_monotonic,
):
    """Full64-path historical execution is restricted to the guarded cloud runtime."""
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError(
            "Full stock economic study runs only in reviewed Kaggle runtime"
        )
    if not math.isfinite(service_deadline_monotonic):
        raise ValueError(
            "Startup-inclusive externally supplied finite deadline required"
        )

    def budget_check():
        if time.monotonic() >= service_deadline_monotonic:
            raise TimeoutError("Startup-inclusive economic service budget exhausted")

    budget_check()
    config = tomllib.loads(Path(config_path).read_text())
    policies = {
        "schema_version": 1,
        "phase": "development_only_no_holdout_or_cost_based_selection",
        "measurement_first": "2007-03-01",
        "authenticated_signal_months": 96,
        "economic_signal_months": 95,
        "outside_window_signal": "2007-01-31_authenticated_not_executed",
        "ranking": "saved_prediction_descending_source_asset_id_ties_ordinal_only",
        "missing_marks": "preserve_actual_prefix_and_unresolved_owners_no_fill_no_terminal_return",
        "cash": "explicit_zero_interest_assumption",
        "price_basis": "source_OHLC_adjustment_unverified",
        "fills": "full_at_observed_scheduled_open_research_proxy",
        "terminal": "observed_close_no_liquidation_final_pending_target_unexecuted",
        "comparison": "all16_candidates_all4_costs_same1975_sessions_or_no_complete_metrics",
    }
    if any(config.get(key) != value for key, value in policies.items()):
        raise ValueError(
            "Declared economic policy differs from implemented fixed semantics"
        )
    config_sha = file_sha(config_path)
    source_sha = file_sha(__file__)
    source_closure, runtime = economic_source_closure(), economic_runtime()
    if (
        config["initial_signal"] != "2007-02-28"
        or config["end_close"] != "2014-12-31"
        or config["fee_bps"] != [0, 5, 10, 25]
        or config["positions"] != 50
        or config["single_asset_cap"] != 0.03
        or config["expected_measured_sessions"] != 1975
    ):
        raise ValueError("Registered common economic dates/cost lattice required")
    if not math.isfinite(config["initial_cash"]) or config["initial_cash"] <= 0:
        raise ValueError("Explicit positive finite initial cash required")
    study = authenticate_prediction_study(
        study_root, config["prediction_study_manifest_sha256"], schedule
    )
    budget_check()
    verify_prepared_panel(prepared_root, PREPARED_SHA)
    budget_check()
    identity = identity_from_source(snapshot, schedule)
    if (
        identity.manifest_sha256 != SOURCE_MANIFEST_SHA256
        or identity.archive_sha256 != ARCHIVE_SHA
    ):
        raise ValueError("Original full7195-stock source snapshot required")
    start, end = date(2007, 2, 28), date(2014, 12, 31)
    measured_sessions = tuple(d for d in schedule.sessions if start < d <= end)
    if len(measured_sessions) != 1975:
        raise ValueError("Exact common1975-session economic cohort required")
    signals = {c.candidate_id: {} for c in study.candidates}
    for day in study.signal_sessions:
        current = read_prediction_signal(study, day, schedule, prepared_root)
        if day >= start:
            for candidate, signal in current.items():
                signals[candidate][day] = signal
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "stock_economic_development_v1",
        "complete": False,
        "prediction_manifest_sha256": study.manifest_sha256,
        "prepared_manifest_sha256": PREPARED_SHA,
        "source_identity": identity.as_dict(),
        "calendar": schedule.provenance,
        "config_sha256": config_sha,
        "economic_source_sha256": source_sha,
        "economic_source_closure_sha256": source_closure,
        "economic_runtime": runtime,
        "candidate_ids": list(signals),
        "authenticated_signal_months": 96,
        "economic_signal_months": 95,
        "excluded_signal": "2007-01-31 outside registered common economic window",
        "initial_signal": str(start),
        "measurement_first": str(measured_sessions[0]),
        "end_close": str(end),
        "fee_bps": config["fee_bps"],
        "initial_cash": config["initial_cash"],
        "cash_policy": "explicit_zero_interest_assumption",
        "full_fills_assumed": True,
        "terminal": "observed_close_no_forced_liquidation_final_target_right_censored",
        "scores_are_ordinals_only": True,
        "no_cost_based_winner_selection": True,
        "final_holdout_evaluated": False,
        "paths": [],
    }
    summaries, supports = {}, []
    with snapshot.verified_read_session():
        for candidate, monthly in signals.items():
            assets = sorted(
                {
                    asset
                    for signal in monthly.values()
                    for asset, _ in (signal_decision(signal).target or ())
                }
            )
            if not assets:
                raise ValueError(
                    "No selected actual source histories; no invented cash-path anchor"
                )
            histories = {
                asset: snapshot.bars("stock", asset.split(":", 1)[1])
                for asset in assets
            }
            for fee in config["fee_bps"]:
                path_id = candidate + "__fee_" + str(fee)

                def decision_budget(view, state, path_id=path_id):
                    # This is a real reached close, not a completed-path ledger.
                    if time.monotonic() >= service_deadline_monotonic:
                        (output / "interrupted-close.json").write_bytes(
                            encoded(
                                {
                                    "path_id": path_id,
                                    "session": view.session,
                                    "state": asdict(state),
                                    "status": "unexecuted_decision_budget_stop",
                                    "not_a_complete_path_or_resume_cursor": True,
                                }
                            )
                        )
                    budget_check()

                try:
                    budget_check()
                    result = execute_candidate_window(
                        histories,
                        schedule,
                        monthly,
                        start_signal=start,
                        end=end,
                        initial_cash=config["initial_cash"],
                        fee_bps=fee,
                        source_identity=identity,
                        budget_callback=decision_budget,
                    )
                except TimeoutError:
                    if (
                        economic_source_closure() != source_closure
                        or economic_runtime() != runtime
                        or file_sha(config_path) != config_sha
                    ):
                        raise ValueError(
                            "Execution source/runtime/config changed at budget stop"
                        ) from None
                    manifest.update(
                        status="INCOMPLETE_SERVICE_BUDGET",
                        complete=False,
                        interrupted_path=path_id,
                        metrics=None,
                        finished_at=datetime.now(UTC),
                        continuation="no_automatic_resume_completed_paths_preserved",
                        artifacts=output_inventory(output),
                    )
                    (output / "manifest.json").write_bytes(encoded(manifest))
                    return manifest
                if result.daily and (
                    result.daily[0].session != start
                    or result.daily[0].state.nav != config["initial_cash"]
                    or result.daily[0].execution.fee != 0
                    or result.daily[0].state.values
                ):
                    raise ValueError(
                        "Initial signal close must retain uninvested initial cash"
                    )
                records = result.daily[1:]
                complete = (
                    result.failure is None
                    and result.incomplete_day is None
                    and tuple(r.session for r in records) == measured_sessions
                )
                path_id = candidate + "__fee_" + str(fee)
                folder = output / path_id
                folder.mkdir()
                with (folder / "money-ledger.jsonl").open("xb") as stream:
                    for record in result.daily:
                        stream.write(
                            encoded(asdict(record)).replace(b"\n", b" ") + b"\n"
                        )
                receipt = {
                    "candidate_id": candidate,
                    "fee_bps": fee,
                    "economic_support": (
                        "COMPLETE" if complete else "UNSUPPORTED_VALID_PREFIX"
                    ),
                    "valid_measured_sessions": len(records),
                    "failure": asdict(result.failure) if result.failure else None,
                    "incomplete_opening": (
                        asdict(result.incomplete_day) if result.incomplete_day else None
                    ),
                    "endpoint_status": result.endpoint_status,
                    "pending_status": result.pending_status,
                    "pending_decision": (
                        asdict(result.pending) if result.pending else None
                    ),
                    "complete_metrics": None,
                }
                (folder / "path-status.json").write_bytes(encoded(receipt))
                supports.append(complete)
                if complete:
                    summaries[path_id] = full_metrics(records, config["initial_cash"])
                manifest["paths"].append({"path_id": path_id, **receipt})
    # A complete economic comparison never silently drops an unsupported arm.
    all_complete = len(supports) == 64 and all(supports)
    manifest.update(
        complete=all_complete,
        status=(
            "complete_all64_common_paths"
            if all_complete
            else "unsupported_full_comparison"
        ),
        metrics=summaries if all_complete else None,
        finished_at=datetime.now(UTC),
    )
    if file_sha(study.root / "manifest.json") != study.manifest_sha256:
        raise ValueError("Prediction study changed during economic execution")
    authenticate_prediction_study(study.root, study.manifest_sha256, schedule)
    if file_sha(config_path) != config_sha or file_sha(__file__) != source_sha:
        raise ValueError("Economic configuration/source changed during execution")
    if economic_source_closure() != source_closure or economic_runtime() != runtime:
        raise ValueError("Economic transitive execution source/runtime changed")
    verify_prepared_panel(prepared_root, PREPARED_SHA)
    manifest["artifacts"] = output_inventory(output)
    all_complete = completion_budget_guard(manifest, service_deadline_monotonic)
    (output / "manifest.json").write_bytes(encoded(manifest))
    if all_complete:
        (output / "COMPLETE").write_text(file_sha(output / "manifest.json") + "\n")
    return manifest
