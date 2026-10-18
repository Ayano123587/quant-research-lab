"""One frozen release: execute, export forecasts, then consume mature outcomes."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import struct
from datetime import date, datetime
from pathlib import Path

from quantlab.evaluation.holdout import END, START, file_sha, run_shared_holdout
from quantlab.evaluation.holdout_outcomes import (
    covariance_outcomes,
    portfolio_outcomes,
    read_prediction_exports,
    stock_outcomes,
)
from quantlab.evaluation.holdout_registry import build_registry
from quantlab.evaluation.holdout_tables import (
    REPORTING_POLICY,
    comparison_intervals,
    comparison_tables,
    daily_tables,
)


def _plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _name(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", value) is None
    ):
        raise ValueError("Safe unique output identifier required")
    return value


def _write(path, value):
    raw = json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "bytes": len(raw), "sha256": file_sha(path)}


def read_stock_outcomes(provider, days, *, scoring_cutoff):
    """Metadata projection precedes all raw label value projections."""
    from quantlab.stocks.development import Outcome

    outcomes, statuses = {}, {}
    for day in days:
        path = provider.partitions["labels", day]
        metadata = provider._read(
            path,
            ["asset_id", "signal_session", "planned_available_at", "label_available_at", "status"],
        )
        if len({r["asset_id"] for r in metadata}) != len(metadata) or any(
            r["signal_session"] != day for r in metadata
        ):
            raise ValueError("Actual label metadata identity differs")
        if any(r["planned_available_at"] is None for r in metadata):
            raise ValueError("Calendar maturity metadata cannot be silently treated as censoring")
        if any(
            r["planned_available_at"] > scoring_cutoff
            or (r["label_available_at"] is not None and r["label_available_at"] > scoring_cutoff)
            for r in metadata
        ):
            outcomes[day], statuses[day] = (), "right_censored"
            continue
        records = provider._read(
            path,
            [
                "asset_id",
                "signal_session",
                "label_available_at",
                "status",
                "return_value",
                "input_rows_sha256",
            ],
        )
        indexed = {r["asset_id"]: r for r in metadata}
        if len(records) != len(indexed) or {r["asset_id"] for r in records} != set(indexed):
            raise ValueError("Outcome value cohort differs from authenticated clocks")
        result = []
        for row in records:
            clock = indexed[row["asset_id"]]
            if (
                row["signal_session"] != day
                or row["status"] != clock["status"]
                or row["label_available_at"] != clock["label_available_at"]
            ):
                raise ValueError("Outcome values changed from the authenticated metadata")
            if row["status"] == "valid":
                if (
                    row["label_available_at"] is None
                    or not isinstance(row["return_value"], (float, int))
                    or not math.isfinite(row["return_value"])
                ):
                    raise ValueError("Valid label requires an actual availability timestamp")
                result.append(
                    Outcome(
                        row["asset_id"],
                        day,
                        row["label_available_at"],
                        row["return_value"],
                        row["input_rows_sha256"],
                    )
                )
        outcomes[day] = tuple(result)
        statuses[day] = "complete" if len(result) == len(metadata) else "incomplete_source"
    return outcomes, statuses


def run_frozen_pipeline(
    release_path,
    expected_sha256,
    *,
    project,
    inputs,
    schedule,
    output,
    dispatch_ticket_path,
    expected_dispatch_ticket_sha256,
):
    """Cloud-only through the shared gate; never chooses recipes or retries."""
    freeze, registry, providers = build_registry(
        release_path, expected_sha256, project=project, inputs=inputs, schedule=schedule
    )
    own = Path(__file__).resolve().relative_to(Path(project).resolve()).as_posix()
    if freeze["source_sha256"].get(own) != file_sha(__file__):
        raise ValueError("Pipeline itself must be bound by the actual external freeze")
    if freeze.get("reporting_policy") != REPORTING_POLICY:
        raise ValueError(
            "Readable tables and fixed uncertainty policy must be frozen before release"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for arm_id, wrapper in providers.items():
        wrapper.export_root = output / "streams" / _name(arm_id)
    artifacts = {}
    try:
        execution = run_shared_holdout(
            release_path,
            expected_sha256,
            project=project,
            inputs=inputs,
            registry=registry,
            schedule=schedule,
            dispatch_ticket_path=dispatch_ticket_path,
            expected_dispatch_ticket_sha256=expected_dispatch_ticket_sha256,
        )
        window, results = execution["window"], execution["results"]
        prediction_exports, forecast_exports, reporting_tables = {}, {}, {}
        for arm_id, wrapper in providers.items():
            _name(arm_id)
            provider = wrapper.provider
            for record in wrapper.stream_exports.values():
                artifacts[str(Path(record["path"]).relative_to(output))] = record
            if arm_id in results:
                path = output / "portfolios" / (arm_id + ".json")
                artifacts[str(path.relative_to(output))] = _write(path, results[arm_id])
                path = output / "economic-sources" / (arm_id + ".json")
                artifacts[str(path.relative_to(output))] = _write(
                    path, execution["economic_source_receipts"][arm_id]
                )
            for field in ("targets", "decisions", "panel_support"):
                if hasattr(provider, field):
                    path = output / "provider-audits" / arm_id / (field + ".json")
                    artifacts[str(path.relative_to(output))] = _write(
                        path, getattr(provider, field)
                    )
            if hasattr(provider, "export_predictions"):
                path = output / "predictions" / (arm_id + ".json")
                path.parent.mkdir(exist_ok=True)
                record = provider.export_predictions(path)
                stock = getattr(provider, "stock", provider)
                prediction_exports[arm_id] = {
                    **record,
                    "candidate_id": stock.candidate.candidate_id,
                }
                artifacts[str(path.relative_to(output))] = record
            if hasattr(provider, "forecast_provider"):
                forecast = provider.forecast_provider
                records = [
                    {"session": day, "reference": forecast.references[day], "estimates": estimates}
                    for day, estimates in sorted(forecast.forecasts.items())
                ]
                path = output / "forecasts" / (arm_id + ".json")
                forecast_exports[arm_id] = _write(path, records)
                artifacts[str(path.relative_to(output))] = forecast_exports[arm_id]
        # The exclusive, fully hashed exports now exist before any outcome reader.
        comparisons, used, supported = {}, set(), True
        unsupported, forecast_support_rows = {}, []
        for arm, wrapper in providers.items():
            if set(wrapper.stream_exports) != set(window.rebalance_signals):
                raise ValueError("Missing full-date forecast stream cannot complete protocol")
            for day, item in wrapper.stream_exports.items():
                if (
                    file_sha(item["path"]) != item["sha256"]
                    or Path(item["path"]).stat().st_size != item["bytes"]
                ):
                    raise ValueError("Forecast support receipt authentication failed")
                record = json.loads(Path(item["path"]).read_bytes())
                if record["session"] != day.isoformat():
                    raise ValueError("Forecast support date differs")
                fit = record.get("fit")
                forecast_support_rows.append(
                    {
                        "arm_id": arm,
                        "signal_session": day,
                        "status": record["status"],
                        "support_code": record.get("support_code"),
                        "decision_at": schedule.session(day).decision_at,
                        "forecast_only": arm in execution["forecast_only_arms"],
                        "fit_id": None if fit is None else fit["fit_id"],
                        "training_rows": None if fit is None else len(fit["used_keys"]),
                        "maximum_feature_available_at": None
                        if fit is None
                        else fit["maximum_feature_available_at"],
                        "fit_cutoff": None if fit is None else fit["cutoff"],
                    }
                )
                if record["status"] == "unsupported_forecast":
                    from quantlab.evaluation.holdout_registry import ForecastSupportError

                    if record["support_code"] not in ForecastSupportError.CODES:
                        raise ValueError("Unknown forecast support code")
                    unsupported[arm, day] = record["support_code"]
                elif record["status"] != "forecast_exported":
                    raise ValueError("Unknown forecast terminal status")
        performance_allowed, mature_statistical_support = {}, {}
        for spec in freeze["comparisons"]:
            name = _name(spec["comparison_id"])
            if name in comparisons:
                raise ValueError("Duplicate frozen comparison")
            arms = tuple(spec["arm_ids"])
            if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(registry):
                raise ValueError("Every comparison arm must be explicitly registered")
            used.update(arms)
            if spec["kind"] == "portfolio":
                if not set(arms) <= set(results):
                    raise ValueError("Forecast-only registers cannot enter economic comparisons")
                result = portfolio_outcomes(
                    {a: results[a] for a in arms},
                    expected_arms=arms,
                    expected_sessions=window.measured_sessions,
                    initial_cash=freeze["initial_cash"],
                )
                supported &= result["full_common_economic_support"]
                performance_allowed[name] = result["full_common_economic_support"]
            elif spec["kind"] == "stock_prediction":
                exports = {a: prediction_exports[a] for a in arms}
                predictions = read_prediction_exports(
                    exports,
                    expected_arms=arms,
                    expected_sessions=window.rebalance_signals,
                    unsupported_forecasts={k: v for k, v in unsupported.items() if k[0] in arms},
                )
                if spec["same_score_bytes"]:
                    for day in window.rebalance_signals:
                        if any((a, day) not in predictions for a in arms):
                            continue
                        reference = [
                            (a, struct.pack("<d", v)) for a, v in predictions[arms[0], day]
                        ]
                        if any(
                            [(a, struct.pack("<d", v)) for a, v in predictions[arm, day]]
                            != reference
                            for arm in arms[1:]
                        ):
                            raise ValueError(
                                "Stock integration changed frozen prediction scores/IDs"
                            )
                source = providers[spec["outcome_arm_id"]].provider
                source = getattr(source, "stock", source)
                if hasattr(source, "prepare_outcomes"):
                    source.prepare_outcomes(cutoff=schedule.session(END).decision_at)
                outcomes, statuses = read_stock_outcomes(
                    source,
                    window.rebalance_signals,
                    scoring_cutoff=schedule.session(END).decision_at,
                )
                result = stock_outcomes(
                    predictions,
                    outcomes,
                    expected_arms=arms,
                    expected_sessions=window.rebalance_signals,
                    minimum_support=spec["minimum_support"],
                    scoring_cutoff=schedule.session(END).decision_at,
                    outcome_status_by_session=statuses,
                    unsupported_forecasts={k: v for k, v in unsupported.items() if k[0] in arms},
                )
                performance_allowed[name] = all(
                    row["status"] == "valid" for row in result["per_arm_date"].values()
                )
                mature = tuple(
                    d for d in window.rebalance_signals if statuses[d] != "right_censored"
                )
                mature_statistical_support[name] = bool(mature) and all(
                    result["per_arm_date"][a, d]["status"] == "valid" for a in arms for d in mature
                )
            elif spec["kind"] == "covariance_proxy":
                if len(arms) != 1:
                    raise ValueError("One dated forecast register contains all paired estimators")
                wrapper = providers[arms[0]]
                forecast = wrapper.provider.forecast_provider
                receipt = forecast_exports[arms[0]]
                if file_sha(receipt["path"]) != receipt["sha256"]:
                    raise ValueError("Durable forecast register changed before outcome read")
                result = covariance_outcomes(
                    forecast.forecasts,
                    forecast.references,
                    wrapper.loaded.histories,
                    schedule,
                    expected_sessions=window.rebalance_signals,
                    expected_estimators=spec["estimator_ids"],
                    registered_pairs=tuple(tuple(p) for p in spec["pairs"]),
                    archive_sha256=wrapper.loaded.source_identity.source_id,
                    archive_endpoint=END,
                    scoring_cutoff=schedule.session(END).decision_at,
                    evaluation_plan_id=name,
                    recipe_frozen_at=datetime.fromisoformat(freeze["approved_at"]),
                    recipe_information_cutoff=datetime.fromisoformat(
                        freeze["historical_information_cutoff"]
                    ),
                    unsupported_forecasts={
                        d: v for (a, d), v in unsupported.items() if a == arms[0]
                    },
                )
                performance_allowed[name] = all(
                    isinstance(rows, tuple) and all(row.status == "scored" for row in rows)
                    for rows in result["rows"].values()
                )
                mature_rows = [
                    rows
                    for rows in result["rows"].values()
                    if not (
                        isinstance(rows, tuple)
                        and rows
                        and all(
                            row.status == "unscored_incomplete_outcome"
                            and row.reason.startswith("right_censored:")
                            for row in rows
                        )
                    )
                ]
                mature_statistical_support[name] = bool(mature_rows) and all(
                    isinstance(rows, tuple) and all(row.status == "scored" for row in rows)
                    for rows in mature_rows
                )
            else:
                raise ValueError("Unknown comparison cannot silently complete")
            comparisons[name] = result
            path = output / "comparisons" / (name + ".json")
            artifacts[str(path.relative_to(output))] = _write(path, result)
            for table_name, rows in comparison_tables(spec, result).items():
                key = "tables/" + name + "/" + table_name + ".json"
                reporting_tables[key] = _write(output / key, rows)
                artifacts[key] = reporting_tables[key]
            key = "tables/" + name + "/paired_intervals.json"
            reporting_tables[key] = _write(
                output / key, comparison_intervals(spec, result, results)
            )
            artifacts[key] = reporting_tables[key]
        for key, rows in (
            ("tables/forecast_support.json", forecast_support_rows),
            ("tables/portfolio_daily.json", daily_tables(results, window.measured_sessions)),
        ):
            reporting_tables[key] = _write(output / key, rows)
            artifacts[key] = reporting_tables[key]
        for wrapper in providers.values():
            source = getattr(wrapper.provider, "stock", wrapper.provider)
            for item in (
                *getattr(source, "generated", {}).values(),
                *getattr(source, "eligibility_artifacts", {}).values(),
            ):
                path = Path(item["path"])
                relative = str(path.resolve().relative_to(output.resolve()))
                if relative in artifacts and artifacts[relative] != item:
                    raise ValueError("Generated scientific output identity collision")
                artifacts[relative] = item
        if used != set(registry):
            raise ValueError("Registered arms omitted from outcome comparisons")
        if {a.component for a in registry.values()} != {
            "stocks",
            "etf",
            "covariance",
            "integrations",
        }:
            raise ValueError("Every declared scientific component is required")
        # Recheck the actual immutable recipe/input/source/runtime after outcomes.
        from quantlab.evaluation.holdout import _authenticate

        _authenticate(release_path, expected_sha256, project, inputs, registry, schedule)
        for record in artifacts.values():
            if (
                Path(record["path"]).stat().st_size != record["bytes"]
                or file_sha(record["path"]) != record["sha256"]
            ):
                raise ValueError("Saved scientific output changed before completion")
        manifest = {
            "schema": "shared_quant_frozen_evaluation_v1",
            "release_sha256": expected_sha256,
            "registered_comparisons_terminal": True,
            "evaluation_attempts_terminal": True,
            "source_and_protocol_authentication_passed": True,
            "full_economic_common_support": bool(supported),
            "full_mature_statistical_support": all(mature_statistical_support.values()),
            "mature_paired_performance_allowed": mature_statistical_support,
            "full_period_performance_allowed": performance_allowed,
            "global_protocol_complete": True,
            "all_economic_support_complete": bool(supported),
            "global_holdout_complete": False,
            "forecast_only_arms": execution["forecast_only_arms"],
            "reporting_policy": REPORTING_POLICY,
            "reporting_tables": reporting_tables,
            "declared_scope": {
                "signal_dates": len(window.rebalance_signals),
                "measured_sessions": len(window.measured_sessions),
                "covariance_forecast_cases": sum(
                    s["kind"] == "covariance_proxy" for s in freeze["comparisons"]
                ),
                "risk_economic_paths": sum(
                    r["component"] == "covariance" for r in freeze["portfolio_arms"]
                ),
            },
            "comparisons": tuple(comparisons),
            "artifacts": artifacts,
            "selection_performed": False,
            "initial_signal": START,
            "terminal_close": END,
        }
        _write(output / "manifest.json", manifest)
        _write(
            output / "COMPLETE",
            {
                "manifest_sha256": file_sha(output / "manifest.json"),
                "scope": "registered_evaluation_protocol_not_universal_performance_support",
                "global_protocol_complete": True,
                "full_economic_common_support": bool(supported),
            },
        )
        return manifest
    except Exception as error:
        _write(
            output / "FAILURE.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "reason": str(error),
                "release_sha256": expected_sha256,
                "artifacts_preserved": artifacts,
                "automatic_retry": False,
            },
        )
        raise
