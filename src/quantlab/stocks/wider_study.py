"""Prospective fixed-model wider-feature experiment; native fits only on Kaggle.

This source is not an executed study. The main cloud notebook invokes these
stages visibly. A fresh six-column reference must reproduce all96 original
float64 predictions before comparative output can be scientifically complete.
"""

import ast
import inspect
import math
import platform
import struct
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from quantlab.data.archive import file_sha256
from quantlab.stocks import development as d
from quantlab.stocks.ablation import (
    CANDIDATE_ID,
    REFERENCE_SHA,
    authenticate_reference,
    score_bytes,
)
from quantlab.stocks.feature_reproduction import DIAGNOSTIC_SHA256, POLICY_SHA256, WHITELIST_SHA256
from quantlab.stocks.prepare_wider import ARMS, OPTIONAL_ADDITIONS
from quantlab.stocks.training_targets import RAW, training_targets
from quantlab.stocks.wider_panel import PARENT_SHA, code_pins, verify_wider_panel


def load_spec(path):
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "feature_reproduction": {
            "policy_sha256": POLICY_SHA256,
            "diagnostic_sha256": DIAGNOSTIC_SHA256,
            "whitelist_sha256": WHITELIST_SHA256,
            "parent_platform": "Windows",
        },
        "parent_prepared_manifest_sha256": PARENT_SHA,
        "reference_manifest_sha256": REFERENCE_SHA,
        "candidate_id": CANDIDATE_ID,
        "arms": [a for a, _ in ARMS],
        "model_columns": [6, 16, 15, 36],
        "required_prediction_months": 96,
        "required_paired_mature_months": 94,
        "target_transform": RAW,
        "optional_missing": "raw_null_preserved_model_rank_neutral_zero_plus_explicit_indicator",
        "feature_eligibility": (
            "unchanged_original_six_feature_selected_cohorts_all_prior_training_dates"
        ),
        "reference_gate": "all96_asset_keys_raw_training_sha_and_float64_score_bytes_and_IC_exact",
        "final_holdout_evaluation": False,
        "resume_supported": False,
        "status": "prospective_post_observation_development_not_final_holdout",
    }
    if spec != required:
        raise ValueError("Exactly the prospective four-arm fixed RAW protocol required")
    return spec


def project_rows(rows, arm_id):
    """Project genuine added columns, with explicit conditional missing flags."""
    ids = dict(ARMS)[arm_id]
    optional = tuple(f for f in ids if f in OPTIONAL_ADDITIONS)
    columns = ids + tuple("missing_" + f for f in optional)
    result = []
    for row in rows:
        values = []
        for feature in ids:
            value = row["rank_" + feature]
            missing = row["missing_" + feature]
            if missing != (row["raw_" + feature] is None):
                raise ValueError("Raw missingness flag contradiction")
            if value is None:
                if feature not in optional or not missing:
                    raise ValueError("Required feature unavailable; never silently encode")
                value = 0.0
            if not math.isfinite(value) or not -0.5 <= value <= 0.5:
                raise ValueError("Invalid centered observed cross-sectional rank")
            values.append(value)
        values.extend(float(row["missing_" + f]) for f in optional)
        result.append(tuple(values))
    return columns, tuple(result)


def _runtime():
    result = {
        n: version(n)
        for n in ("numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow", "threadpoolctl")
    }
    result.update(
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    return result


def fit_predict_visible(parameters, matrix, targets, weights, prediction_matrix):
    """Actual fixed estimator construction/fit/predict, shown in the main notebook."""
    from pathlib import Path

    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Native wider estimator fitting is restricted to Kaggle")
    import numpy as np
    from lightgbm import LGBMRegressor
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        model = LGBMRegressor(**dict(parameters))
        model.fit(
            np.asarray(matrix, dtype=float),
            np.asarray(targets, dtype=float),
            sample_weight=np.asarray(weights, dtype=float),
        )
        predicted = model.predict(np.asarray(prediction_matrix, dtype=float))
    return model, predicted


def validate_visible_fit(callback):
    """Notebook may expose the exact approved function, never a changed learner."""
    original = ast.dump(ast.parse(inspect.getsource(fit_predict_visible)), include_attributes=False)
    supplied = ast.dump(ast.parse(inspect.getsource(callback)), include_attributes=False)
    if supplied != original:
        raise ValueError("Notebook fit function differs from fixed reviewed estimator source")
    return d._digest({"exact_visible_fit_ast": original})


def _closure():
    from quantlab.stocks import ablation, prepare_wider, wider_features

    return {
        **d.development_source_closure(),
        **code_pins(),
        "wider_study": file_sha256(Path(__file__)),
        "reference_authentication": file_sha256(Path(ablation.__file__)),
        "projection_registry": file_sha256(Path(prepare_wider.__file__)),
        "wider_feature_formulas": file_sha256(Path(wider_features.__file__)),
    }


def _load_rows(root, manifest, data):
    expected = {(r.signal_session, r.asset_id): r for r in data.features}
    actual = {}
    for item in manifest["outputs"]:
        for row in d._read_columns(Path(root) / item["path"], None):
            key = row["signal_session"], row["asset_id"]
            original = expected.get(key)
            if original is None or key in actual:
                raise ValueError("Wider panel changed original historical cohort")
            if (
                row["input_rows_sha256"] != original.input_rows_sha256
                or row["signal_at"] != original.signal_at
                or row["intended_execution_at"] != original.intended_execution_at
                or any(
                    struct.pack("<d", row["rank_" + f]) != struct.pack("<d", value)
                    for f, value in zip(d.FEATURE_IDS, original.values, strict=True)
                )
                or any(
                    struct.pack("<d", row["raw_" + f]) != struct.pack("<d", value)
                    for f, value in zip(d.FEATURE_IDS, original.raw_values, strict=True)
                )
            ):
                raise ValueError("Wider original raw/rank/clocks/source identity differs")
            actual[key] = row
    if actual.keys() != expected.keys():
        raise ValueError("Every original prior training cohort row required")
    return actual


def run_wider_study(
    prepared_root,
    wider_root,
    expected_wider_sha256,
    base_spec_path,
    wider_spec_path,
    schedule,
    reference_root,
    output,
    *,
    service_deadline_monotonic,
    visible_fit_callback=None,
):
    """Run four96-date fits; no native host run, no checkpoint importer yet.

    Complete cells carry model/prediction/fit receipts plus chained pins for a
    future separately reviewed external resume contract. This initial harness
    retains partial evidence but never imports it or declares partial completion.
    """
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Wider native training is restricted to Kaggle")

    def budget():
        if (
            not math.isfinite(service_deadline_monotonic)
            or time.monotonic() >= service_deadline_monotonic
        ):
            raise TimeoutError("Startup-inclusive wider study deadline exhausted")

    budget()
    spec, base = load_spec(wider_spec_path), d.load_spec(base_spec_path)
    if base["target_transform"] != RAW:
        raise ValueError("Original fixed RAW base contract required")
    reference, reference_metrics, reference_fits = authenticate_reference(reference_root, base)
    data = d.read_development_inputs(prepared_root, PARENT_SHA, base, schedule)
    manifest = verify_wider_panel(wider_root, expected_wider_sha256)
    if not data.full_research_scope or manifest["config"]["calendar"] != schedule.provenance:
        raise ValueError("Full original cohort and exact wider calendar required")
    if (
        manifest["config"]["feature_reproduction_policy_sha256"]
        != spec["feature_reproduction"]["policy_sha256"]
        or manifest["config"]["feature_reproduction_parent_platform"] != "Windows"
    ):
        raise ValueError("Wider panel must use the explicitly frozen representation policy")
    wider = _load_rows(wider_root, manifest, data)
    dates = tuple((day, cutoff) for day, cutoff in data.decision_times if day in reference)
    if len(dates) != 96 or {day for day, _ in dates} != set(reference):
        raise ValueError("All96 original monthly decision dates required")
    candidate = next(c for c in d.registered_candidates(base) if c.candidate_id == CANDIDATE_ID)
    import pyarrow as pa
    import pyarrow.parquet as pq

    callback = fit_predict_visible if visible_fit_callback is None else visible_fit_callback
    visible_fit_sha = validate_visible_fit(callback)

    root = Path(output).resolve()
    if any(
        root.is_relative_to(Path(p).resolve()) for p in (prepared_root, wider_root, reference_root)
    ):
        raise ValueError("Fresh study output outside immutable inputs required")
    runtime, closure = _runtime(), _closure()
    if runtime["python"] != "3.12.15":
        raise ValueError("Exact locked Python3.12.15 required")
    hashes = {
        "base": file_sha256(Path(base_spec_path)),
        "wider": file_sha256(Path(wider_spec_path)),
    }
    pins = {
        "prepared_manifest_sha256": PARENT_SHA,
        "wider_manifest_sha256": expected_wider_sha256,
        "reference_manifest_sha256": REFERENCE_SHA,
        "runtime": runtime,
        "source": closure,
        "config_hashes": hashes,
        "candidate": asdict(candidate),
        "visible_estimator_ast_sha256": visible_fit_sha,
    }
    root.mkdir(parents=True, exist_ok=False)
    cells, metrics, controls = [], [], []
    current = None
    try:
        d._write(
            root / "config.json",
            {
                "pins": pins,
                "spec": spec,
                "base_spec": base,
                "registered_at_actual_utc": datetime.now(UTC),
                "resume_supported": False,
                "holdout_evaluated": False,
                "outcome_value_partitions_refused": data.outcome_value_partitions_refused,
                "outcome_value_partitions_read": data.outcome_value_partitions_read,
            },
        )
        by_date, outcomes = defaultdict(list), defaultdict(list)
        for row in data.features:
            by_date[row.signal_session].append(row)
        for row in data.outcomes:
            outcomes[row.signal_session].append(row)
        for day, cutoff in dates:
            budget()
            batch = d.training_batch(data, day, cutoff, base)
            rows = tuple(by_date[day])
            oldfit = reference_fits[day]
            if not batch.ready or (
                oldfit["training_input_sha256"] != batch.input_sha256
                or oldfit["training_rows"] != len(batch.feature_rows)
                or oldfit["training_dates"] != len(batch.date_counts)
                or oldfit["information_cutoff"] != cutoff.isoformat()
                or dict(oldfit["parameters"]) != dict(candidate.parameters)
            ):
                raise ValueError("Original complete RAW cohort/weights/parameters changed")
            current_wide = tuple(wider[day, r.asset_id] for r in rows)
            train_wide = tuple(wider[r.signal_session, r.asset_id] for r in batch.feature_rows)
            if {r.asset_id for r in rows} != {r["asset_id"] for r in reference[day]}:
                raise ValueError("Original reference asset cohort changed")
            for arm, _ in ARMS:
                budget()
                current = {"signal_session": day, "arm_id": arm}
                columns, matrix = project_rows(train_wide, arm)
                prediction_columns, prediction_matrix = project_rows(current_wide, arm)
                if columns != prediction_columns:
                    raise ValueError("Training/prediction feature order differs")
                projected_sha = d._digest(
                    {
                        "raw_training_input_sha256": batch.input_sha256,
                        "column_ids": columns,
                        "matrix": matrix,
                        "weights": batch.weights,
                    }
                )
                fit_id = d._digest(
                    {"pins": pins, "day": day, "arm": arm, "projected_input_sha256": projected_sha}
                )
                receipt = {
                    **current,
                    "fit_id": fit_id,
                    "column_ids": columns,
                    "information_cutoff": cutoff,
                    "max_label_available_at": batch.max_label_available_at,
                    "training_rows": len(batch.feature_rows),
                    "training_dates": len(batch.date_counts),
                    "date_counts": batch.date_counts,
                    "target_transform": RAW,
                    "original_raw_training_input_sha256": batch.input_sha256,
                    "projected_input_sha256": projected_sha,
                    "parameters": candidate.parameters,
                    "actual_fit_started_at": datetime.now(UTC),
                }
                model, prediction = callback(
                    candidate.parameters,
                    matrix,
                    training_targets(batch, RAW),
                    batch.weights,
                    prediction_matrix,
                )
                scores = tuple(
                    (r.asset_id, float(v)) for r, v in zip(rows, prediction, strict=True)
                )
                score_bytes(scores)
                receipt["actual_fit_finished_at"] = datetime.now(UTC)
                path = root / "cells" / str(day) / arm
                path.parent.mkdir(parents=True, exist_ok=True)
                saved = [
                    {
                        "asset_id": r.asset_id,
                        "signal_session": day,
                        "signal_at": cutoff,
                        "arm_id": arm,
                        "score": score,
                        "score_kind": "decimal_return_forecast",
                        "fit_id": fit_id,
                        "feature_input_sha256": r.input_rows_sha256,
                    }
                    for r, (_, score) in zip(rows, scores, strict=True)
                ]
                parquet = path.with_suffix(".parquet")
                pending = path.with_suffix(".pending")
                pq.write_table(pa.Table.from_pylist(saved), pending, compression="zstd")
                pending.rename(parquet)
                model_path = path.with_suffix(".txt")
                model.booster_.save_model(str(model_path))
                d._write(path.with_suffix(".json"), receipt)
                # Immutable prediction is sealed before the current outcome join.
                sealed = d._read_columns(parquet, None)
                sealed_scores = tuple((r["asset_id"], r["score"]) for r in sealed)
                measured = d.date_ic(sealed_scores, outcomes[day], minimum_support=30)
                if arm == ARMS[0][0]:
                    if score_bytes(sealed_scores) != score_bytes(
                        tuple((r["asset_id"], r["score"]) for r in reference[day])
                    ):
                        raise ValueError("Fresh exact-six float64 predictions do not reproduce b46")
                    if any(
                        measured[k] != reference_metrics[day][k]
                        for k in ("status", "ic", "n_predictions", "n_resolved")
                    ):
                        raise ValueError("Fresh six IC/support differs from b46")
                metrics.append({**current, **measured})
                artifacts = [
                    {
                        "path": p.relative_to(root).as_posix(),
                        "sha256": file_sha256(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in (parquet, model_path, path.with_suffix(".json"))
                ]
                cell = {**current, "fit_id": fit_id, "artifacts": artifacts}
                previous = root / "checkpoints" / f"{len(cells):06d}.json"
                cells.append(cell)
                d._write(
                    root / "checkpoints" / f"{len(cells):06d}.json",
                    {
                        "complete": False,
                        "pins": pins,
                        "cell": cell,
                        "parent_checkpoint_sha256": file_sha256(previous)
                        if previous.exists()
                        else None,
                    },
                )
                print(f"Completed actual wider cell {len(cells)}/384 {day} {arm}", flush=True)
            # Separate fixed controls, without fitting or choosing a new recipe.
            original_controls = tuple(
                c for c in d.registered_candidates(base) if c.kind == "control"
            )
            control_values = {r.asset_id: {} for r in rows}
            for control in original_controls:
                for asset, value in d.control_scores(control, rows):
                    control_values[asset][control.candidate_id] = value
            for row in rows:
                x = dict(zip(d.FEATURE_IDS, row.values, strict=True))
                control_values[row.asset_id]["separate_blueprint_three_factor"] = math.fsum(
                    (x["F01"] / 3, x["F07"] / 3, -x["F09"] / 3)
                )
            d._write(
                root / "controls" / (str(day) + ".json"),
                {"signal_session": day, "signal_at": cutoff, "scores": control_values},
            )
            for name in next(iter(control_values.values())):
                scores = tuple((a, v[name]) for a, v in control_values.items())
                controls.append(
                    {
                        "signal_session": day,
                        "control_id": name,
                        **d.date_ic(scores, outcomes[day], minimum_support=30),
                    }
                )
        paired = defaultdict(dict)
        for metric in metrics:
            paired[metric["signal_session"]][metric["arm_id"]] = metric
        common = sorted(
            day
            for day, arms in paired.items()
            if len(arms) == 4 and all(row["status"] == "valid" for row in arms.values())
        )
        if len(cells) != 384 or len(common) != 94:
            raise ValueError("All384 cells and94 common mature months required")
        budget()
        verify_wider_panel(wider_root, expected_wider_sha256)
        d.verify_prepared_panel(prepared_root, PARENT_SHA)
        authenticate_reference(reference_root, base)
        if (
            closure != _closure()
            or visible_fit_sha != validate_visible_fit(callback)
            or runtime != _runtime()
            or hashes
            != {
                "base": file_sha256(Path(base_spec_path)),
                "wider": file_sha256(Path(wider_spec_path)),
            }
        ):
            raise ValueError("Final source/config/runtime changed")
        d._write(root / "metrics.json", metrics)
        d._write(root / "control-metrics.json", controls)
        d._write(
            root / "paired-comparison.json",
            {
                "common_mature_dates": common,
                "selection": "none",
                "post_observation": True,
                "uncertainty": "not_yet_evaluated",
                "fresh_six_reference_exact": True,
                "mean_ic": {
                    arm: math.fsum(paired[day][arm]["ic"] for day in common) / 94 for arm, _ in ARMS
                },
            },
        )
        artifacts = [
            {
                "path": p.relative_to(root).as_posix(),
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ]
        budget()
        d._write(
            root / "manifest.json",
            {
                "complete": True,
                "pins": pins,
                "cells": 384,
                "common_mature_months": 94,
                "reference_reproduced_exactly": True,
                "holdout_evaluated": False,
                "artifacts": artifacts,
            },
        )
        budget()
        (root / "COMPLETE").write_text(file_sha256(root / "manifest.json") + "\n", encoding="ascii")
        return {
            "study": str(root),
            "cells": 384,
            "common_mature_months": 94,
            "manifest_sha256": file_sha256(root / "manifest.json"),
        }
    except BaseException as error:
        d._write(
            root / "FAILURE.json",
            {
                "complete": False,
                "current": current,
                "completed_cells": len(cells),
                "error_type": type(error).__name__,
                "message": str(error),
                "comparative_metrics_authorized": False,
            },
        )
        raise
