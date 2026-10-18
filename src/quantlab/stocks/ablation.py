"""Prospective column-removal study; native training runs only in Kaggle.

All cohorts, raw targets and monthly training weights come from the unchanged
development reader. The completed original predictions are an externally bound
reproducibility oracle, never a training input. This is post-observation research.
"""

import math
import platform
import struct
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from quantlab.data.archive import file_sha256
from quantlab.stocks import development as d
from quantlab.stocks.features import FEATURE_IDS
from quantlab.stocks.prepare import verify_prepared_panel
from quantlab.stocks.training_targets import RAW, training_targets

REFERENCE_SHA = "b46c22e5a6f46330668f6f3870c0c8cc2ddde15c283d3f9af8637f93557492d6"
CANDIDATE_ID = "lgbm_regression_l1_leaves15_iterations100"
FAMILIES = {
    "momentum_trend": ("F01", "F13", "F15"),
    "reversal": ("F07",),
    "risk": ("F09",),
    "activity": ("F19",),
}


@dataclass(frozen=True)
class Projection:
    arm_id: str
    feature_ids: tuple[str, ...]
    indices: tuple[int, ...]


def load_spec(path):
    spec = tomllib.loads(Path(path).read_text())
    if (
        spec["status"] != "post_observation_development_ablation_not_final_holdout"
        or spec["historical_preregistration"] is not False
        or datetime.fromisoformat(spec["registered_at"]).utcoffset() is None
        or spec["reference_manifest_sha256"] != REFERENCE_SHA
        or spec["candidate_id"] != CANDIDATE_ID
        or spec["target_transform"] != RAW
        or tuple(spec["feature_ids"]) != FEATURE_IDS
        or {k: tuple(v) for k, v in spec["removed_families"].items()} != FAMILIES
        or spec["projection"] != "remove_actual_columns_without_zero_replacement"
        or spec["reference_gate"]
        != "exact_keys_float64_score_bytes_target_units_and_IC_before_comparisons"
        or spec["minimum_common_mature_dates"] != 94
        or spec["scheduled_prediction_dates"] != 96
        or spec["threads"] != 4
        or spec["random_seed"] != 20261017
        or spec["selection"] != "none_descriptive_registered_family_contrasts"
    ):
        raise ValueError("Exact registered raw-target feature ablation required")
    return spec


def projections():
    return tuple(
        Projection(
            name,
            tuple(f for f in FEATURE_IDS if f not in removed),
            tuple(i for i, f in enumerate(FEATURE_IDS) if f not in removed),
        )
        for name, removed in [("all_features", ()), *FAMILIES.items()]
    )


def project_values(rows, projection):
    """Copy real feature columns only; preserve the original immutable rows."""
    if projection not in projections():
        raise ValueError("Unregistered feature projection")
    values = []
    for row in rows:
        if len(row.values) != len(FEATURE_IDS) or any(not math.isfinite(v) for v in row.values):
            raise ValueError("Six finite observed feature columns required")
        values.append(tuple(row.values[i] for i in projection.indices))
    return tuple(values)


def projected_input_sha(batch, projection):
    return d._digest(
        {
            "original_raw_training_input_sha256": batch.input_sha256,
            "feature_ids": projection.feature_ids,
            "rows": tuple(
                (r.asset_id, r.signal_session, r.input_rows_sha256, v)
                for r, v in zip(
                    batch.feature_rows, project_values(batch.feature_rows, projection), strict=True
                )
            ),
        }
    )


def score_bytes(scores):
    ordered = sorted(scores)
    if len({asset for asset, _ in ordered}) != len(ordered):
        raise ValueError("Unique original source identities required")
    if any(not math.isfinite(value) for _, value in ordered):
        raise ValueError("Finite observed prediction scores required")
    return tuple(asset for asset, _ in ordered), b"".join(
        struct.pack("<d", value) for _, value in ordered
    )


def _bound(root, relative, digest):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe reference artifact path")
    path = root.joinpath(*name.parts)
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or file_sha256(path) != digest
    ):
        raise ValueError("Reference artifact differs")
    return path


def authenticate_reference(root, base_spec=None):
    """Authenticate only saved scalar predictions/metrics, never native models."""
    import json

    root = Path(root)
    if (
        file_sha256(root / "manifest.json") != REFERENCE_SHA
        or (root / "COMPLETE").read_text().strip() != REFERENCE_SHA
    ):
        raise ValueError("Original complete b46 reference required")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        not manifest["complete"]
        or manifest["spec"]["target_transform"] != RAW
        or manifest["final_holdout_evaluation"]
    ):
        raise ValueError("Completed raw-target development reference required")
    if base_spec is not None and manifest["spec"] != base_spec:
        raise ValueError("Original raw-target base recipe changed")
    inventory = {a["path"]: a["sha256"] for a in manifest["artifacts"]}
    paths = sorted(p for p in inventory if p.startswith("predictions/") and p.endswith(".parquet"))
    if len(paths) != 96:
        raise ValueError("All 96 original prediction dates required")
    fit_index = json.loads(_bound(root, "fit-index.json", inventory["fit-index.json"]).read_bytes())
    fits = {
        date.fromisoformat(fit["signal_session"]): fit
        for fit in fit_index
        if fit["candidate_id"] == CANDIDATE_ID
    }
    if len(fits) != 96:
        raise ValueError("All original monthly fit receipts required")
    references = {}
    for relative in paths:
        rows = d._read_columns(_bound(root, relative, inventory[relative]), None)
        selected = tuple(r for r in rows if r["candidate_id"] == CANDIDATE_ID)
        day = date.fromisoformat(Path(relative).stem)
        if (
            day >= d.HOLDOUT_START
            or not selected
            or any(
                r["signal_session"] != day or r["score_kind"] != "decimal_return_forecast"
                for r in selected
            )
        ):
            raise ValueError("Original reference identity/units mismatch")
        score_bytes(tuple((r["asset_id"], r["score"]) for r in selected))
        references[day] = selected
        fit = fits[day]
        if (
            fit["candidate_id"] != CANDIDATE_ID
            or fit["signal_session"] != day.isoformat()
            or fit["status"] != "fitted"
            # Original b46 predates this per-fit field; its bound manifest/base
            # spec above explicitly supplies the raw-target contract.
            or ("training_target_transform" in fit and fit["training_target_transform"] != RAW)
            or {r["fit_id"] for r in selected} != {fit["fit_id"]}
        ):
            raise ValueError("Original reference fit semantics differ")
    metrics = json.loads(_bound(root, "metrics.json", inventory["metrics.json"]).read_bytes())
    metrics = {
        date.fromisoformat(r["signal_session"]): r
        for r in metrics
        if r["candidate_id"] == CANDIDATE_ID
    }
    if set(metrics) != set(references):
        raise ValueError("Reference metric date support differs")
    return references, metrics, fits


def _closure():
    return {**d.development_source_closure(), "ablation": file_sha256(Path(__file__))}


def run_feature_ablation(
    prepared_root,
    expected_manifest_sha256,
    base_spec_path,
    ablation_spec_path,
    schedule,
    reference_root,
    output,
    *,
    service_deadline_monotonic=None,
):
    """Fit 5x96 prescribed cells; no comparative result before reference replay.

    Checkpoints retain complete cells and models. This initial harness does not
    import checkpoints: interrupted jobs preserve evidence but require a fresh
    run, until an externally frozen resume contract is separately approved.
    """
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Native ablation fits are restricted to Kaggle")

    def budget():
        if service_deadline_monotonic is not None and (
            not math.isfinite(service_deadline_monotonic)
            or time.monotonic() >= service_deadline_monotonic
        ):
            raise TimeoutError("Startup-inclusive ablation service budget exhausted")

    budget()
    spec = load_spec(ablation_spec_path)
    base = d.load_spec(base_spec_path)
    if base["target_transform"] != RAW:
        raise ValueError("Unchanged raw-target base spec required")
    candidate = next(c for c in d.registered_candidates(base) if c.candidate_id == CANDIDATE_ID)
    reference, reference_metrics, reference_fits = authenticate_reference(reference_root, base)
    data = d.read_development_inputs(prepared_root, expected_manifest_sha256, base, schedule)
    if not data.full_research_scope:
        raise ValueError("Entire authenticated prepared population required")
    dates = tuple((day, cutoff) for day, cutoff in data.decision_times if day in reference)
    if len(dates) != 96 or {day for day, _ in dates} != set(reference):
        raise ValueError("Exact original 96 monthly decisions required")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lightgbm import LGBMRegressor
    from threadpoolctl import threadpool_limits

    root = Path(output).resolve()
    if (
        root == Path(prepared_root).resolve()
        or Path(prepared_root).resolve() in root.parents
        or root == Path(reference_root).resolve()
        or Path(reference_root).resolve() in root.parents
    ):
        raise ValueError("Fresh output outside immutable input trees required")
    root.mkdir(parents=True, exist_ok=False)
    hashes = {
        "base_spec": file_sha256(Path(base_spec_path)),
        "ablation_spec": file_sha256(Path(ablation_spec_path)),
    }
    runtime = {
        n: version(n)
        for n in ("numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow", "threadpoolctl")
    }
    runtime.update(
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    closure = _closure()
    pins = {
        "prepared_manifest_sha256": expected_manifest_sha256,
        "reference_manifest_sha256": REFERENCE_SHA,
        "spec_hashes": hashes,
        "source_closure": closure,
        "runtime": runtime,
        "candidate": asdict(candidate),
    }
    cells, metrics = [], []
    schema = pa.schema(
        [
            ("arm_id", pa.string()),
            ("asset_id", pa.string()),
            ("signal_session", pa.date32()),
            ("signal_at", pa.timestamp("us", tz="UTC")),
            ("score", pa.float64()),
            ("score_kind", pa.string()),
            ("fit_id", pa.string()),
            ("feature_input_sha256", pa.string()),
        ]
    )
    current = None
    try:
        d._write(
            root / "config.json",
            {
                "complete": False,
                "pins": pins,
                "spec": spec,
                "outcome_value_partitions_refused": data.outcome_value_partitions_refused,
                "outcome_value_partitions_read": data.outcome_value_partitions_read,
                "holdout_evaluated": False,
                "resume_supported": False,
            },
        )
        features = defaultdict(list)
        outcomes = defaultdict(list)
        for row in data.features:
            features[row.signal_session].append(row)
        for row in data.outcomes:
            outcomes[row.signal_session].append(row)
        for day, cutoff in dates:
            batch = d.training_batch(data, day, cutoff, base)
            rows = tuple(features[day])
            if not batch.ready or not rows:
                raise ValueError("Original full monthly training support required")
            if (
                reference_fits[day]["training_input_sha256"] != batch.input_sha256
                or reference_fits[day]["training_rows"] != len(batch.feature_rows)
                or reference_fits[day]["training_dates"] != len(batch.date_counts)
                or dict(reference_fits[day]["parameters"]) != dict(candidate.parameters)
                or reference_fits[day]["information_cutoff"] != cutoff.isoformat()
            ):
                raise ValueError("Original cohort/weights/raw targets/model parameters changed")
            old = {r["asset_id"]: r for r in reference[day]}
            if set(old) != {r.asset_id for r in rows} or any(
                old[r.asset_id]["feature_input_sha256"] != r.input_rows_sha256
                or old[r.asset_id]["signal_at"] != cutoff
                for r in rows
            ):
                raise ValueError("Reference and current real feature cohort differs")
            for projection in projections():
                budget()
                current = (day, projection.arm_id)
                input_sha = projected_input_sha(batch, projection)
                fit_id = d._digest(
                    {"pins": pins, "day": day, "arm": asdict(projection), "input_sha256": input_sha}
                )
                record = {
                    "fit_id": fit_id,
                    "arm": asdict(projection),
                    "signal_session": day,
                    "information_cutoff": cutoff,
                    "original_raw_training_input_sha256": batch.input_sha256,
                    "projected_input_sha256": input_sha,
                    "training_rows": len(batch.feature_rows),
                    "training_dates": len(batch.date_counts),
                    "date_counts": batch.date_counts,
                    "max_label_available_at": batch.max_label_available_at,
                    "parameters": candidate.parameters,
                    "target_transform": RAW,
                    "actual_fit_started_at": datetime.now(UTC),
                }
                with threadpool_limits(limits=4):
                    model = LGBMRegressor(**dict(candidate.parameters))
                    model.fit(
                        np.asarray(project_values(batch.feature_rows, projection), dtype=float),
                        np.asarray(training_targets(batch, RAW), dtype=float),
                        sample_weight=np.asarray(batch.weights, dtype=float),
                    )
                    predicted = model.predict(
                        np.asarray(project_values(rows, projection), dtype=float)
                    )
                scores = tuple((r.asset_id, float(v)) for r, v in zip(rows, predicted, strict=True))
                score_bytes(scores)
                record["actual_fit_finished_at"] = datetime.now(UTC)
                relative = f"cells/{day}/{projection.arm_id}"
                path = root / (relative + ".parquet")
                path.parent.mkdir(parents=True, exist_ok=True)
                saved = [
                    {
                        "arm_id": projection.arm_id,
                        "asset_id": r.asset_id,
                        "signal_session": day,
                        "signal_at": cutoff,
                        "score": score,
                        "score_kind": "decimal_return_forecast",
                        "fit_id": fit_id,
                        "feature_input_sha256": r.input_rows_sha256,
                    }
                    for r, (_, score) in zip(rows, scores, strict=True)
                ]
                pending = path.with_suffix(".pending")
                pq.write_table(
                    pa.Table.from_pylist(saved, schema=schema), pending, compression="zstd"
                )
                pending.rename(path)
                model_path = root / (relative + ".txt")
                model.booster_.save_model(str(model_path))
                record.update(
                    model_sha256=file_sha256(model_path), prediction_sha256=file_sha256(path)
                )
                d._write(root / (relative + ".json"), record)
                # Score artifact is immutable before label joining or comparison.
                sealed = d._read_columns(path, None)
                sealed_scores = tuple((r["asset_id"], r["score"]) for r in sealed)
                measured = d.date_ic(sealed_scores, outcomes[day], minimum_support=30)
                if projection.arm_id == "all_features":
                    if score_bytes(sealed_scores) != score_bytes(
                        tuple((r["asset_id"], r["score"]) for r in reference[day])
                    ):
                        raise ValueError(
                            "Fresh all-feature scores do not reproduce original float64 bytes"
                        )
                    previous = reference_metrics[day]
                    if any(
                        measured[key] != previous[key]
                        for key in ("status", "ic", "n_predictions", "n_resolved")
                    ):
                        raise ValueError("Fresh all-feature IC/support does not reproduce original")
                metrics.append({"signal_session": day, "arm_id": projection.arm_id, **measured})
                cell = {
                    "signal_session": day,
                    "arm_id": projection.arm_id,
                    "fit_id": fit_id,
                    "artifacts": [
                        {"path": p.relative_to(root).as_posix(), "sha256": file_sha256(p)}
                        for p in (path, model_path, root / (relative + ".json"))
                    ],
                }
                cells.append(cell)
                previous = root / "checkpoints" / f"{len(cells) - 1:06d}.json"
                d._write(
                    root / "checkpoints" / f"{len(cells):06d}.json",
                    {
                        "complete": False,
                        "pins": pins,
                        "cell": cell,
                        "parent_checkpoint_sha256": file_sha256(previous)
                        if len(cells) > 1
                        else None,
                    },
                )
        budget()
        if len(cells) != 480:
            raise ValueError("Full five-arm 96-date grid required")
        by_date = defaultdict(dict)
        for row in metrics:
            by_date[row["signal_session"]][row["arm_id"]] = row
        common = [
            day
            for day, armrows in by_date.items()
            if len(armrows) == 5 and all(r["status"] == "valid" for r in armrows.values())
        ]
        if len(common) != 94:
            raise ValueError("Full 94 mature paired dates required; no partial comparison")
        means = {
            p.arm_id: math.fsum(by_date[day][p.arm_id]["ic"] for day in common) / len(common)
            for p in projections()
        }
        summary = {
            "common_mature_dates": sorted(common),
            "mean_ic": means,
            "removed_family_minus_all_features": {
                k: v - means["all_features"] for k, v in means.items() if k != "all_features"
            },
            "reference_reproduced_exactly": True,
            "selection": "none",
            "post_observation": True,
            "uncertainty": "not_yet_evaluated",
            "economic_evaluation": "not_run",
        }
        if _closure() != closure or hashes != {
            "base_spec": file_sha256(Path(base_spec_path)),
            "ablation_spec": file_sha256(Path(ablation_spec_path)),
        }:
            raise ValueError("Ablation source/config changed")
        verify_prepared_panel(prepared_root, expected_manifest_sha256)
        authenticate_reference(reference_root)
        if _closure() != closure or hashes != {
            "base_spec": file_sha256(Path(base_spec_path)),
            "ablation_spec": file_sha256(Path(ablation_spec_path)),
        }:
            raise ValueError("Ablation source/config changed during input postflight")
        final_runtime = {
            name: version(name)
            for name in runtime
            if name not in ("python", "platform", "processor")
        }
        final_runtime.update(
            python=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor(),
        )
        if final_runtime != runtime:
            raise ValueError("Ablation runtime changed during input postflight")
        budget()
        d._write(root / "metrics.json", metrics)
        d._write(root / "paired-comparison.json", summary)
        manifest = {
            "complete": True,
            "pins": pins,
            "cells": len(cells),
            "reference_reproduced_exactly": True,
            "holdout_evaluated": False,
            "artifacts": [
                {
                    "path": p.relative_to(root).as_posix(),
                    "sha256": file_sha256(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ],
        }
        d._write(root / "manifest.json", manifest)
        budget()
        (root / "COMPLETE").write_text(file_sha256(root / "manifest.json") + "\n")
        return manifest
    except BaseException as error:
        d._write(
            root / "FAILURE.json",
            {
                "complete": False,
                "current_cell": current,
                "completed_cells": len(cells),
                "error_type": type(error).__name__,
                "reason": str(error)[:2000],
                "comparative_result_authorized": False,
            },
        )
        raise
