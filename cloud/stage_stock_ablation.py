"""Stage the registered stock ablation plus explicit saved scalar reference input."""

import ast
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path

from cloud import stock_development_runner as prepared_helper
from cloud.stock_ablation_runner import (
    PLAN_NAME,
    PREDICTION_MANIFEST,
    canonical_runner,
    encoded,
    read_bound,
    require,
    sha,
    validate_plan,
    validate_prediction_freeze,
)

CONFIGS = ("configs/stock-development.toml", "configs/stock-feature-ablation-development.toml")
TRANSPORT = (
    "cloud/stock_development_runner.py",
    "cloud/stock_ablation_runner.py",
    "cloud/stage_stock_ablation.py",
)
PROJECT = ("pyproject.toml", "uv.lock", ".python-version", "README.md")


def import_closure(repo):
    src = Path(repo) / "src"
    pending, found = [src / "quantlab/stocks/ablation.py"], set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for parent in path.parents:
            if parent == src:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer != path:
                pending.append(initializer)
        package = ".".join(path.relative_to(src).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join((*parts[: len(parts) - node.level + 1], base)).strip(".")
                names = [base, *(base + "." + a.name for a in node.names)]
            for name in names:
                if name == "quantlab" or name.startswith("quantlab."):
                    target = src.joinpath(*name.split("."))
                    for candidate in (
                        target.with_suffix(".py"),
                        target / "__init__.py",
                    ):
                        if candidate.is_file():
                            pending.append(candidate)
    return tuple(sorted(p.relative_to(Path(repo)).as_posix() for p in found))


def production_inventory(repo):
    names = import_closure(repo) + CONFIGS + TRANSPORT + PROJECT
    require(len(names) == len(set(names)), "UNIQUE_TRANSITIVE_SOURCE_REQUIRED")
    return {
        name: {
            "bytes": (Path(repo) / name).stat().st_size,
            "sha256": sha((Path(repo) / name).read_bytes()),
        }
        for name in names
    }


def runtime_from_lock(raw):
    lock = tomllib.loads(raw.decode())
    names = (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "threadpoolctl",
        "exchange-calendars",
    )
    result = {}
    for name in names:
        matches = [p["version"] for p in lock["package"] if p["name"] == name]
        require(len(matches) == 1, "EXACT_LOCKED_ML_RUNTIME_REQUIRED")
        result[name] = matches[0]
    return result


def reference_metrics(root):
    """Bind actual derived development metrics to the original manifest inventory."""
    root = Path(root)
    manifest = json.loads(read_bound(root / "manifest.json", PREDICTION_MANIFEST))
    require(
        manifest["complete"] is True
        and (root / "COMPLETE").read_text().strip() == PREDICTION_MANIFEST,
        "ACTUAL_COMPLETE_ORIGINAL_REFERENCE_REQUIRED",
    )
    matches = [a for a in manifest["artifacts"] if a["path"] == "metrics.json"]
    require(len(matches) == 1, "ONE_ORIGINAL_METRIC_ARTIFACT_REQUIRED")
    item = matches[0]
    raw = read_bound(root / "metrics.json", item["sha256"], item["bytes"])
    return raw, {
        "path": "reference/metrics.json",
        "sha256": sha(raw),
        "bytes": len(raw),
        "upstream_manifest_sha256": PREDICTION_MANIFEST,
        "role": "derived_development_scalar_metrics_no_outcome_prices_or_models",
    }


def build_stage(
    repo,
    output,
    *,
    prepared_data_freeze_path,
    expected_prepared_data_freeze_sha256,
    prediction_data_freeze_path,
    expected_prediction_data_freeze_sha256,
    source_freeze_path,
    expected_source_freeze_sha256,
    reference_root,
    source_dataset_id,
    kernel_id,
    service_budget_seconds=36000,
):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    frozen = {}
    for name, path, digest in (
        (
            "prepared-data-freeze.json",
            prepared_data_freeze_path,
            expected_prepared_data_freeze_sha256,
        ),
        (
            "prediction-data-freeze.json",
            prediction_data_freeze_path,
            expected_prediction_data_freeze_sha256,
        ),
        ("source-freeze.json", source_freeze_path, expected_source_freeze_sha256),
    ):
        frozen[name] = read_bound(path, digest)
    prepared, predictions, approval = [json.loads(raw) for raw in frozen.values()]
    prepared_helper.validate_data_freeze(prepared)
    validate_prediction_freeze(predictions)
    require(
        approval.get("schema") == "quant_stock_feature_ablation_source_freeze_v1"
        and approval.get("approved_for_cloud") is True
        and approval.get("holdout_outcomes") is False,
        "ACTUAL_ROOT_SOURCE_APPROVAL_REQUIRED",
    )
    timestamp = datetime.fromisoformat(approval["approved_at"])
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "AWARE_APPROVAL_REQUIRED",
    )
    inventory = production_inventory(repo)
    require(approval.get("files") == inventory, "EXACT_APPROVED_SOURCE_BYTES_REQUIRED")
    metrics_raw, metrics_pin = reference_metrics(reference_root)
    require(
        approval.get("reference_metrics") == metrics_pin,
        "EXPLICIT_REFERENCE_METRICS_APPROVAL_REQUIRED",
    )
    for name in (source_dataset_id, kernel_id):
        require(
            re.fullmatch(r"aikyatansinha/[a-z0-9][a-z0-9-]+", name), "EXPLICIT_OWNED_ID_REQUIRED"
        )
    package = {
        "__".join(Path(name).parts): {"path": name, **pin} for name, pin in inventory.items()
    }
    require(len(package) == len(inventory), "FLAT_NAME_COLLISION")
    for name, raw in frozen.items():
        package[name] = {"path": "provenance/" + name, "bytes": len(raw), "sha256": sha(raw)}
    package["reference-metrics.json"] = {k: metrics_pin[k] for k in ("path", "bytes", "sha256")}
    template = (repo / "cloud/stock_ablation_runner.py").read_bytes()
    require(b"\r" not in template, "LF_RUNNER_SOURCE_REQUIRED")
    plan = {
        "schema": "quant_stock_feature_ablation_plan_v1",
        "arms": ["all_features", "momentum_trend", "reversal", "risk", "activity"],
        "scheduled_dates": 96,
        "fit_cells": 480,
        "common_mature_dates": 94,
        "target_transform": "none_no_clipping_or_synthetic_replacement",
        "reference_manifest_sha256": PREDICTION_MANIFEST,
        "reference_metrics": metrics_pin,
        "holdout_outcomes": False,
        "python": "3.12.15",
        "uv": "0.13.1",
        "uv_group": "ml",
        "threads": 4,
        "service_budget_seconds": service_budget_seconds,
        "runtime": runtime_from_lock((repo / "uv.lock").read_bytes()),
        "package_files": package,
        "root_freeze_sha256": {name: sha(raw) for name, raw in frozen.items()},
        "runner_canonical_sha256": sha(canonical_runner(template)),
        "datasets": [prepared["dataset"], predictions["dataset"], source_dataset_id],
        "kernel_id": kernel_id,
        "resume": "not_supported_complete_cell_evidence_preserved",
        "upstream_archive_directly_reread": False,
        "source_package_contains_derived_reference_metrics": True,
    }
    validate_plan(plan)
    plan_raw = encoded(plan)
    rendered = canonical_runner(template)
    for name, digest in (
        ("ROOT_PLAN_SHA256", sha(plan_raw)),
        ("RUNNER_CANONICAL_SHA256", plan["runner_canonical_sha256"]),
    ):
        rendered = rendered.replace(
            name.encode() + b' = "UNBOUND"', name.encode() + b' = "' + digest.encode() + b'"'
        )
    payload = {
        flat: (repo / item["path"]).read_bytes()
        for flat, item in package.items()
        if not item["path"].startswith(("provenance/", "reference/"))
    }
    payload.update(frozen)
    payload["reference-metrics.json"] = metrics_raw
    require(production_inventory(repo) == inventory, "SOURCE_CHANGED_DURING_STAGING")
    require(reference_metrics(reference_root)[1] == metrics_pin, "REFERENCE_CHANGED_DURING_STAGING")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("payload", "kernel", "upload-metadata"):
        (output / name).mkdir()
    for name, raw in payload.items():
        require(
            sha(raw) == package[name]["sha256"] and len(raw) == package[name]["bytes"],
            "COLLECTED_BYTES_DIFFER",
        )
        (output / "payload" / name).write_bytes(raw)
    (output / "payload" / PLAN_NAME).write_bytes(plan_raw)
    (output / "kernel/stock_ablation_runner.py").write_bytes(rendered)
    (output / "kernel/kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[-1].replace("-", " "),
                "code_file": "stock_ablation_runner.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": plan["datasets"],
                "competition_sources": [],
                "kernel_sources": [],
            }
        )
    )
    (output / "upload-metadata/dataset-metadata.json").write_bytes(
        encoded(
            {
                "id": source_dataset_id,
                "title": "Quant stock feature ablation source and derived reference metrics",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            }
        )
    )
    receipt = {
        "status": "staged_not_uploaded",
        "plan_sha256": sha(plan_raw),
        "runner_raw_sha256": sha(rendered),
        "production_files": inventory,
        "datasets": plan["datasets"],
        "reference_metrics": metrics_pin,
        "payload_files": package,
        "source_package_contains_derived_reference_metrics": True,
    }
    (output / "stage-receipt.json").write_bytes(encoded(receipt))
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "output",
        "prepared-data-freeze-path",
        "expected-prepared-data-freeze-sha256",
        "prediction-data-freeze-path",
        "expected-prediction-data-freeze-sha256",
        "source-freeze-path",
        "expected-source-freeze-sha256",
        "reference-root",
        "source-dataset-id",
        "kernel-id",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--service-budget-seconds", type=int, default=36000)
    build_stage(**vars(parser.parse_args()))
