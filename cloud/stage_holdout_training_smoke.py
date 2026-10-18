"""Local source-only stage for the explicitly bounded real 2010 adapter check."""

from __future__ import annotations

import ast
import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

from cloud.holdout_training_smoke_runner import canonical_runner, require
from cloud.stock_development_runner import validate_data_freeze
from quantlab.evaluation.holdout import source_closure


def raw_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(
    repo,
    output,
    *,
    readiness_path,
    expected_readiness_sha256,
    data_freeze_path,
    expected_data_freeze_sha256,
):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    require(not output.exists(), "NEW_IMMUTABLE_LOCAL_STAGE_REQUIRED")
    readiness_raw, data_raw = Path(readiness_path).read_bytes(), Path(data_freeze_path).read_bytes()
    require(sha(readiness_raw) == expected_readiness_sha256, "ACTUAL_SOURCE_CHECKPOINT_DIFFERS")
    require(sha(data_raw) == expected_data_freeze_sha256, "ACTUAL_PREPARED_FREEZE_DIFFERS")
    readiness, data = json.loads(readiness_raw), json.loads(data_raw)
    validate_data_freeze(data)
    closure = source_closure(repo, ["src/quantlab/evaluation/holdout_training_smoke.py"])
    require(
        closure == readiness["full_production_closure_sha256"],
        "ACTUAL_REVIEWED_47_SOURCE_CLOSURE_REQUIRED",
    )
    lock = tomllib.loads((repo / "uv.lock").read_text(encoding="utf-8"))
    versions = {p["name"]: p["version"] for p in lock["package"] if "version" in p}
    runtime_names = (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "exchange-calendars",
        "threadpoolctl",
    )
    runtime = {
        "python": "3.12.15",
        "machine": "x86_64",
        "packages": {p: versions[p] for p in runtime_names},
    }
    config = "configs/stock-development.toml"
    engineering = {
        "schema": "pre2015_stock_adapter_training_engineering_v1",
        "prediction_sessions": ["2010-06-30"],
        "candidate_ids": ["ridge_alpha_1", "lgbm_regression_l2_leaves7_iterations100"],
        "contains_final_evaluation": False,
        "scores_prediction_outcomes": False,
        "source_sha256": closure,
        "runtime": runtime,
        "config_path": config,
        "config_sha256": sha((repo / config).read_bytes()),
        "panel_manifest_path": "manifest.json",
        "panel_manifest_sha256": data["prepared_manifest_sha256"],
    }
    additional = (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        config,
        "cloud/stock_development_runner.py",
        "cloud/holdout_training_smoke_runner.py",
        "cloud/stage_holdout_training_smoke.py",
    )
    files = {name: (repo / name).read_bytes() for name in (*closure, *additional)}
    files.update(
        {
            "engineering-plan.json": raw_json(engineering),
            "prepared-data-freeze.json": data_raw,
            "source-readiness.json": readiness_raw,
        }
    )
    runner_raw = files["cloud/holdout_training_smoke_runner.py"]
    canonical = canonical_runner(runner_raw)
    require(
        canonical == runner_raw.replace(b"\r\n", b"\n"), "UNRENDERED_CANONICAL_TEMPLATE_REQUIRED"
    )
    for name, raw in files.items():
        if name.endswith(".py"):
            ast.parse(raw)
    mapping = {
        "source__" + name.replace("/", "__"): {
            "path": name,
            "bytes": len(raw),
            "sha256": sha(raw),
        }
        for name, raw in files.items()
    }
    require(len(mapping) == len(files), "FLAT_SOURCE_COLLISION")
    source = "aikyatansinha/portfolio-quant-holdout-smoke-source-v1"
    kernel = "aikyatansinha/portfolio-quant-holdout-smoke-v1"
    plan = {
        "schema": "pre2015_stock_adapter_cloud_transport_v1",
        "package_files": mapping,
        "prediction_sessions": engineering["prediction_sessions"],
        "candidate_ids": engineering["candidate_ids"],
        "holdout_outcomes": False,
        "service_budget_seconds": 3600,
        "engineering_plan_sha256": sha(files["engineering-plan.json"]),
        "runner_raw_sha256": sha(runner_raw),
        "runner_canonical_sha256": sha(canonical),
        "source_readiness_sha256": sha(readiness_raw),
        "prepared_data_freeze_sha256": sha(data_raw),
        "datasets": [source, data["dataset"]],
        "prepared_dataset_id": data["dataset_id"],
        "prepared_dataset_version": data["version"],
    }
    plan_raw = raw_json(plan)
    rendered = canonical.replace(
        b'ROOT_PLAN_SHA256 = "UNBOUND"', ('ROOT_PLAN_SHA256 = "' + sha(plan_raw) + '"').encode(), 1
    )
    rendered = rendered.replace(
        b'RUNNER_CANONICAL_SHA256 = "UNBOUND"',
        ('RUNNER_CANONICAL_SHA256 = "' + sha(canonical) + '"').encode(),
        1,
    )
    require(
        canonical_runner(rendered) == canonical and b"\r\n" not in rendered,
        "ACTUAL_LF_RENDER_ROUNDTRIP_REQUIRED",
    )
    ast.parse(rendered)
    payload, upload, job = output / "payload", output / "upload", output / "kernel"
    for directory in (payload, upload, job):
        directory.mkdir(parents=True, exist_ok=False)
    payload_files = {flat: files[row["path"]] for flat, row in mapping.items()}
    payload_files["holdout-smoke-plan.json"] = plan_raw
    for name, raw in payload_files.items():
        (payload / name).write_bytes(raw)
    with zipfile.ZipFile(upload / "payload.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payload_files.items()):
            archive.writestr(name, raw)
    with zipfile.ZipFile(upload / "payload.zip") as archive:
        require(set(archive.namelist()) == set(payload_files), "ZIP_INVENTORY_DIFFERS")
        require(
            all(archive.read(name) == raw for name, raw in payload_files.items()),
            "ZIP_RAW_BYTES_DIFFER",
        )
    (upload / "dataset-metadata.json").write_bytes(
        raw_json(
            {
                "id": source,
                "title": "Quant Holdout Adapter Smoke Source V1",
                "isPrivate": True,
                "licenses": [{"name": "other"}],
            }
        )
    )
    (job / "run.py").write_bytes(rendered)
    (job / "kernel-metadata.json").write_bytes(
        raw_json(
            {
                "id": kernel,
                "title": "Quant Holdout Adapter Smoke V1",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": True,
                "dataset_sources": plan["datasets"],
                "kernel_sources": [],
                "competition_sources": [],
            }
        )
    )
    proof = {
        "source_only": True,
        "provider_mutation_performed": False,
        "root_review_required_before_publication": True,
        "production_closure_count": len(closure),
        "payload_files": {
            n: {"bytes": len(raw), "sha256": sha(raw)} for n, raw in payload_files.items()
        },
        "plan_sha256": sha(plan_raw),
        "rendered_runner_sha256": sha(rendered),
        "archive_sha256": sha((upload / "payload.zip").read_bytes()),
        "runtime_expected_from_actual_lock_not_observed_yet": runtime,
    }
    (output / "local-stage-proof.json").write_bytes(raw_json(proof))
    return proof
