"""Stage ETF sensitivities only after actual whole Yahoo data and source approval freezes."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path

from cloud import etf_development_runner as data_helper
from cloud.etf_sensitivity_runner import (
    PLAN_NAME,
    canonical_runner,
    encoded,
    read_bound,
    require,
    sha,
    validate_plan,
)

CONFIG = "configs/etf-sensitivity-development.toml"
TRANSPORT = (
    "cloud/risk_forecast_runner.py",
    "cloud/etf_development_runner.py",
    "cloud/etf_sensitivity_runner.py",
    "cloud/stage_etf_sensitivity.py",
)
PROJECT = ("pyproject.toml", "uv.lock", ".python-version", "README.md")


def import_closure(repo):
    src = Path(repo) / "src"
    pending, found = [src / "quantlab/allocation/sensitivity_study.py"], set()
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
                    base = ".".join(
                        (*parts[: len(parts) - node.level + 1], base)
                    ).strip(".")
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
    names = import_closure(repo) + (CONFIG,) + TRANSPORT + PROJECT
    require(len(names) == len(set(names)), "UNIQUE_TRANSITIVE_SOURCE_REQUIRED")
    result = {
        name: {
            "bytes": (Path(repo) / name).stat().st_size,
            "sha256": sha((Path(repo) / name).read_bytes()),
        }
        for name in names
    }
    return result


def runtime_from_lock(raw):
    lock = tomllib.loads(raw.decode())
    result = {}
    for name in (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "exchange-calendars",
    ):
        matches = [p["version"] for p in lock["package"] if p["name"] == name]
        require(len(matches) == 1, "EXACT_ONE_LOCKED_RUNTIME_VERSION_REQUIRED")
        result[name] = matches[0]
    return result


def build_stage(
    repo,
    output,
    *,
    data_freeze_path,
    expected_data_freeze_sha256,
    source_freeze_path,
    expected_source_freeze_sha256,
    source_dataset_id,
    kernel_id,
    service_budget_seconds=36000,
):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    frozen = {}
    for name, path, digest in (
        ("data-freeze.json", data_freeze_path, expected_data_freeze_sha256),
        ("source-freeze.json", source_freeze_path, expected_source_freeze_sha256),
    ):
        frozen[name] = read_bound(path, digest)
    data, approval = [json.loads(raw) for raw in frozen.values()]
    data_helper.validate_data_freeze(data)
    require(
        approval.get("schema") == "quant_etf_sensitivity_source_freeze_v1"
        and approval.get("approved_for_cloud") is True
        and approval.get("holdout_outcomes") is False,
        "ACTUAL_ROOT_SOURCE_APPROVAL_REQUIRED",
    )
    timestamp = datetime.fromisoformat(approval["approved_at"])
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "ACTUAL_AWARE_APPROVAL_REQUIRED",
    )
    inventory = production_inventory(repo)
    require(
        approval.get("files") == inventory, "EXACT_ROOT_APPROVED_SOURCE_BYTES_REQUIRED"
    )
    for name in (source_dataset_id, kernel_id):
        require(
            re.fullmatch(r"aikyatansinha/[a-z0-9][a-z0-9-]+", name),
            "EXPLICIT_OWNED_ID_REQUIRED",
        )
    package = {
        "__".join(Path(name).parts): {"path": name, **pin}
        for name, pin in inventory.items()
    }
    require(len(package) == len(inventory), "FLAT_NAME_COLLISION")
    for name, raw in frozen.items():
        package[name] = {
            "path": "provenance/" + name,
            "bytes": len(raw),
            "sha256": sha(raw),
        }
    template = (repo / "cloud/etf_sensitivity_runner.py").read_bytes()
    plan = {
        "schema": "quant_etf_sensitivity_plan_v1",
        "scenarios": [
            "reference_sample_target10",
            "target05_only",
            "target15_only",
            "covariance_ledoit_wolf_only",
            "covariance_ewma63_only",
            "weight_inverse_vol_only",
            "weight_risk_budget_only",
        ],
        "fee_bps": 10,
        "initial_signal": "2007-02-28",
        "end_close": "2014-12-31",
        "measured_sessions": 1975,
        "holdout_outcomes": False,
        "python": "3.12.15",
        "threads": 4,
        "service_budget_seconds": service_budget_seconds,
        "runtime": runtime_from_lock((repo / "uv.lock").read_bytes()),
        "package_files": package,
        "root_freeze_sha256": {name: sha(raw) for name, raw in frozen.items()},
        "runner_canonical_sha256": sha(canonical_runner(template)),
        "datasets": [data["dataset"], source_dataset_id],
        "kernel_id": kernel_id,
        "resume": "none_completed_paths_and_partial_prefixes_preserved",
    }
    validate_plan(plan)
    plan_raw = encoded(plan)
    rendered = canonical_runner(template)
    for name, digest in (
        ("ROOT_PLAN_SHA256", sha(plan_raw)),
        ("RUNNER_CANONICAL_SHA256", plan["runner_canonical_sha256"]),
    ):
        rendered = rendered.replace(
            name.encode() + b' = "UNBOUND"',
            name.encode() + b' = "' + digest.encode() + b'"',
        )
    payload = {
        flat: (repo / item["path"]).read_bytes()
        for flat, item in package.items()
        if not item["path"].startswith("provenance/")
    }
    payload.update(frozen)
    require(production_inventory(repo) == inventory, "SOURCE_CHANGED_DURING_STAGING")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("payload", "kernel", "upload-metadata"):
        (output / name).mkdir()
    for name, raw in payload.items():
        require(
            sha(raw) == package[name]["sha256"] and len(raw) == package[name]["bytes"],
            "COLLECTED_BYTE_MISMATCH",
        )
        (output / "payload" / name).write_bytes(raw)
    (output / "payload" / PLAN_NAME).write_bytes(plan_raw)
    (output / "kernel/etf_sensitivity_runner.py").write_bytes(rendered)
    (output / "kernel/kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[-1].replace("-", " "),
                "code_file": "etf_sensitivity_runner.py",
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
                "title": "Quant ETF one change sensitivity frozen source",
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
        "no_native_models_or_holdout_inputs": True,
    }
    (output / "stage-receipt.json").write_bytes(encoded(receipt))
    return receipt
