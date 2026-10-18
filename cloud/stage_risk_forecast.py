"""Stage plain risk forecast development source only after actual root freeze receipts.

This module performs no upload, kernel submission, model run or environment
installation. Reserved dataset upload metadata is emitted outside the payload.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from pathlib import Path

from cloud.risk_forecast_runner import (
    CONFIG_FILES,
    DATA_ID,
    ETF_DATA_ID,
    ETF_MANIFEST_SHA,
    ETF_PRICE_BASIS,
    ETF_RAW_BUNDLE_SHA,
    MANIFEST_SHA,
    PROJECT_FILES,
    SCIENTIFIC,
    SOURCE_FILES,
    TRANSPORT_FILES,
    canonical_json,
    canonical_runner,
    read_bound,
    require,
    safe_name,
    sha,
    valid_sha,
    validate_data_freeze,
    validate_etf_data_freeze,
    validate_plan,
)


def flat_name(relative):
    path = safe_name(relative)
    return "__".join(path.parts)


def production_inventory(repo):
    """Hash actual reviewed source bytes; this is not an approval receipt."""
    result = {}
    for name in SOURCE_FILES + CONFIG_FILES + PROJECT_FILES + TRANSPORT_FILES:
        raw = (Path(repo) / name).read_bytes()
        result[name] = {"sha256": sha(raw), "bytes": len(raw)}
    return result


def local_import_closure(repo):
    """Actual AST import audit catches an omitted transitive project module."""
    root = Path(repo) / "src"
    seen, pending = set(), ["quantlab.risk.study"]
    while pending:
        name = pending.pop()
        relative = Path(*name.split("."))
        candidates = (root / relative.with_suffix(".py"), root / relative / "__init__.py")
        path = next((p for p in candidates if p.is_file()), None)
        require(path is not None, "LOCAL_IMPORT_MODULE_MISSING")
        normalized = path.relative_to(root).as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for index in range(1, len(name.split("."))):
            pending.append(".".join(name.split(".")[:index]))
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                pending.extend(
                    alias.name for alias in node.names if alias.name.startswith("quantlab.")
                )
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    base = base[: len(base) - node.level + 1]
                    if node.module:
                        base += node.module.split(".")
                    target = ".".join(base)
                else:
                    target = node.module or ""
                if target == "quantlab" or target.startswith("quantlab."):
                    pending.append(target)
                    for alias in node.names:
                        possible = target + "." + alias.name
                        local = root.joinpath(*possible.split("."))
                        if local.with_suffix(".py").is_file() or (local / "__init__.py").is_file():
                            pending.append(possible)
    return tuple(sorted("src/" + name for name in seen))


def validate_source_freeze(repo, value):
    require(
        value.get("schema") == "quant_risk_source_freeze_v1" and value.get("status") == "READY",
        "ROOT_SOURCE_FREEZE_REQUIRED",
    )
    inventory = production_inventory(repo)
    require(value.get("files") == inventory, "CURRENT_RAW_SOURCE_DIFFERS_FROM_ROOT_FREEZE")
    require(
        set(local_import_closure(repo)) == set(SOURCE_FILES),
        "TRANSITIVE_PROJECT_SOURCE_CLOSURE_DIFFERS",
    )
    return inventory


def runtime_from_lock(raw):
    lock = tomllib.loads(raw.decode())
    names = (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "exchange-calendars",
        "threadpoolctl",
    )
    result = {}
    for name in names:
        versions = {p["version"] for p in lock["package"] if p["name"] == name}
        require(len(versions) == 1, "UNAMBIGUOUS_LOCKED_NUMERICAL_VERSION_REQUIRED")
        result[name] = versions.pop()
    return result


def render_runner(template, plan_sha):
    require(valid_sha(plan_sha), "EXTERNAL_PLAN_SHA_REQUIRED")
    canonical_sha = sha(canonical_runner(template))
    raw = template.replace(b"\r\n", b"\n")
    for name, digest in (
        ("ROOT_PLAN_SHA256", plan_sha),
        ("RUNNER_CANONICAL_SHA256", canonical_sha),
    ):
        raw, count = re.subn(
            rb"(?m)^" + name.encode() + rb' = "UNBOUND"$',
            name.encode() + b' = "' + digest.encode() + b'"',
            raw,
        )
        require(count == 1, "UNRENDERED_TEMPLATE_BINDING_REQUIRED")
    require(sha(canonical_runner(raw)) == canonical_sha, "RENDERED_RUNNER_IDENTITY_DIFFERS")
    compile(raw, "risk_forecast_runner.py", "exec")
    return raw


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
    service_budget_seconds,
    etf_data_freeze_path=None,
    expected_etf_data_freeze_sha256=None,
):
    repo, output = Path(repo), Path(output)
    # No directory is created before both actual, externally pinned approvals.
    data_raw = read_bound(data_freeze_path, expected_data_freeze_sha256)
    data_freeze = json.loads(data_raw)
    validate_data_freeze(data_freeze)
    source_raw = read_bound(source_freeze_path, expected_source_freeze_sha256)
    source_freeze = json.loads(source_raw)
    inventory = validate_source_freeze(repo, source_freeze)
    require(
        etf_data_freeze_path is not None and valid_sha(expected_etf_data_freeze_sha256),
        "EXTERNAL_ETF_FREEZE_REQUIRED_BEFORE_STAGING",
    )
    etf_raw = read_bound(etf_data_freeze_path, expected_etf_data_freeze_sha256)
    etf_freeze = json.loads(etf_raw)
    validate_etf_data_freeze(etf_freeze)
    for identity in (source_dataset_id, kernel_id):
        require(
            isinstance(identity, str)
            and re.fullmatch(r"aikyatansinha/[a-z0-9][a-z0-9-]+", identity) is not None,
            "EXPLICIT_OWNED_DATASET_AND_KERNEL_IDS_REQUIRED",
        )
    files = {flat_name(name): {"path": name, **details} for name, details in inventory.items()}
    for name, raw in (
        ("data-freeze.json", data_raw),
        ("etf-data-freeze.json", etf_raw),
        ("source-freeze.json", source_raw),
    ):
        files[name] = {"path": "provenance/" + name, "sha256": sha(raw), "bytes": len(raw)}
    template = (repo / "cloud/risk_forecast_runner.py").read_bytes()
    plan = {
        "schema": "quant_risk_development_plan_v1",
        "start_signal": "2007-02-28",
        "end_close": "2014-12-31",
        "calendar_start": "1962-01-01",
        "calendar_end": "2018-12-31",
        "cells_per_date": 96,
        "forecast_only": True,
        "service_budget_seconds": service_budget_seconds,
        "resume_policy": (
            "new_reviewed_transport_required_source_level_checkpoints_do_not_skip_input_preparation"
        ),
        "python": "3.12.15",
        "uv": "0.13.1",
        "data_freeze_sha256": expected_data_freeze_sha256,
        "etf_data_freeze_sha256": expected_etf_data_freeze_sha256,
        "etf_data_dataset": etf_freeze["dataset"],
        "etf_data_dataset_id": etf_freeze["dataset_id"],
        "etf_data_dataset_version": etf_freeze["version"],
        "etf_source_manifest_sha256": ETF_MANIFEST_SHA,
        "etf_raw_bundle_manifest_sha256": ETF_RAW_BUNDLE_SHA,
        "etf_price_basis": ETF_PRICE_BASIS,
        "source_freeze_sha256": expected_source_freeze_sha256,
        "runner_canonical_sha256": sha(canonical_runner(template)),
        "package_files": files,
        "runtime": runtime_from_lock((repo / "uv.lock").read_bytes()),
        "scientific_source_sha256": {
            name: inventory["src/quantlab/" + name]["sha256"] for name in SCIENTIFIC
        },
        "data_dataset": data_freeze["dataset"],
        "source_dataset": source_dataset_id,
        "kernel_id": kernel_id,
        "source_manifest_sha256": MANIFEST_SHA,
        "normalization_policy": "kernel_CRLF_to_LF_only_payload_raw_exact",
    }
    validate_plan(plan)
    plan_raw = canonical_json(plan)
    rendered = render_runner(template, sha(plan_raw))
    # Recheck source bytes after collecting the entire immutable package.
    require(production_inventory(repo) == inventory, "SOURCE_CHANGED_DURING_STAGING")
    payload = {
        flat: (repo / item["path"]).read_bytes()
        for flat, item in files.items()
        if not item["path"].startswith("provenance/")
    }
    payload.update(
        {
            "data-freeze.json": data_raw,
            "etf-data-freeze.json": etf_raw,
            "source-freeze.json": source_raw,
            "risk-forecast-plan.json": plan_raw,
        }
    )
    for flat, item in files.items():
        require(
            sha(payload[flat]) == item["sha256"] and len(payload[flat]) == item["bytes"],
            "COLLECTED_PAYLOAD_CHANGED",
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "payload").mkdir()
    (output / "kernel").mkdir()
    (output / "upload-metadata").mkdir()
    for name, raw in payload.items():
        (output / "payload" / name).write_bytes(raw)
    (output / "kernel/risk_forecast_runner.py").write_bytes(rendered)
    dataset_metadata = {
        "id": source_dataset_id,
        "title": "Quant risk forecast frozen source",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (output / "upload-metadata/dataset-metadata.json").write_bytes(canonical_json(dataset_metadata))
    kernel_metadata = {
        "id": kernel_id,
        "title": kernel_id.split("/")[-1].replace("-", " "),
        "code_file": "risk_forecast_runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [DATA_ID, ETF_DATA_ID, source_dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (output / "kernel/kernel-metadata.json").write_bytes(canonical_json(kernel_metadata))
    receipt = {
        "status": "staged_not_uploaded",
        "plan_sha256": sha(plan_raw),
        "runner_raw_sha256": sha(rendered),
        "runner_canonical_sha256": sha(canonical_runner(rendered)),
        "payload_sha256": {name: sha(raw) for name, raw in payload.items()},
        "source_dataset": source_dataset_id,
        "kernel_id": kernel_id,
        "reserved_metadata_not_bound_or_required_on_mount": True,
    }
    (output / "stage-receipt.json").write_bytes(canonical_json(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "output",
        "data-freeze-path",
        "etf-data-freeze-path",
        "expected-etf-data-freeze-sha256",
        "expected-data-freeze-sha256",
        "source-freeze-path",
        "expected-source-freeze-sha256",
        "source-dataset-id",
        "kernel-id",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--service-budget-seconds", required=True, type=int)
    args = vars(parser.parse_args())
    build_stage(**args)
