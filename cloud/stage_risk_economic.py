"""Source-only immutable staging after three actual data freezes and root source approval."""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path

from cloud import risk_forecast_runner as legacy
from cloud.risk_economic_runner import (
    PLAN_NAME,
    canonical_runner,
    encoded,
    read_bound,
    require,
    sha,
    validate_forecast_freeze,
    validate_plan,
)

CONFIG_FILES = (
    "configs/risk-development.toml",
    "configs/risk-economic-development.toml",
    "configs/execution-projection.json",
)
PROJECT_FILES = ("pyproject.toml", "uv.lock", ".python-version", "README.md")
TRANSPORT_FILES = (
    "cloud/risk_forecast_runner.py",
    "cloud/package_risk_economic_inputs.py",
    "cloud/risk_economic_runner.py",
    "cloud/stage_risk_economic.py",
)


def import_closure(repo):
    src = Path(repo) / "src"
    pending, found = [src / "quantlab/risk/economic_study.py"], set()
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
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join((*parts[: len(parts) - node.level + 1], base)).strip(".")
                names = [base, *(base + "." + alias.name for alias in node.names)]
            for name in names:
                if name == "quantlab" or name.startswith("quantlab."):
                    target = src.joinpath(*name.split("."))
                    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                        if candidate.is_file():
                            pending.append(candidate)
    return tuple(sorted(path.relative_to(Path(repo)).as_posix() for path in found))


def production_inventory(repo):
    paths = import_closure(repo) + CONFIG_FILES + PROJECT_FILES + TRANSPORT_FILES
    require(len(set(paths)) == len(paths), "UNIQUE_SOURCE_PACKAGE_REQUIRED")
    return {
        name: {
            "sha256": sha((Path(repo) / name).read_bytes()),
            "bytes": (Path(repo) / name).stat().st_size,
        }
        for name in paths
    }


def runtime_from_lock(raw):
    lock = tomllib.loads(raw.decode())
    names = (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "exchange-calendars",
        "cvxpy",
        "clarabel",
        "osqp",
        "threadpoolctl",
    )
    result = {}
    for name in names:
        versions = {package["version"] for package in lock["package"] if package["name"] == name}
        require(len(versions) == 1, "ONE_LOCKED_RUNTIME_VERSION_REQUIRED")
        result[name] = versions.pop()
    return result


def source_approval(repo, value):
    require(
        value.get("schema") == "quant_risk_economic_source_freeze_v1"
        and value.get("status") == "READY",
        "ACTUAL_ROOT_SOURCE_APPROVAL_REQUIRED",
    )
    inventory = production_inventory(repo)
    require(value.get("files") == inventory, "ACTUAL_SOURCE_BYTES_DIFFER_FROM_ROOT_APPROVAL")
    policy = value["solver_policy"]
    timestamp = datetime.fromisoformat(policy["frozen_at"])
    require(
        timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        and re.fullmatch(r"[0-9a-f]{64}", policy["policy_id"]),
        "EXPLICIT_FROZEN_SOLVER_POLICY_REQUIRED",
    )
    runtime = runtime_from_lock((Path(repo) / "uv.lock").read_bytes())
    require(
        dict(policy["expected_versions"])
        == {name: runtime[name] for name in ("cvxpy", "clarabel", "osqp", "numpy", "scipy")},
        "SOLVER_POLICY_DIFFERS_FROM_LOCK",
    )
    return inventory, runtime


def build_stage(
    repo,
    output,
    *,
    data_freeze_path,
    expected_data_freeze_sha256,
    etf_data_freeze_path,
    expected_etf_data_freeze_sha256,
    forecast_data_freeze_path,
    expected_forecast_data_freeze_sha256,
    source_freeze_path,
    expected_source_freeze_sha256,
    source_dataset_id,
    kernel_id,
    service_budget_seconds=36000,
):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    # Every required approval is read/authenticated before mkdir; absent forecast
    # publication proof is a refusal, never an invented positive dataset ID.
    frozen = {}
    for name, path, digest in (
        ("data-freeze.json", data_freeze_path, expected_data_freeze_sha256),
        ("etf-data-freeze.json", etf_data_freeze_path, expected_etf_data_freeze_sha256),
        (
            "forecast-data-freeze.json",
            forecast_data_freeze_path,
            expected_forecast_data_freeze_sha256,
        ),
        ("source-freeze.json", source_freeze_path, expected_source_freeze_sha256),
    ):
        frozen[name] = read_bound(path, digest)
    data, etf, forecasts, approval = [json.loads(frozen[name]) for name in frozen]
    legacy.validate_data_freeze(data)
    legacy.validate_etf_data_freeze(etf)
    validate_forecast_freeze(forecasts)
    inventory, runtime = source_approval(repo, approval)
    for name in (source_dataset_id, kernel_id):
        require(
            re.fullmatch(r"aikyatansinha/[a-z0-9][a-z0-9-]+", name),
            "EXPLICIT_OWNED_RESOURCE_REQUIRED",
        )
    files = {
        "__".join(Path(name).parts): {"path": name, **details}
        for name, details in inventory.items()
    }
    require(len(files) == len(inventory), "FLAT_SOURCE_NAME_COLLISION")
    for name, raw in frozen.items():
        files[name] = {"path": "provenance/" + name, "sha256": sha(raw), "bytes": len(raw)}
    template = (repo / "cloud/risk_economic_runner.py").read_bytes()
    plan = {
        "schema": "quant_risk_economic_plan_v1",
        "attempted_paths": 208,
        "fees_bps": [0, 5, 10, 25],
        "start_signal": "2007-02-28",
        "end_close": "2014-12-31",
        "holdout_outcomes": False,
        "python": "3.12.15",
        "threads": 4,
        "service_budget_seconds": service_budget_seconds,
        "resume": "none_this_transport_preserves_complete_path_checkpoints",
        "unsupported": "terminal_actual_prefix_no_full_score_no_COMPLETE_family",
        "runtime": runtime,
        "solver_policy": approval["solver_policy"],
        "package_files": files,
        "root_freeze_sha256": {name: sha(raw) for name, raw in frozen.items()},
        "runner_canonical_sha256": sha(canonical_runner(template)),
        "config_sha256": inventory["configs/risk-economic-development.toml"]["sha256"],
        "projection_sha256": inventory["configs/execution-projection.json"]["sha256"],
        "datasets": [legacy.DATA_ID, legacy.ETF_DATA_ID, forecasts["dataset"], source_dataset_id],
        "kernel_id": kernel_id,
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
        for flat, item in files.items()
        if not item["path"].startswith("provenance/")
    }
    payload.update(frozen)
    require(production_inventory(repo) == inventory, "SOURCE_CHANGED_DURING_STAGING")
    for name, item in files.items():
        require(
            sha(payload[name]) == item["sha256"] and len(payload[name]) == item["bytes"],
            "COLLECTED_PAYLOAD_DIFFERS",
        )
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("payload", "kernel", "upload-metadata"):
        (output / folder).mkdir()
    for name, raw in payload.items():
        (output / "payload" / name).write_bytes(raw)
    (output / "payload" / PLAN_NAME).write_bytes(plan_raw)
    (output / "kernel/risk_economic_runner.py").write_bytes(rendered)
    (output / "kernel/kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[-1].replace("-", " "),
                "code_file": "risk_economic_runner.py",
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
                "title": "Quant matched risk economics frozen source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            }
        )
    )
    receipt = {
        "status": "staged_not_uploaded",
        "plan_sha256": sha(plan_raw),
        "runner_raw_sha256": sha(rendered),
        "payload_files": files,
        "source_only": True,
        "datasets": plan["datasets"],
    }
    (output / "stage-receipt.json").write_bytes(encoded(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "output",
        "data-freeze-path",
        "expected-data-freeze-sha256",
        "etf-data-freeze-path",
        "expected-etf-data-freeze-sha256",
        "forecast-data-freeze-path",
        "expected-forecast-data-freeze-sha256",
        "source-freeze-path",
        "expected-source-freeze-sha256",
        "source-dataset-id",
        "kernel-id",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--service-budget-seconds", type=int, default=36000)
    build_stage(**vars(parser.parse_args()))
