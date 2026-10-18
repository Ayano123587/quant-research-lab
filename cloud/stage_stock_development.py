"""Stage stock development source only after externally pinned root approvals.

No upload, model fitting, runtime installation, or job submission occurs here.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from pathlib import Path

from cloud.stock_development_runner import (
    CONFIG_FILES,
    DATA_ID,
    MANIFEST_SHA,
    PREPARATION_SOURCES,
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
    validate_plan,
    validate_resume_freeze,
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
    seen, pending = set(), ["quantlab.stocks.development"]
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
        value.get("schema") == "quant_stock_source_freeze_v1" and value.get("status") == "READY",
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
        "lightgbm",
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
    compile(raw, "stock_development_runner.py", "exec")
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
    stock_spec_path="configs/stock-development.toml",
    resume_freeze_path=None,
    expected_resume_freeze_sha256=None,
):
    repo, output = Path(repo), Path(output)
    require(stock_spec_path in CONFIG_FILES, "EXPLICIT_REGISTERED_STOCK_SPEC_REQUIRED")
    # All root approvals are authenticated before any output directory exists.
    data_raw = read_bound(data_freeze_path, expected_data_freeze_sha256)
    data = json.loads(data_raw)
    validate_data_freeze(data)
    source_raw = read_bound(source_freeze_path, expected_source_freeze_sha256)
    inventory = validate_source_freeze(repo, json.loads(source_raw))
    for name, digest in PREPARATION_SOURCES.items():
        section = "stocks" if name in {"features", "labels", "prepare"} else "data"
        require(
            inventory[f"src/quantlab/{section}/{name}.py"]["sha256"] == digest,
            "CURRENT_PREPARATION_SOURCE_DIFFERS_FROM_COMPLETED_PANEL",
        )
    for identity in (source_dataset_id, kernel_id):
        require(
            isinstance(identity, str)
            and re.fullmatch(r"aikyatansinha/[a-z0-9][a-z0-9-]+", identity) is not None,
            "EXPLICIT_OWNED_SOURCE_DATASET_AND_KERNEL_IDS_REQUIRED",
        )
    require(
        (resume_freeze_path is None) == (expected_resume_freeze_sha256 is None),
        "RESUME_PATH_AND_EXTERNAL_SHA_REQUIRED_TOGETHER",
    )
    files = {flat_name(name): {"path": name, **details} for name, details in inventory.items()}
    approval_raw = {"data-freeze.json": data_raw, "source-freeze.json": source_raw}
    resume = None
    if resume_freeze_path is not None:
        raw = read_bound(resume_freeze_path, expected_resume_freeze_sha256)
        approval_raw["resume-freeze.json"] = raw
        resume = {"freeze_sha256": expected_resume_freeze_sha256}
    for name, raw in approval_raw.items():
        files[name] = {"path": "provenance/" + name, "sha256": sha(raw), "bytes": len(raw)}
    template = (repo / "cloud/stock_development_runner.py").read_bytes()
    plan = {
        "schema": "quant_stock_development_plan_v1",
        "stock_spec_path": stock_spec_path,
        "selection_start": "2007-01-01",
        "selection_information_cutoff": "2014-12-31",
        "calendar_start": "1962-01-01",
        "calendar_end": "2018-12-31",
        "candidate_count": 16,
        "prediction_and_intent_only": True,
        "holdout_outcome_values_permitted": False,
        "python": "3.12.15",
        "uv": "0.13.1",
        "uv_group": "ml",
        "threads": 4,
        "service_budget_seconds": 36000,
        "resume_policy": (
            "optional_external_root_freeze_with_exact_artifact_and_source_runtime_pins"
        ),
        "resume": resume,
        "data_freeze_sha256": expected_data_freeze_sha256,
        "source_freeze_sha256": expected_source_freeze_sha256,
        "runner_canonical_sha256": sha(canonical_runner(template)),
        "package_files": files,
        "runtime": runtime_from_lock((repo / "uv.lock").read_bytes()),
        "scientific_source_sha256": {
            name: inventory["src/quantlab/" + name]["sha256"] for name in SCIENTIFIC
        },
        "prepared_manifest_sha256": MANIFEST_SHA,
        "preparation_source_sha256": PREPARATION_SOURCES,
        "data_dataset": data["dataset"],
        "data_dataset_id": data["dataset_id"],
        "data_dataset_version": data["version"],
        "source_dataset": source_dataset_id,
        "kernel_id": kernel_id,
        "normalization_policy": "kernel_CRLF_to_LF_only_payload_raw_exact",
    }
    validate_plan(plan)
    if resume is not None:
        validate_resume_freeze(json.loads(approval_raw["resume-freeze.json"]), plan)
    plan_raw = canonical_json(plan)
    rendered = render_runner(template, sha(plan_raw))
    require(production_inventory(repo) == inventory, "SOURCE_CHANGED_DURING_STAGING")
    payload = {
        flat: (repo / item["path"]).read_bytes()
        for flat, item in files.items()
        if not item["path"].startswith("provenance/")
    }
    payload.update(approval_raw)
    payload["stock-development-plan.json"] = plan_raw
    for flat, item in files.items():
        require(
            sha(payload[flat]) == item["sha256"] and len(payload[flat]) == item["bytes"],
            "COLLECTED_PAYLOAD_CHANGED",
        )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("payload", "kernel", "upload-metadata"):
        (output / name).mkdir()
    for name, raw in payload.items():
        (output / "payload" / name).write_bytes(raw)
    (output / "kernel/stock_development_runner.py").write_bytes(rendered)
    (output / "upload-metadata/dataset-metadata.json").write_bytes(
        canonical_json(
            {
                "id": source_dataset_id,
                "title": "Quant stock prequential development source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            }
        )
    )
    sources = [DATA_ID, source_dataset_id]
    if resume is not None:
        sources.append(json.loads(approval_raw["resume-freeze.json"])["dataset"])
    (output / "kernel/kernel-metadata.json").write_bytes(
        canonical_json(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[-1].replace("-", " "),
                "code_file": "stock_development_runner.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": sources,
                "competition_sources": [],
                "kernel_sources": [],
            }
        )
    )
    receipt = {
        "status": "staged_not_uploaded",
        "plan_sha256": sha(plan_raw),
        "runner_raw_sha256": sha(rendered),
        "runner_canonical_sha256": sha(canonical_runner(rendered)),
        "payload_sha256": {name: sha(raw) for name, raw in payload.items()},
        "resume_freeze_sha256": expected_resume_freeze_sha256,
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
        "expected-data-freeze-sha256",
        "source-freeze-path",
        "expected-source-freeze-sha256",
        "source-dataset-id",
        "kernel-id",
    ):
        parser.add_argument("--" + name, required=True)
    for name in ("resume-freeze-path", "expected-resume-freeze-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--stock-spec-path", choices=CONFIG_FILES, default=CONFIG_FILES[0])
    build_stage(**vars(parser.parse_args()))
