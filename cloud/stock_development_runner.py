"""Authenticated prepared-stock development transport; no host fitting or fallback data.

Only standard-library code runs until the externally bound source, completed
prepared panel, and optional reviewed resume inventory are fully authenticated.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
RUNNER_CANONICAL_SHA256 = "UNBOUND"
DATA_ID = "aikyatansinha/portfolio-quant-stock-panel-v1"
DATASET_ID = 12657513
DATA_SHA = "251d2d8875b2e7d7f1b5f820c34153c3f7a8e1f6d5267d48183b60524860a2b3"
MANIFEST_SHA = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
SNAPSHOT_ID = "690e78739f0be396b61c1097e03795ce4d1b0217639e8fb2b70cf136b90c9f1e"
NORMALIZED_MANIFEST_SHA = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
ARCHIVE_SHA = "d9317c8fb2d63b9b00db5f933b6c9639d2bf7ea3b918169bb5cec5903dce85a1"
CALENDAR_SHA = "9e5c0a972ce2c7f83f5ffdde86234c96332807fb278eb8a3954422e8da7a83ce"
TRANSPORT_SHA = "dcd54e5cb1f49efa191d669a669c4235c54196f1d5ccb2669779e6d74fecb7c7"
CONTAINER_NAME = "stock-panel-v1.bin"
CONTAINER_BYTES = 507964547
PRICE_BASIS = "source_OHLC_adjustment_unverified"
SCIENTIFIC = (
    "stocks/development.py",
    "stocks/training_targets.py",
    "stocks/prepare.py",
    "stocks/features.py",
    "stocks/labels.py",
    "backtest/engine.py",
    "backtest/accounting.py",
    "data/archive.py",
    "data/calendar.py",
    "data/snapshot.py",
    "data/ingest.py",
    "data/returns.py",
    "data/provenance.py",
    "data/yahoo.py",
)
SOURCE_FILES = tuple("src/quantlab/" + p for p in SCIENTIFIC) + tuple(
    "src/quantlab/" + prefix + "__init__.py" for prefix in ("", "data/", "stocks/", "backtest/")
)
CONFIG_FILES = (
    "configs/stock-development.toml",
    "configs/stock-rank-target-development.toml",
)
PROJECT_FILES = ("pyproject.toml", "uv.lock", ".python-version", "README.md")
TRANSPORT_FILES = ("cloud/stock_development_runner.py", "cloud/stage_stock_development.py")
RUNTIME_NAMES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "exchange-calendars",
    "lightgbm",
    "threadpoolctl",
)

PREPARATION_SOURCES = {
    "archive": "7d8976a2b5edd10df9b62f8e7e4aff6323706984655bdb90c0f71462aeb4439c",
    "calendar": "350c4971cd4f5fce675e76e9d7375560e317bac57fa457e85904c76105c7d7ee",
    "features": "36662cc00d28d69e2465dce2ce28c0b0f40ca793f0a2fc1d572febe0da95596b",
    "ingest": "02b0168b94f8f4426ce989decd76c7bfcde65cb4e204025ee1b07f4d5df239f0",
    "labels": "e956dcc47eef517d868a5fe118207c405ce42e8d2275d177b3f8dc8d60070d57",
    "prepare": "717645105d5314ce485843b205b60bf9cad76254e834b30776c6003dd08de8e2",
    "returns": "4c760447953ab764c917db49428641b896f670db9c6f94cd40f4039f9a6f9748",
    "snapshot": "fdfbea76b8f571fe4e08a77c719380cb0df46bd2a05f4e508509dd75775785e6",
}


class TransportError(ValueError):
    """A bounded transport, runtime, or causal contract violation."""

    def __init__(self, reason, proof=None):
        super().__init__(reason)
        self.proof = proof


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def remaining_budget(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Startup-inclusive service budget exhausted; partial output retained")
    return seconds


def hash_stream(stream, budget_check=lambda: None):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        budget_check()
        digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def require(condition, reason):
    if not condition:
        raise TransportError(reason)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def safe_name(name):
    require(isinstance(name, str) and bool(name), "RELATIVE_PATH_REQUIRED")
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and ":" not in name
        and path.as_posix() == name,
        "UNSAFE_RELATIVE_PATH",
    )
    return path


def regular(path):
    path = Path(path)
    for item in (path, *path.parents):
        info = item.lstat()
        require(
            not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
            "SYMLINK_PATH_FORBIDDEN",
        )
    require(path.is_file(), "REGULAR_FILE_REQUIRED")
    return path


def read_bound(path, digest, size=None):
    raw = regular(path).read_bytes()
    actual = sha(raw)
    if not valid_sha(digest) or actual != digest:
        raise TransportError(
            "BOUND_FILE_HASH_MISMATCH",
            {
                "file": Path(path).name,
                "expected_sha256": digest,
                "actual_sha256": actual,
                "actual_bytes": len(raw),
            },
        )
    require(size is None or len(raw) == size, "BOUND_FILE_SIZE_MISMATCH")
    return raw


def canonical_runner(raw):
    require(b"\r" not in raw.replace(b"\r\n", b""), "LONE_CR_IN_RUNNER")
    raw = raw.replace(b"\r\n", b"\n")
    for field in (b"ROOT_PLAN_SHA256", b"RUNNER_CANONICAL_SHA256"):
        expression = rb"(?m)^" + field + rb' = "(?:UNBOUND|[0-9a-f]{64})"$'
        raw, count = re.subn(expression, field + b' = "' + b"0" * 64 + b'"', raw)
        require(count == 1, "EXACT_RUNNER_BINDING_LINES_REQUIRED")
    return raw


def materialize(authenticated, project):
    project = Path(project)
    project.mkdir(parents=True, exist_ok=False)
    for name, raw in authenticated.items():
        target = project.joinpath(*safe_name(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)


def write_json(path, value):
    raw = canonical_json(value)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def private_environment(private, project):
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("UV_", "PIP_", "PYTHON")) or name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            environment.pop(name)
    environment.update(
        {
            "UV_CACHE_DIR": str(private / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(private / "python"),
            "UV_PROJECT_ENVIRONMENT": str(private / "runtime/.venv"),
            "UV_TOOL_DIR": str(private / "tools"),
            "UV_TOOL_BIN_DIR": str(private / "bin"),
            "UV_NO_CONFIG": "1",
            "UV_NO_PROGRESS": "1",
            "UV_PYTHON_NO_REGISTRY": "1",
            "PIP_CACHE_DIR": str(private / "pip-cache"),
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(project / "src"),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    return environment


def bootstrap(private, project, deadline):
    environment = private_environment(private, project)
    uv_package = private / "uv-package"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(uv_package),
            "uv==0.13.1",
        ],
        check=True,
        env=environment,
        timeout=remaining_budget(deadline),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    uv = uv_package / "bin/uv"
    uv_version = subprocess.check_output(
        [str(uv), "--version"], env=environment, text=True, timeout=remaining_budget(deadline)
    ).strip()
    require(
        re.fullmatch(r"uv 0\.13\.1(?: \([^\r\n]+\))?", uv_version) is not None,
        "UV_VERSION_MISMATCH",
    )
    subprocess.run(
        [str(uv), "python", "install", "3.12.15"],
        check=True,
        env=environment,
        timeout=remaining_budget(deadline),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    interpreter = Path(
        subprocess.check_output(
            [str(uv), "python", "find", "--managed-python", "--no-project", "3.12.15"],
            env=environment,
            text=True,
            timeout=remaining_budget(deadline),
        ).strip()
    )
    require(
        interpreter.resolve().is_relative_to((private / "python").resolve()),
        "MANAGED_INTERPRETER_OUTSIDE_PRIVATE_ROOT",
    )
    subprocess.run(
        [
            str(uv),
            "sync",
            "--locked",
            "--no-install-project",
            "--no-default-groups",
            "--group",
            "ml",
            "--project",
            str(project),
            "--python",
            str(interpreter),
        ],
        check=True,
        env=environment,
        timeout=remaining_budget(deadline),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return (
        private / "runtime/.venv/bin/python",
        environment,
        {"actual_uv": uv_version, "managed_interpreter": str(interpreter)},
    )


def bootstrap_with_receipt(private, project, output, deadline):
    """Persist the bootstrap boundary even when pip/uv fails before child startup."""
    receipt = {
        "complete": False,
        "phase": "bootstrap",
        "expected_python": "3.12.15",
        "expected_uv": "0.13.1",
        "base_python": platform.python_version(),
    }
    write_json(Path(output) / "bootstrap-status.json", receipt)
    try:
        result = bootstrap(private, project, deadline)
    except BaseException as error:
        receipt.update(phase="bootstrap_failed", error_type=type(error).__name__)
        receipt.update(child_failure_detail(error))
        write_json(Path(output) / "bootstrap-status.json", receipt)
        raise
    receipt.update(phase="bootstrap_complete", complete=True, actual=result[2])
    write_json(Path(output) / "bootstrap-status.json", receipt)
    return result


def check_project(project, plan):
    for item in plan["package_files"].values():
        read_bound(Path(project) / item["path"], item["sha256"], item["bytes"])


def child_failure_detail(error):
    """Bounded private diagnostics; preserve the actual child exception context."""
    detail = {
        "complete": False,
        "error_type": type(error).__name__,
        "error_message": str(error)[:2000],
        "traceback": "".join(traceback.format_exception(error))[-12000:],
    }
    if isinstance(error, TransportError):
        detail.update(guard_reason=str(error), guard_proof=error.proof)
    return detail


def validate_data_freeze(value):
    require(
        value.get("schema") == "quant_stock_prepared_data_freeze_v1"
        and value.get("status") == "READY"
        and value.get("private") is True,
        "ACTUAL_READY_PRIVATE_PREPARED_DATA_FREEZE_REQUIRED",
    )
    require(
        value.get("dataset") == DATA_ID
        and type(value.get("dataset_id")) is int
        and value["dataset_id"] == DATASET_ID
        and type(value.get("version")) is int
        and value["version"] == 1,
        "ACTUAL_ACCEPTED_DATASET_ID_VERSION_REQUIRED",
    )
    require(
        value.get("container")
        == {"name": CONTAINER_NAME, "sha256": DATA_SHA, "bytes": CONTAINER_BYTES}
        and value.get("transport_manifest_sha256") == TRANSPORT_SHA,
        "REMOTE_WHOLE_CONTAINER_PROOF_REQUIRED",
    )
    require(
        value.get("prepared_manifest_sha256") == MANIFEST_SHA
        and value.get("original_archive_sha256") == ARCHIVE_SHA
        and value.get("original_snapshot_id") == SNAPSHOT_ID
        and value.get("source_normalized_manifest_sha256") == NORMALIZED_MANIFEST_SHA
        and value.get("calendar_sha256") == CALENDAR_SHA
        and value.get("preparation_source_sha256") == PREPARATION_SOURCES
        and valid_sha(value.get("remote_verification_receipt_sha256")),
        "ACTUAL_FULL_PREPARATION_SOURCE_PROOF_REQUIRED",
    )


def validate_resume_freeze(value, plan):
    """Optional previous actual cells need their own externally bound approval."""
    require(
        value.get("schema") == "quant_stock_development_resume_freeze_v1"
        and value.get("status") == "READY"
        and value.get("private") is True,
        "ACTUAL_READY_PRIVATE_RESUME_FREEZE_REQUIRED",
    )
    require(
        isinstance(value.get("dataset"), str)
        and re.fullmatch(r"aikyatansinha/[a-z0-9][a-z0-9-]+", value["dataset"]) is not None
        and type(value.get("dataset_id")) is int
        and value["dataset_id"] > 0
        and type(value.get("version")) is int
        and value["version"] >= 1,
        "ACTUAL_ACCEPTED_RESUME_DATASET_REQUIRED",
    )
    spec_item = next(
        item for item in plan["package_files"].values() if item["path"] == plan["stock_spec_path"]
    )
    require(
        value.get("prepared_manifest_sha256") == MANIFEST_SHA
        and value.get("source_freeze_sha256") == plan["source_freeze_sha256"]
        and value.get("spec_sha256") == spec_item["sha256"]
        and value.get("scientific_source_sha256") == plan["scientific_source_sha256"]
        and valid_sha(value.get("remote_verification_receipt_sha256")),
        "RESUME_SOURCE_CONFIG_PREPARED_PINS_DIFFER",
    )
    runtime = value.get("scientific_runtime", {})
    names = ("numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow", "threadpoolctl")
    require(
        set(runtime) == set(names) | {"python", "platform", "processor"}
        and all(runtime.get(name) == plan["runtime"][name] for name in names)
        and runtime.get("python") == "3.12.15"
        and all(isinstance(runtime.get(name), str) for name in ("platform", "processor")),
        "EXACT_PREVIOUS_SCIENTIFIC_RUNTIME_REQUIRED",
    )
    files, checkpoint = value.get("files", {}), value.get("checkpoint", {})
    require(isinstance(files, dict) and bool(files), "ACTUAL_RESUME_INVENTORY_REQUIRED")
    for name, item in files.items():
        relative = safe_name(name)
        require(
            relative.parts[0] in {"checkpoints", "fits", "models", "cells", "intents"}
            and valid_sha(item.get("sha256"))
            and type(item.get("bytes")) is int
            and item["bytes"] >= 0,
            "BOUND_RESUME_ARTIFACT_REQUIRED",
        )
    require(
        isinstance(checkpoint.get("path"), str)
        and safe_name(checkpoint["path"]).parts[0] == "checkpoints"
        and checkpoint["path"] in files
        and checkpoint.get("sha256") == files[checkpoint["path"]]["sha256"],
        "EXTERNALLY_BOUND_ACTUAL_CHECKPOINT_REQUIRED",
    )


def validate_plan(plan):
    require(plan.get("stock_spec_path") in CONFIG_FILES, "EXPLICIT_REGISTERED_STOCK_SPEC_REQUIRED")
    require(plan.get("schema") == "quant_stock_development_plan_v1", "UNSUPPORTED_PLAN")
    require("resume" in plan, "EXPLICIT_RESUME_NONE_OR_BOUND_FREEZE_REQUIRED")
    require(
        plan.get("selection_start") == "2007-01-01"
        and plan.get("selection_information_cutoff") == "2014-12-31"
        and plan.get("calendar_start") == "1962-01-01"
        and plan.get("calendar_end") == "2018-12-31"
        and plan.get("candidate_count") == 16
        and plan.get("prediction_and_intent_only") is True
        and plan.get("holdout_outcome_values_permitted") is False,
        "FULL_DEVELOPMENT_FAMILY_AND_NO_HOLDOUT_VALUES_REQUIRED",
    )
    require(
        plan.get("python") == "3.12.15"
        and plan.get("uv") == "0.13.1"
        and plan.get("uv_group") == "ml"
        and plan.get("threads") == 4
        and plan.get("service_budget_seconds") == 36000
        and type(plan["service_budget_seconds"]) is int,
        "LOCKED_ML_RUNTIME_AND_STARTUP_INCLUSIVE_BUDGET_REQUIRED",
    )
    require(
        all(
            valid_sha(plan.get(name))
            for name in ("data_freeze_sha256", "source_freeze_sha256", "runner_canonical_sha256")
        ),
        "EXTERNALLY_BOUND_ROOT_APPROVALS_REQUIRED",
    )
    require(
        plan.get("prepared_manifest_sha256") == MANIFEST_SHA
        and plan.get("preparation_source_sha256") == PREPARATION_SOURCES
        and plan.get("data_dataset") == DATA_ID
        and type(plan.get("data_dataset_id")) is int
        and plan["data_dataset_id"] == DATASET_ID
        and type(plan.get("data_dataset_version")) is int
        and plan["data_dataset_version"] == 1,
        "EXPLICIT_AUTHENTICATED_PREPARED_STOCK_INPUT_REQUIRED",
    )
    files = plan.get("package_files", {})
    expected = set(SOURCE_FILES + CONFIG_FILES + PROJECT_FILES + TRANSPORT_FILES) | {
        "provenance/data-freeze.json",
        "provenance/source-freeze.json",
    }
    if plan.get("resume") is not None:
        require(
            valid_sha(plan["resume"].get("freeze_sha256")), "EXTERNAL_RESUME_FREEZE_SHA_REQUIRED"
        )
        expected.add("provenance/resume-freeze.json")
    require(
        {v.get("path") for v in files.values()} == expected and len(files) == len(expected),
        "EXACT_AST_PLAIN_SOURCE_PACKAGE_REQUIRED",
    )
    for flat, item in files.items():
        require(
            safe_name(flat).name == flat and flat != "dataset-metadata.json",
            "PLAIN_FLAT_PACKAGE_NAME_REQUIRED",
        )
        safe_name(item["path"])
        require(
            valid_sha(item.get("sha256"))
            and type(item.get("bytes")) is int
            and 0 <= item["bytes"] <= 8_000_000,
            "BOUND_PACKAGE_BYTES_REQUIRED",
        )
    require(
        set(plan.get("scientific_source_sha256", {})) == set(SCIENTIFIC),
        "EXACT_SCIENTIFIC_SOURCE_CLOSURE_REQUIRED",
    )
    require(
        set(plan.get("runtime", {})) == set(RUNTIME_NAMES)
        and all(isinstance(v, str) and v for v in plan["runtime"].values()),
        "COMPLETE_LOCKED_ML_RUNTIME_REQUIRED",
    )


def authenticate_package(plan, mount):
    validate_plan(plan)
    require(
        {p.name for p in Path(mount).iterdir()} - {"dataset-metadata.json"}
        == set(plan["package_files"]) | {"stock-development-plan.json"},
        "EXACT_MOUNTED_PLAIN_SOURCE_REQUIRED",
    )
    result = {
        item["path"]: read_bound(Path(mount) / flat, item["sha256"], item["bytes"])
        for flat, item in plan["package_files"].items()
    }
    data = result["provenance/data-freeze.json"]
    require(sha(data) == plan["data_freeze_sha256"], "ROOT_DATA_FREEZE_HASH_MISMATCH")
    freeze = json.loads(data)
    validate_data_freeze(freeze)
    require(
        plan["data_dataset"] == freeze["dataset"]
        and plan["data_dataset_id"] == freeze["dataset_id"]
        and plan["data_dataset_version"] == freeze["version"],
        "PLAN_DATA_PROOF_DIFFERS",
    )
    source = result["provenance/source-freeze.json"]
    require(sha(source) == plan["source_freeze_sha256"], "ROOT_SOURCE_FREEZE_HASH_MISMATCH")
    source = json.loads(source)
    inventory = {
        item["path"]: {"sha256": item["sha256"], "bytes": item["bytes"]}
        for item in plan["package_files"].values()
        if not item["path"].startswith("provenance/")
    }
    require(
        source.get("schema") == "quant_stock_source_freeze_v1"
        and source.get("status") == "READY"
        and source.get("files") == inventory,
        "CURRENT_SOURCE_DIFFERS_FROM_ROOT_FREEZE",
    )
    for name, digest in plan["scientific_source_sha256"].items():
        require(sha(result["src/quantlab/" + name]) == digest, "SCIENTIFIC_SOURCE_HASH_DIFFERS")
    require(
        sha(canonical_runner(result["cloud/stock_development_runner.py"]))
        == plan["runner_canonical_sha256"],
        "PAYLOAD_RUNNER_TEMPLATE_DIFFERS",
    )
    if plan["resume"] is not None:
        raw = result["provenance/resume-freeze.json"]
        require(sha(raw) == plan["resume"]["freeze_sha256"], "RESUME_FREEZE_HASH_MISMATCH")
        validate_resume_freeze(json.loads(raw), plan)
    return result


def inspect_container(archive, transport):
    require(
        transport.get("schema") == "quant_full_stock_panel_transport_v1"
        and transport.get("container") == CONTAINER_NAME
        and transport.get("container_bytes") == CONTAINER_BYTES
        and transport.get("container_sha256") == DATA_SHA
        and transport.get("container_format") == "ZIP_STORED_explicit_bin_extension"
        and transport.get("prepared_manifest_sha256") == MANIFEST_SHA
        and transport.get("original_archive_sha256") == ARCHIVE_SHA
        and transport.get("original_snapshot_id") == SNAPSHOT_ID
        and transport.get("preparation_source_sha256") == PREPARATION_SOURCES
        and transport.get("research_scope_complete") is True
        and transport.get("roundtrip_all_member_sha256_verified") is True,
        "EXACT_FULL_PREPARED_STOCK_TRANSPORT_REQUIRED",
    )
    require(
        transport.get("counts")
        == {
            "candidates_rows": 1539730,
            "features_rows": 88309,
            "labels_rows": 1539730,
            "empty_source_histories": 32,
            "source_histories_read": 7195,
            "source_rows_read": 14887665,
        },
        "FULL_OBSERVED_STOCK_POPULATION_REQUIRED",
    )
    expected = transport.get("members", {})
    members = archive.infolist()
    names = [m.filename for m in members]
    require(
        len(names) == 648
        and len(set(names)) == 648
        and set(names) == set(expected)
        and expected.get("manifest.json", {}).get("sha256") == MANIFEST_SHA,
        "EXACT_ALL_648_PREPARED_MEMBERS_REQUIRED",
    )
    for member in members:
        safe_name(member.filename)
        item = expected[member.filename]
        require(
            not member.is_dir()
            and not member.flag_bits & 1
            and stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG)
            and member.compress_type == zipfile.ZIP_STORED
            and valid_sha(item.get("sha256"))
            and type(item.get("bytes")) is int
            and item["bytes"] == member.file_size,
            "UNSAFE_PREPARED_CONTAINER_MEMBER",
        )
    return members


def extract_data(container, transport_raw, destination, budget_check=lambda: None):
    require(sha(transport_raw) == TRANSPORT_SHA, "TRANSPORT_MANIFEST_HASH_MISMATCH")
    transport = json.loads(transport_raw)
    container = regular(container)
    require(container.stat().st_size == CONTAINER_BYTES, "CONTAINER_SIZE_MISMATCH")
    with container.open("rb") as stream:
        actual = hash_stream(stream, budget_check)
        require(actual == DATA_SHA, "WHOLE_PREPARED_CONTAINER_HASH_MISMATCH")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = inspect_container(archive, transport)
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=False)
            for member in members:
                budget_check()
                path = destination.joinpath(*safe_name(member.filename).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(member) as source, path.open("xb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        budget_check()
                        digest.update(block)
                        target.write(block)
                require(
                    digest.hexdigest() == transport["members"][member.filename]["sha256"],
                    "PREPARED_MEMBER_HASH_MISMATCH",
                )
    return {
        "source_type": "completed_full_prepared_stock_panel",
        "binary_sha256": actual,
        "binary_bytes": CONTAINER_BYTES,
        "transport_manifest_sha256": TRANSPORT_SHA,
        "prepared_manifest_sha256": MANIFEST_SHA,
        "original_archive_sha256": ARCHIVE_SHA,
        "source_normalized_manifest_sha256": NORMALIZED_MANIFEST_SHA,
        "preparation_source_sha256": PREPARATION_SOURCES,
        "authenticated_members": 648,
        "price_basis": PRICE_BASIS,
        "outcome_value_parsing": False,
    }


def authenticate_resume(mount, freeze, destination=None, budget_check=lambda: None):
    """Hash previous actual artifacts; no model or prediction values are parsed."""
    mount = Path(mount)
    expected = freeze["files"]
    actual = {p.relative_to(mount).as_posix() for p in mount.rglob("*") if p.is_file()}
    require(actual - {"dataset-metadata.json"} == set(expected), "EXACT_RESUME_MOUNT_REQUIRED")
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
    for name, item in expected.items():
        budget_check()
        source = regular(mount.joinpath(*safe_name(name).parts))
        require(source.stat().st_size == item["bytes"], "RESUME_ARTIFACT_SIZE_DIFFERS")
        target = destination.joinpath(*safe_name(name).parts) if destination else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream:
            if target is None:
                digest = hash_stream(stream, budget_check)
            else:
                value = hashlib.sha256()
                with target.open("xb") as output:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        budget_check()
                        value.update(block)
                        output.write(block)
                digest = value.hexdigest()
        require(digest == item["sha256"], "RESUME_ARTIFACT_HASH_DIFFERS")


def child(plan_path, project, prepared_root, resume_root, output):
    plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
    validate_plan(plan)
    check_project(project, plan)
    require(
        sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
        "CHILD_CANONICAL_SELF_MISMATCH",
    )
    from importlib.metadata import PackageNotFoundError, version

    runtime = {}
    for name in plan["runtime"]:
        try:
            runtime[name] = version(name)
        except PackageNotFoundError:
            runtime[name] = None
    write_json(
        Path(output) / "runtime-before.json",
        {
            "expected": plan["runtime"],
            "actual": runtime,
            "python": platform.python_version(),
            "executable": sys.executable,
        },
    )
    require(
        platform.python_version() == "3.12.15" and runtime == plan["runtime"],
        "EXACT_LOCKED_CHILD_ML_RUNTIME_REQUIRED",
    )
    deadline = float(os.environ["QUANTLAB_SERVICE_DEADLINE"]) - 60
    remaining_budget(deadline)
    resume = None
    if plan["resume"] is not None:
        resume = json.loads((Path(project) / "provenance/resume-freeze.json").read_bytes())
        validate_resume_freeze(resume, plan)
        authenticate_resume(resume_root, resume, budget_check=lambda: remaining_budget(deadline))
        actual = {
            name: runtime[name]
            for name in ("numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow", "threadpoolctl")
        }
        actual.update(
            python=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor(),
        )
        require(actual == resume["scientific_runtime"], "PREVIOUS_RUNTIME_MISMATCH_NO_OVERRIDE")
    else:
        require(resume_root == "-", "UNAPPROVED_RESUME_ROOT_FORBIDDEN")
    # Parent has authenticated the entire prepared BIN and all 648 members.
    # Optional previous cell/model artifacts were also hashed before native imports.
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.stocks.development import run_stock_development
    from quantlab.stocks.prepare import verify_prepared_panel

    require(
        Path(sys.modules[run_stock_development.__module__].__file__).resolve()
        == (Path(project) / "src/quantlab/stocks/development.py").resolve(),
        "PROJECT_IMPORT_PATH_MISMATCH",
    )
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    require(
        schedule.provenance["schedule_sha256"] == CALENDAR_SHA,
        "EXACT_PREPARATION_CALENDAR_REQUIRED",
    )
    kwargs = {"service_deadline_monotonic": deadline}
    if resume is not None:
        kwargs.update(
            resume_checkpoint=Path(resume_root) / resume["checkpoint"]["path"],
            expected_resume_sha256=resume["checkpoint"]["sha256"],
        )
    result = run_stock_development(
        prepared_root,
        MANIFEST_SHA,
        Path(project) / plan["stock_spec_path"],
        schedule,
        Path(output) / "study",
        **kwargs,
    )
    require(
        result.get("complete") is True and len(result.get("candidate_ids", [])) == 16,
        "FULL_REGISTERED_SCIENTIFIC_GRID_INCOMPLETE",
    )
    actual_dates = {p.stem for p in (Path(output) / "study/predictions").glob("*.parquet")}
    expected_dates = {
        schedule.month_end(year, month).session.isoformat()
        for year in range(2007, 2015)
        for month in range(1, 13)
    }
    require(actual_dates == expected_dates, "ALL_96_VALIDATION_DATES_REQUIRED")
    require(
        len(tuple((Path(output) / "study/checkpoints").glob("*.json"))) == 96 * 16,
        "FULL_1536_COMPLETED_CELLS_REQUIRED",
    )
    require(
        all(Path(p).stem < "2015-01-01" for p in result["outcome_value_partitions_read"]),
        "HOLDOUT_OUTCOME_VALUE_PARTITION_FORBIDDEN",
    )
    remaining_budget(deadline)
    check_project(project, plan)
    verify_prepared_panel(prepared_root, MANIFEST_SHA)
    if resume is not None:
        authenticate_resume(resume_root, resume, budget_check=lambda: remaining_budget(deadline))
    after = {name: version(name) for name in plan["runtime"]}
    write_json(
        Path(output) / "runtime-after.json",
        {"expected": runtime, "actual": after, "python": platform.python_version()},
    )
    require(after == runtime, "NUMERICAL_RUNTIME_CHANGED")
    remaining_budget(deadline)
    write_json(
        Path(output) / "child-complete.json",
        {
            "complete": True,
            "runtime": runtime,
            "prepared_manifest_sha256": MANIFEST_SHA,
            "original_archive_sha256": ARCHIVE_SHA,
            "price_basis": PRICE_BASIS,
            "study_manifest_sha256": sha((Path(output) / "study/manifest.json").read_bytes()),
            "source_postflight": "exact",
            "candidate_cells": 1536,
            "validation_dates": 96,
            "selection_information_cutoff": "2014-12-31",
            "holdout_outcome_values": False,
            "economic_execution": "not_run_target_intentions_only",
        },
    )


def run(input_root, output_root):
    started = time.monotonic()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    state = {"complete": False, "phase": "startup", "started_at": datetime.now(UTC).isoformat()}
    write_json(output / "status.json", state)
    try:
        require(
            valid_sha(ROOT_PLAN_SHA256) and valid_sha(RUNNER_CANONICAL_SHA256),
            "ROOT_BINDINGS_REQUIRED",
        )
        require(
            sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
            "CANONICAL_SELF_MISMATCH",
        )
        plans = [
            p
            for p in Path(input_root).rglob("stock-development-plan.json")
            if sha(regular(p).read_bytes()) == ROOT_PLAN_SHA256
        ]
        require(len(plans) == 1, "ONE_EXTERNALLY_BOUND_PLAN_REQUIRED")
        plan_path = plans[0]
        plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
        validate_plan(plan)
        deadline = started + plan["service_budget_seconds"]

        def budget():
            return remaining_budget(deadline)

        budget()
        authenticated = authenticate_package(plan, plan_path.parent)
        transports = [
            p
            for p in Path(input_root).rglob("transport-manifest.json")
            if sha(regular(p).read_bytes()) == TRANSPORT_SHA
        ]
        require(len(transports) == 1, "ONE_BOUND_PREPARED_PANEL_TRANSPORT_REQUIRED")
        data_mount = transports[0].parent
        private = Path(tempfile.mkdtemp(prefix="quant-stock-development-", dir="/tmp"))
        project, prepared = private / "project", private / "prepared-stocks"
        materialize(authenticated, project)
        state.update(
            phase="prepared_data_authentication",
            root_plan_sha256=ROOT_PLAN_SHA256,
            scientific_source_sha256=plan["scientific_source_sha256"],
        )
        write_json(output / "status.json", state)
        proof = extract_data(
            data_mount / CONTAINER_NAME,
            read_bound(transports[0], TRANSPORT_SHA),
            prepared,
            budget_check=budget,
        )
        write_json(output / "prepared-data-authentication.json", proof)
        resume_mount, resume_root, resume = None, "-", None
        if plan["resume"] is not None:
            resume = json.loads(authenticated["provenance/resume-freeze.json"])
            checkpoint = resume["checkpoint"]
            relative = safe_name(checkpoint["path"])
            matches = [
                p
                for p in Path(input_root).rglob(relative.name)
                if sha(regular(p).read_bytes()) == checkpoint["sha256"]
            ]
            require(len(matches) == 1, "ONE_EXTERNALLY_BOUND_RESUME_CHECKPOINT_REQUIRED")
            resume_mount = matches[0].parents[len(relative.parts) - 1]
            resume_root = private / "reviewed-resume"
            authenticate_resume(resume_mount, resume, resume_root, budget)
            write_json(
                output / "resume-authentication.json",
                {
                    "freeze_sha256": plan["resume"]["freeze_sha256"],
                    "checkpoint": checkpoint,
                    "authenticated_artifacts": len(resume["files"]),
                    "model_values_parsed": False,
                },
            )
        state.update(phase="private_locked_ml_runtime_bootstrap")
        write_json(output / "status.json", state)
        interpreter, environment, bootstrap_actual = bootstrap_with_receipt(
            private, project, output, deadline
        )
        environment["QUANTLAB_SERVICE_DEADLINE"] = str(deadline)
        state.update(phase="development_only", bootstrap=bootstrap_actual)
        write_json(output / "status.json", state)
        subprocess.run(
            [
                str(interpreter),
                "-s",
                "-P",
                str(Path(__file__).resolve()),
                "--child",
                str(plan_path),
                str(project),
                str(prepared),
                str(resume_root),
                str(output),
            ],
            check=True,
            env=environment,
            cwd=project,
            timeout=budget(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        budget()
        read_bound(plan_path, ROOT_PLAN_SHA256)
        authenticate_package(plan, plan_path.parent)
        check_project(project, plan)
        read_bound(transports[0], TRANSPORT_SHA)
        with regular(data_mount / CONTAINER_NAME).open("rb") as stream:
            require(hash_stream(stream, budget) == DATA_SHA, "POSTFLIGHT_CONTAINER_HASH_MISMATCH")
        if resume is not None:
            authenticate_resume(resume_mount, resume, budget_check=budget)
        require(
            sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
            "POSTFLIGHT_SELF_MISMATCH",
        )
        child_result = json.loads((output / "child-complete.json").read_bytes())
        require(child_result.get("complete") is True, "CHILD_POSTFLIGHT_INCOMPLETE")
        budget()
        state.update(
            phase="complete",
            complete=True,
            child=child_result,
            finished_at=datetime.now(UTC).isoformat(),
        )
        write_json(output / "status.json", state)
        write_json(
            output / "COMPLETE.json", {"status_sha256": sha((output / "status.json").read_bytes())}
        )
    except BaseException as error:
        state.update(
            phase="failed", finished_at=datetime.now(UTC).isoformat(), **child_failure_detail(error)
        )
        write_json(output / "status.json", state)
        write_json(output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--child":
        try:
            child(*sys.argv[2:])
        except BaseException as error:
            write_json(Path(sys.argv[6]) / "child-failure.json", child_failure_detail(error))
            raise
    else:
        run("/kaggle/input", "/kaggle/working/stock-development")
