"""Authenticated plain-source transport for the common-probe risk development forecasts.

This file is also the kernel template. Only its two explicit binding lines are
rendered. Nothing in this module imports project code or native data libraries
until the externally pinned package and data have been authenticated.
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
UTC = UTC

DATA_ID = "aikyatansinha/portfolio-quant-lab-market-v1"
DATA_SHA = "28c680041c0fdc90ff9f639eb842d677dbc5a366f0c96b576aeba7a627e490d5"
MANIFEST_SHA = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
SNAPSHOT_ID = "690e78739f0be396b61c1097e03795ce4d1b0217639e8fb2b70cf136b90c9f1e"
ARCHIVE_SHA = "d9317c8fb2d63b9b00db5f933b6c9639d2bf7ea3b918169bb5cec5903dce85a1"
TRANSPORT_SHA = "2bf74b7981788d29fabd79d7d01bf70ba784797370b2865943f376d9f23d2740"
ETF_DATA_ID = "aikyatansinha/portfolio-quant-etf-price-index-v1"
ETF_DATASET_ID = 12657227
ETF_DATA_SHA = "60f221014cf7029eeee09f80117b19f5a25cedd6965cffebe74e539d05123b3e"
ETF_MANIFEST_SHA = "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3"
ETF_RAW_BUNDLE_SHA = "c7cc178eb60b3674388deac4b84315d6776ed66800ba5f7e5418cbf6f736c240"
ETF_CALENDAR_SHA = "9e5c0a972ce2c7f83f5ffdde86234c96332807fb278eb8a3954422e8da7a83ce"
ETF_PRICE_BASIS = "provider_adjusted_close_ratio_price_index_research_proxy"
ETF_CONTAINER_NAME = "yahoo-etf-price-index-v1.bin"
ETF_TRANSPORT_SHA = "b8d72ecdb5392e6ad98ed8d6b8dec2dcf6c95fe126ece3637f337bf0b5bb2396"

SCIENTIFIC = (
    "risk/study.py",
    "risk/covariance.py",
    "data/archive.py",
    "data/calendar.py",
    "data/snapshot.py",
    "data/ingest.py",
    "data/returns.py",
    "data/provenance.py",
    "data/yahoo.py",
)
SOURCE_FILES = tuple("src/quantlab/" + p for p in SCIENTIFIC) + tuple(
    "src/quantlab/" + prefix + "__init__.py" for prefix in ("", "data/", "risk/")
)
CONFIG_FILES = ("configs/risk-development.toml",)
PROJECT_FILES = ("pyproject.toml", "uv.lock", ".python-version", "README.md")
TRANSPORT_FILES = ("cloud/risk_forecast_runner.py", "cloud/stage_risk_forecast.py")


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


def validate_data_freeze(value):
    require(
        value.get("schema") == "quant_market_data_freeze_v1"
        and value.get("status") == "READY"
        and value.get("private") is True,
        "ACTUAL_READY_PRIVATE_DATA_FREEZE_REQUIRED",
    )
    dataset = value.get("dataset", {})
    require(
        dataset.get("canonical") == DATA_ID
        and dataset.get("status") == "READY"
        and dataset.get("is_private") is True
        and type(dataset.get("id")) is int
        and dataset["id"] == 12654575
        and type(dataset.get("version")) is int
        and dataset["version"] == 1,
        "ACCEPTED_READY_DATASET_ID_AND_VERSION_REQUIRED",
    )
    remote = value.get("remote_files", {})
    require(
        set(remote) == {"market-v1.bin", "transport-manifest.json"},
        "EXACT_REMOTE_BINARY_AND_MANIFEST_PROOF_REQUIRED",
    )
    require(
        remote["market-v1.bin"] == {"sha256": DATA_SHA, "bytes": 1417949089}
        and remote["transport-manifest.json"] == {"sha256": TRANSPORT_SHA, "bytes": 1170525},
        "REMOTE_ALL_BYTE_PROOF_MISMATCH",
    )
    require(
        value.get("snapshot_manifest_sha256") == MANIFEST_SHA
        and value.get("snapshot_id") == SNAPSHOT_ID
        and value.get("archive_sha256") == ARCHIVE_SHA
        and valid_sha(value.get("remote_verification_receipt_sha256")),
        "ROOT_REMOTE_VERIFICATION_RECEIPT_REQUIRED",
    )


def validate_etf_data_freeze(value):
    require(
        value.get("schema") == "quant_yahoo_etf_data_freeze_v1"
        and value.get("status") == "READY"
        and value.get("private") is True,
        "ACTUAL_READY_PRIVATE_DATA_FREEZE_REQUIRED",
    )
    require(
        value.get("dataset") == ETF_DATA_ID
        and type(value.get("dataset_id")) is int
        and value["dataset_id"] == ETF_DATASET_ID
        and type(value.get("version")) is int
        and value["version"] == 1,
        "ACCEPTED_READY_DATASET_ID_AND_VERSION_REQUIRED",
    )
    require(
        value.get("container")
        == {"name": ETF_CONTAINER_NAME, "sha256": ETF_DATA_SHA, "bytes": 21376508}
        and value.get("transport_manifest_sha256") == ETF_TRANSPORT_SHA,
        "REMOTE_ALL_BYTE_PROOF_MISMATCH",
    )
    require(
        value.get("release_manifest_sha256") == ETF_MANIFEST_SHA
        and value.get("raw_bundle_manifest_sha256") == ETF_RAW_BUNDLE_SHA
        and value.get("calendar_sha256") == ETF_CALENDAR_SHA
        and valid_sha(value.get("remote_verification_receipt_sha256")),
        "ROOT_REMOTE_VERIFICATION_RECEIPT_REQUIRED",
    )


def inspect_etf_container(archive, transport):
    """Inspect only after authenticating the entire immutable Yahoo binary."""
    require(
        transport.get("schema") == "quant_yahoo_etf_release_transport_v1"
        and transport.get("container") == ETF_CONTAINER_NAME
        and transport.get("container_sha256") == ETF_DATA_SHA
        and transport.get("container_bytes") == 21376508
        and transport.get("container_format") == "ZIP_STORED_explicit_bin_extension"
        and transport.get("release_manifest_sha256") == ETF_MANIFEST_SHA
        and transport.get("raw_bundle_manifest_sha256") == ETF_RAW_BUNDLE_SHA
        and transport.get("calendar_sha256") == ETF_CALENDAR_SHA
        and transport.get("all_member_roundtrip_sha256_verified") is True,
        "EXACT_BINARY_TRANSPORT_REQUIRED",
    )
    expected = transport.get("members", {})
    members = archive.infolist()
    names = [m.filename for m in members]
    require(
        len(names) == 33 and len(set(names)) == 33 and set(names) == set(expected),
        "EXACT_SAFE_ZIP_MEMBER_INVENTORY_REQUIRED",
    )
    require(
        expected.get("manifest.json", {}).get("sha256") == ETF_MANIFEST_SHA,
        "RELEASE_MEMBER_PINS_DIFFER",
    )
    for member in members:
        safe_name(member.filename)
        item = expected[member.filename]
        require(
            not member.is_dir()
            and stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG)
            and not member.flag_bits & 1
            and member.compress_type == zipfile.ZIP_STORED
            and valid_sha(item.get("sha256"))
            and type(item.get("bytes")) is int
            and item["bytes"] == member.file_size,
            "UNSAFE_ZIP_MEMBER",
        )
    return members


def extract_etf_data(container, transport_raw, destination, budget_check=lambda: None):
    require(sha(transport_raw) == ETF_TRANSPORT_SHA, "TRANSPORT_MANIFEST_HASH_MISMATCH")
    transport = json.loads(transport_raw)
    container = regular(container)
    require(container.stat().st_size == 21376508, "BINARY_SIZE_MISMATCH")
    budget_check()
    with container.open("rb") as stream:
        actual_binary_sha = hash_stream(stream)
        require(actual_binary_sha == ETF_DATA_SHA, "WHOLE_BINARY_HASH_MISMATCH")
        budget_check()
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = inspect_etf_container(archive, transport)
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=False)
            for member in members:
                budget_check()
                path = destination.joinpath(*safe_name(member.filename).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(member) as source, path.open("xb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        budget_check()
                        digest.update(chunk)
                        target.write(chunk)
                require(
                    digest.hexdigest() == transport["members"][member.filename]["sha256"],
                    "ZIP_MEMBER_HASH_MISMATCH",
                )
    return {
        "source_type": "yahoo_etf_release",
        "binary_sha256": actual_binary_sha,
        "binary_bytes": container.stat().st_size,
        "transport_sha256": sha(transport_raw),
        "authenticated_members": len(members),
        "release_manifest_sha256": ETF_MANIFEST_SHA,
        "raw_bundle_manifest_sha256": ETF_RAW_BUNDLE_SHA,
        "calendar_sha256": ETF_CALENDAR_SHA,
        "price_basis": ETF_PRICE_BASIS,
        "archive_sha256": None,
    }


def validate_plan(plan):
    require(plan.get("schema") == "quant_risk_development_plan_v1", "UNSUPPORTED_PLAN")
    require(
        plan.get("start_signal") == "2007-02-28"
        and plan.get("end_close") == "2014-12-31"
        and plan.get("cells_per_date") == 96
        and plan.get("calendar_start") == "1962-01-01"
        and plan.get("calendar_end") == "2018-12-31",
        "DEVELOPMENT_BOUNDARY_OR_CONTROLS_CHANGED",
    )
    require(
        plan.get("python") == "3.12.15" and plan.get("uv") == "0.13.1",
        "PINNED_PRIVATE_RUNTIME_REQUIRED",
    )
    require(
        type(plan.get("service_budget_seconds")) is int
        and 600 <= plan["service_budget_seconds"] <= 36000
        and plan.get("forecast_only") is True
        and plan.get("resume_policy")
        == "new_reviewed_transport_required_source_level_checkpoints_do_not_skip_input_preparation",
        "BOUNDED_FORECAST_ONLY_SERVICE_POLICY_REQUIRED",
    )
    require(
        valid_sha(plan.get("data_freeze_sha256"))
        and valid_sha(plan.get("source_freeze_sha256"))
        and valid_sha(plan.get("etf_data_freeze_sha256"))
        and valid_sha(plan.get("runner_canonical_sha256")),
        "ACTUAL_FREEZE_HASHES_REQUIRED",
    )
    require(
        plan.get("etf_source_manifest_sha256") == ETF_MANIFEST_SHA
        and plan.get("etf_raw_bundle_manifest_sha256") == ETF_RAW_BUNDLE_SHA
        and plan.get("etf_price_basis") == ETF_PRICE_BASIS
        and plan.get("etf_data_dataset") == ETF_DATA_ID
        and type(plan.get("etf_data_dataset_id")) is int
        and plan["etf_data_dataset_id"] == ETF_DATASET_ID
        and plan.get("etf_data_dataset_version") == 1,
        "EXPLICIT_SEPARATE_YAHOO_ETF_PROVENANCE_REQUIRED",
    )
    files = plan.get("package_files", {})
    expected = set(SOURCE_FILES + CONFIG_FILES + PROJECT_FILES + TRANSPORT_FILES) | {
        "provenance/data-freeze.json",
        "provenance/etf-data-freeze.json",
        "provenance/source-freeze.json",
    }
    require(
        {v.get("path") for v in files.values()} == expected and len(files) == len(expected),
        "EXACT_PLAIN_PACKAGE_CLOSURE_REQUIRED",
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
            "EXACT_PACKAGE_BYTES_REQUIRED",
        )
    require(
        set(plan.get("scientific_source_sha256", {})) == set(SCIENTIFIC),
        "EXACT_RISK_SCIENTIFIC_SOURCE_CLOSURE_REQUIRED",
    )
    require(
        set(plan.get("runtime", {}))
        == {
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "scikit-learn",
            "exchange-calendars",
            "threadpoolctl",
        }
        and all(isinstance(v, str) and v for v in plan["runtime"].values()),
        "COMPLETE_LOCKED_RUNTIME_INVENTORY_REQUIRED",
    )


def authenticate_package(plan, mount):
    """Hash every plain payload before creating the project or importing it."""
    validate_plan(plan)
    actual_names = {p.name for p in Path(mount).iterdir()}
    require(
        actual_names - {"dataset-metadata.json"}
        == set(plan["package_files"]) | {"risk-forecast-plan.json"},
        "EXACT_MOUNTED_PLAIN_PACKAGE_REQUIRED",
    )
    authenticated = {}
    for flat, item in plan["package_files"].items():
        authenticated[item["path"]] = read_bound(Path(mount) / flat, item["sha256"], item["bytes"])
    freeze_raw = authenticated["provenance/data-freeze.json"]
    require(sha(freeze_raw) == plan["data_freeze_sha256"], "ROOT_DATA_FREEZE_HASH_MISMATCH")
    validate_data_freeze(json.loads(freeze_raw))
    etf_freeze_raw = authenticated["provenance/etf-data-freeze.json"]
    require(
        sha(etf_freeze_raw) == plan["etf_data_freeze_sha256"], "ROOT_ETF_DATA_FREEZE_HASH_MISMATCH"
    )
    etf_freeze = json.loads(etf_freeze_raw)
    validate_etf_data_freeze(etf_freeze)
    require(
        plan["etf_data_dataset"] == etf_freeze["dataset"]
        and plan["etf_data_dataset_id"] == etf_freeze["dataset_id"]
        and plan["etf_data_dataset_version"] == etf_freeze["version"],
        "PLAN_ETF_DATA_PROOF_DIFFERS",
    )
    require(
        sha(authenticated["provenance/source-freeze.json"]) == plan["source_freeze_sha256"],
        "ROOT_SOURCE_FREEZE_HASH_MISMATCH",
    )
    source_freeze = json.loads(authenticated["provenance/source-freeze.json"])
    expected_source = {
        item["path"]: {"sha256": item["sha256"], "bytes": item["bytes"]}
        for item in plan["package_files"].values()
        if not item["path"].startswith("provenance/")
    }
    require(
        source_freeze.get("schema") == "quant_risk_source_freeze_v1"
        and source_freeze.get("status") == "READY"
        and source_freeze.get("files") == expected_source,
        "ROOT_SOURCE_FREEZE_CONTENT_DIFFERS",
    )
    require(
        plan.get("data_dataset") == json.loads(freeze_raw)["dataset"]
        and plan.get("source_manifest_sha256") == MANIFEST_SHA,
        "PLAN_DATA_PROOF_DIFFERS",
    )
    for name, digest in plan["scientific_source_sha256"].items():
        require(
            sha(authenticated["src/quantlab/" + name]) == digest,
            "SCIENTIFIC_SOURCE_IDENTITY_MISMATCH",
        )
    require(
        sha(canonical_runner(authenticated["cloud/risk_forecast_runner.py"]))
        == plan["runner_canonical_sha256"],
        "PAYLOAD_KERNEL_TEMPLATE_IDENTITY_DIFFERS",
    )
    return authenticated


def materialize(authenticated, project):
    project = Path(project)
    project.mkdir(parents=True, exist_ok=False)
    for name, raw in authenticated.items():
        target = project.joinpath(*safe_name(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)


def inspect_container(archive, transport):
    """Central-directory checks only; caller must first authenticate whole binary."""
    require(
        transport.get("schema") == "real_market_transport_v1"
        and transport.get("file") == "market-v1.bin"
        and transport.get("sha256") == DATA_SHA
        and transport.get("bytes") == 1417949089
        and transport.get("members") == 8515
        and transport.get("uncompressed_bytes") == 1563719212
        and transport.get("source_manifest_sha256") == MANIFEST_SHA
        and transport.get("snapshot_id") == SNAPSHOT_ID
        and transport.get("archive_sha256") == ARCHIVE_SHA,
        "EXACT_BINARY_TRANSPORT_REQUIRED",
    )
    expected = transport.get("source_files", {})
    members = archive.infolist()
    names = [member.filename for member in members]
    require(
        len(names) == 8515 and len(set(names)) == len(names) and set(names) == set(expected),
        "EXACT_SAFE_ZIP_MEMBER_INVENTORY_REQUIRED",
    )
    require(
        expected.get("manifest.json") == MANIFEST_SHA
        and expected.get("contract.json") == SNAPSHOT_ID,
        "SNAPSHOT_MEMBER_PINS_DIFFER",
    )
    total = 0
    for member in members:
        safe_name(member.filename)
        kind = stat.S_IFMT(member.external_attr >> 16)
        require(
            not member.is_dir()
            and kind in (0, stat.S_IFREG)
            and not member.flag_bits & 1
            and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            and valid_sha(expected[member.filename]),
            "UNSAFE_ZIP_MEMBER",
        )
        total += member.file_size
    require(total == 1563719212, "EXACT_UNCOMPRESSED_SIZE_REQUIRED")
    return members


def extract_data(container, transport_raw, destination, budget_check=lambda: None):
    require(sha(transport_raw) == TRANSPORT_SHA, "TRANSPORT_MANIFEST_HASH_MISMATCH")
    transport = json.loads(transport_raw)
    container = regular(container)
    require(container.stat().st_size == 1417949089, "BINARY_SIZE_MISMATCH")
    with container.open("rb") as stream:
        actual_binary_sha = hash_stream(stream, budget_check)
        if actual_binary_sha != DATA_SHA:
            raise TransportError(
                "WHOLE_BINARY_HASH_MISMATCH",
                {"expected_sha256": DATA_SHA, "actual_sha256": actual_binary_sha},
            )
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
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
                require(
                    digest.hexdigest() == transport["source_files"][member.filename],
                    "ZIP_MEMBER_HASH_MISMATCH",
                )
    return {
        "binary_sha256": actual_binary_sha,
        "binary_bytes": container.stat().st_size,
        "transport_sha256": sha(transport_raw),
        "authenticated_members": len(members),
        "uncompressed_bytes": sum(member.file_size for member in members),
        "snapshot_manifest_sha256": MANIFEST_SHA,
        "snapshot_id": SNAPSHOT_ID,
    }


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


def child(plan_path, project, snapshot_root, etf_root, output):
    raw = read_bound(plan_path, ROOT_PLAN_SHA256)
    plan = json.loads(raw)
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
    require(platform.python_version() == "3.12.15", "EXACT_CHILD_INTERPRETER_REQUIRED")
    require(runtime == plan["runtime"], "LOCKED_NUMERICAL_RUNTIME_MISMATCH")
    # All source bytes and the entire extracted dataset have already been
    # authenticated before these project/native imports.
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.snapshot import MarketSnapshot
    from quantlab.data.yahoo import YahooETFRelease
    from quantlab.risk.study import run_covariance_study

    require(
        Path(sys.modules[run_covariance_study.__module__].__file__).resolve()
        == (Path(project) / "src/quantlab/risk/study.py").resolve(),
        "PROJECT_IMPORT_PATH_MISMATCH",
    )
    snapshot = MarketSnapshot(snapshot_root, MANIFEST_SHA)
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    etf_snapshot = YahooETFRelease(etf_root, ETF_MANIFEST_SHA, schedule)
    result = run_covariance_study(
        snapshot,
        schedule,
        Path(project) / "configs/risk-development.toml",
        Path(output) / "study",
        service_deadline_monotonic=float(os.environ["QUANTLAB_SERVICE_DEADLINE"]) - 60,
        etf_snapshot=etf_snapshot,
    )
    require(result.get("complete") is True, "SCIENTIFIC_RUN_INCOMPLETE")
    check_project(project, plan)
    MarketSnapshot(snapshot_root, MANIFEST_SHA)
    YahooETFRelease(etf_root, ETF_MANIFEST_SHA, schedule)
    runtime_after = {name: version(name) for name in plan["runtime"]}
    write_json(
        Path(output) / "runtime-after.json",
        {"expected": runtime, "actual": runtime_after, "python": platform.python_version()},
    )
    require(runtime_after == runtime, "NUMERICAL_RUNTIME_CHANGED")
    write_json(
        Path(output) / "child-complete.json",
        {
            "complete": True,
            "runtime": runtime,
            "study_manifest_sha256": sha((Path(output) / "study/manifest.json").read_bytes()),
            "source_postflight": "exact",
            "stock_manifest_sha256": MANIFEST_SHA,
            "stock_archive_sha256": ARCHIVE_SHA,
            "etf_release_manifest_sha256": ETF_MANIFEST_SHA,
            "etf_raw_bundle_manifest_sha256": ETF_RAW_BUNDLE_SHA,
            "etf_price_basis": ETF_PRICE_BASIS,
            "etf_archive_sha256": None,
            "end_close": "2014-12-31",
            "stage": "forecast_only_economic_execution_and_bootstrap_not_run",
        },
    )


def run(input_root, output_root):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    started_monotonic = time.monotonic()
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
        matches = [
            p
            for p in Path(input_root).rglob("risk-forecast-plan.json")
            if sha(regular(p).read_bytes()) == ROOT_PLAN_SHA256
        ]
        require(len(matches) == 1, "ONE_EXTERNALLY_BOUND_PLAN_REQUIRED")
        plan_path = matches[0]
        plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
        deadline = started_monotonic + plan["service_budget_seconds"]

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError(
                    "Startup-inclusive service budget exhausted; incomplete output retained"
                )
            return seconds

        remaining()
        authenticated = authenticate_package(plan, plan_path.parent)
        freeze = json.loads(authenticated["provenance/data-freeze.json"])
        transport_paths = [
            p
            for p in Path(input_root).rglob("transport-manifest.json")
            if sha(regular(p).read_bytes()) == TRANSPORT_SHA
        ]
        require(len(transport_paths) == 1, "ONE_BOUND_DATA_TRANSPORT_REQUIRED")
        etf_transport_paths = [
            p
            for p in Path(input_root).rglob("transport-manifest.json")
            if sha(regular(p).read_bytes()) == ETF_TRANSPORT_SHA
        ]
        require(len(etf_transport_paths) == 1, "ONE_BOUND_SEPARATE_ETF_TRANSPORT_REQUIRED")
        etf_mount = etf_transport_paths[0].parent
        data_mount = transport_paths[0].parent
        require(freeze["dataset"]["canonical"] == DATA_ID, "DATASET_IDENTITY_MISMATCH")
        private = Path(tempfile.mkdtemp(prefix="quant-risk-", dir="/tmp"))
        project, snapshot_root = private / "project", private / "market-v1"
        etf_root = private / "yahoo-etf-price-index-v1"
        materialize(authenticated, project)
        state.update(
            phase="data_authentication",
            root_plan_sha256=ROOT_PLAN_SHA256,
            source_sha256=plan["scientific_source_sha256"],
        )
        write_json(output / "status.json", state)
        data_proof = extract_data(
            data_mount / "market-v1.bin",
            read_bound(transport_paths[0], TRANSPORT_SHA),
            snapshot_root,
            budget_check=remaining,
        )
        write_json(output / "data-authentication.json", data_proof)
        etf_proof = extract_etf_data(
            etf_mount / ETF_CONTAINER_NAME,
            read_bound(etf_transport_paths[0], ETF_TRANSPORT_SHA),
            etf_root,
            budget_check=remaining,
        )
        write_json(output / "etf-data-authentication.json", etf_proof)
        state.update(phase="private_runtime_bootstrap", input_sha256=DATA_SHA)
        write_json(output / "status.json", state)
        interpreter, environment, bootstrap_actual = bootstrap_with_receipt(
            private, project, output, deadline
        )
        remaining()
        environment["QUANTLAB_SERVICE_DEADLINE"] = str(deadline)
        state["bootstrap"] = bootstrap_actual
        state.update(phase="development_only")
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
                str(snapshot_root),
                str(etf_root),
                str(output),
            ],
            check=True,
            env=environment,
            cwd=project,
            timeout=remaining(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        remaining()
        read_bound(plan_path, ROOT_PLAN_SHA256)
        authenticate_package(plan, plan_path.parent)
        check_project(project, plan)
        read_bound(transport_paths[0], TRANSPORT_SHA)
        with regular(data_mount / "market-v1.bin").open("rb") as stream:
            require(
                hash_stream(stream) == DATA_SHA,
                "POSTFLIGHT_WHOLE_BINARY_HASH_MISMATCH",
            )
        read_bound(etf_transport_paths[0], ETF_TRANSPORT_SHA)
        with regular(etf_mount / ETF_CONTAINER_NAME).open("rb") as stream:
            require(
                hash_stream(stream, budget_check=remaining) == ETF_DATA_SHA,
                "POSTFLIGHT_ETF_WHOLE_BINARY_HASH_MISMATCH",
            )
        require(
            sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
            "POSTFLIGHT_SELF_MISMATCH",
        )
        remaining()
        child_result = json.loads((output / "child-complete.json").read_bytes())
        require(child_result.get("complete") is True, "CHILD_POSTFLIGHT_INCOMPLETE")
        state.update(
            phase="complete",
            complete=True,
            child=child_result,
            finished_at=datetime.now(UTC).isoformat(),
        )
        write_json(output / "status.json", state)
        # Only forecast outputs are under /kaggle/working. Source, raw
        # normalized inputs, uv caches and the private environment remain in /tmp.
        write_json(
            output / "COMPLETE.json", {"status_sha256": sha((output / "status.json").read_bytes())}
        )
    except BaseException as error:
        state.update(
            phase="failed",
            complete=False,
            error_type=type(error).__name__,
            finished_at=datetime.now(UTC).isoformat(),
        )
        if isinstance(error, TransportError):
            state["guard_reason"] = str(error)
            if error.proof is not None:
                state["guard_proof"] = error.proof
        write_json(output / "status.json", state)
        write_json(output / "FAILURE.json", state)
        raise


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


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--child":
        try:
            child(*sys.argv[2:])
        except BaseException as error:
            write_json(
                Path(sys.argv[6]) / "child-failure.json",
                child_failure_detail(error),
            )
            raise
    else:
        run("/kaggle/input", "/kaggle/working/risk-forecast")
