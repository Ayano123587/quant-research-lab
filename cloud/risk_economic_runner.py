"""Prospective private risk economics transport; stdlib-only until all inputs authenticate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import date, datetime
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
RUNNER_CANONICAL_SHA256 = "UNBOUND"
PLAN_NAME = "risk-economic-plan.json"
FORECAST_SHA = "ea7a801782345b38d2f3e0a1be038c5212b00263360eb78c1b910bfeba65fc8a"
PROFILE = ["etf8_T63", "stock100_T252"]
FORECAST_BINARY_SHA = "ffd08ee7537f10d5cef33374a6134b3ea9efde33395ed421fadd5aa8bfa04d11"
FORECAST_BINARY_BYTES = 656019955
FORECAST_TRANSPORT_SHA = "4d726d9c99ccba793f6a05233c0fe4fd061352a30eea5aa64f85bfe5b969fa92"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    require(
        not p.is_absolute()
        and ".." not in p.parts
        and "\\" not in name
        and ":" not in name
        and p.as_posix() == name
        and bool(name),
        "UNSAFE_RELATIVE_PATH",
    )
    return p


def read_bound(path, digest, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "REGULAR_BOUND_FILE_REQUIRED")
    raw = path.read_bytes()
    require(sha(raw) == digest and (size is None or len(raw) == size), "FILE_SHA_OR_SIZE_DIFFERS")
    return raw


def discover_bound(input_root, name, digest, size=None):
    """Find one authenticated regular input at any real provider mount depth."""
    root = Path(input_root)
    require(root.is_dir() and not root.is_symlink(), "REGULAR_INPUT_ROOT_REQUIRED")
    resolved = root.resolve(strict=True)
    require(
        safe_name(name).name == name and re.fullmatch(r"[0-9a-f]{64}", digest),
        "BOUND_BASENAME_AND_SHA_REQUIRED",
    )
    matches = []
    for path in root.rglob(name):
        require(not path.is_symlink() and path.is_file(), "INPUT_SYMLINK_OR_NONREGULAR_REFUSED")
        require(path.resolve(strict=True).is_relative_to(resolved), "INPUT_ESCAPES_ROOT")
        parent = path.parent
        while parent != root:
            require(not parent.is_symlink(), "INPUT_SYMLINK_ANCESTOR_REFUSED")
            parent = parent.parent
        if size is not None and path.stat().st_size != size:
            continue
        if file_sha(path) == digest:
            matches.append(path)
    require(len(matches) == 1, "ONE_EXACT_RECURSIVE_BOUND_INPUT_REQUIRED: " + name)
    return matches[0]


def canonical_runner(raw):
    raw = raw.replace(b"\r\n", b"\n")
    for name in ("ROOT_PLAN_SHA256", "RUNNER_CANONICAL_SHA256"):
        raw, count = re.subn(name.encode() + rb' = "[^"\n]+"', name.encode() + b' = "UNBOUND"', raw)
        require(count == 1, "UNAMBIGUOUS_RUNNER_BINDING_REQUIRED")
    return raw


def validate_forecast_freeze(value):
    require(
        value.get("schema") == "quant_risk_integration_forecast_data_freeze_v1"
        and value.get("status") == "READY"
        and value.get("private") is True,
        "ACTUAL_ROOT_READY_FORECAST_DATA_FREEZE_REQUIRED",
    )
    require(
        isinstance(value.get("dataset_id"), int)
        and not isinstance(value["dataset_id"], bool)
        and value["dataset_id"] > 0
        and value.get("version") == 1
        and re.fullmatch(r"aikyatansinha/[a-z0-9-]+", value.get("dataset", "")),
        "ACTUAL_PRIVATE_DATASET_IDENTITY_REQUIRED",
    )
    require(
        value.get("upstream_manifest_sha256") == FORECAST_SHA
        and value.get("profile") == PROFILE
        and value.get("forecast_files") == 1520
        and value.get("upstream_completed_cells") == 9120,
        "FIXED_ACTUAL_FORECAST_PROFILE_REQUIRED",
    )
    for name in ("transport_manifest_sha256", "remote_verification_receipt_sha256"):
        require(
            re.fullmatch(r"[0-9a-f]{64}", value.get(name, "")), "EXTERNAL_REMOTE_PROOF_REQUIRED"
        )
    item = value["container"]
    require(
        item["name"] == "risk-integration-forecast-v1.bin"
        and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        and isinstance(item["bytes"], int)
        and item["bytes"] > 0,
        "ACTUAL_BINARY_CONTAINER_PIN_REQUIRED",
    )
    require(
        item["sha256"] == FORECAST_BINARY_SHA
        and item["bytes"] == FORECAST_BINARY_BYTES
        and value["transport_manifest_sha256"] == FORECAST_TRANSPORT_SHA,
        "ACTUAL_ROOT_PACKAGED_FORECAST_PINS_REQUIRED",
    )


def validate_plan(plan):
    require(
        plan.get("schema") == "quant_risk_economic_plan_v1"
        and plan.get("attempted_paths") == 208
        and plan.get("fees_bps") == [0, 5, 10, 25]
        and plan.get("start_signal") == "2007-02-28"
        and plan.get("end_close") == "2014-12-31"
        and plan.get("holdout_outcomes") is False
        and plan.get("python") == "3.12.15"
        and plan.get("threads") == 4
        and plan.get("resume") == "none_this_transport_preserves_complete_path_checkpoints"
        and 600 <= plan.get("service_budget_seconds", 0) <= 36000,
        "REGISTERED_ECONOMIC_TRANSPORT_CONTRACT_REQUIRED",
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_package(plan, mount):
    validate_plan(plan)
    files = plan["package_files"]
    require(
        {p.name for p in mount.iterdir() if p.is_file()} == set(files) | {PLAN_NAME},
        "EXACT_FLAT_SOURCE_PACKAGE_REQUIRED",
    )
    raw = {}
    for flat, item in files.items():
        require(safe_name(flat).name == flat, "FLAT_PACKAGE_REQUIRED")
        safe_name(item["path"])
        raw[item["path"]] = read_bound(mount / flat, item["sha256"], item["bytes"])
    for name, expected in plan["root_freeze_sha256"].items():
        require(sha(raw["provenance/" + name]) == expected, "EXTERNAL_ROOT_FREEZE_DIFFERS")
    require(
        sha(canonical_runner(raw["cloud/risk_economic_runner.py"]))
        == plan["runner_canonical_sha256"],
        "PAYLOAD_RUNNER_CANONICAL_DIFFERS",
    )
    return raw


def inspect_forecast_container(container, transport_raw):
    """Actual local byte proof is independent from later remote publication approval."""
    require(
        file_sha(container) == FORECAST_BINARY_SHA
        and container.stat().st_size == FORECAST_BINARY_BYTES,
        "WHOLE_FORECAST_BINARY_DIFFERS",
    )
    require(sha(transport_raw) == FORECAST_TRANSPORT_SHA, "FORECAST_TRANSPORT_SHA_DIFFERS")
    transport = json.loads(transport_raw)
    require(
        transport["complete"] is True
        and transport["profile"] == PROFILE
        and transport["upstream_manifest_sha256"] == FORECAST_SHA
        and transport["copied_forecast_cells"] == 1520
        and transport["upstream_completed_cells"] == 9120,
        "FORECAST_PROFILE_PROOF_DIFFERS",
    )
    inventory = {item["path"]: item for item in transport["members"]}
    require(
        len(inventory) == len(transport["members"]) == 1526, "EXACT_FORECAST_MEMBER_COUNT_REQUIRED"
    )
    return inventory


def extract_forecasts(container, transport_raw, destination, freeze, budget):
    validate_forecast_freeze(freeze)
    inventory = inspect_forecast_container(container, transport_raw)
    with zipfile.ZipFile(container) as archive:
        require(
            set(archive.namelist()) == set(inventory) and len(archive.infolist()) == 1526,
            "EXACT_FORECAST_ZIP_INVENTORY_REQUIRED",
        )
        for member in archive.infolist():
            budget()
            safe_name(member.filename)
            require(
                member.compress_type == zipfile.ZIP_STORED
                and not member.is_dir()
                and member.file_size == inventory[member.filename]["bytes"],
                "STORED_MEMBER_SIZE_REQUIRED",
            )
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with archive.open(member) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    budget()
                    h.update(block)
                    output.write(block)
            require(
                h.hexdigest() == inventory[member.filename]["sha256"], "FORECAST_MEMBER_SHA_DIFFERS"
            )
    require(
        file_sha(destination / "manifest.json") == FORECAST_SHA
        and (destination / "COMPLETE").read_bytes() == (FORECAST_SHA + "\n").encode(),
        "ORIGINAL_COMPLETE_9120_MANIFEST_REQUIRED",
    )


def run_child_with_diagnostics(command, environment, private, output, timeout):
    """Retain a bounded diagnostic tail; native logs stay in the private temp tree."""
    log = private / "native-child.log"
    try:
        with log.open("xb") as stream:
            subprocess.run(
                command,
                env=environment,
                check=True,
                timeout=timeout,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    finally:
        if log.is_file():
            size = log.stat().st_size
            with log.open("rb") as stream:
                stream.seek(max(0, size - 16384))
                tail = stream.read().decode("utf-8", errors="replace")
            (output / "child-diagnostics.json").write_bytes(
                encoded(
                    {
                        "native_log_bytes": size,
                        "tail_bytes_limit": 16384,
                        "truncated": size > 16384,
                        "tail": tail,
                    }
                )
            )


def child(plan_path, project, stock_root, etf_root, forecast_root, output, private):
    plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
    validate_plan(plan)
    require(
        sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
        "CHILD_CANONICAL_SELF_DIFFERS",
    )
    for item in plan["package_files"].values():
        read_bound(project / item["path"], item["sha256"], item["bytes"])
    require(platform.python_version() == "3.12.15", "EXACT_PRIVATE_PYTHON_REQUIRED")
    lexical = Path(os.path.abspath(sys.executable))
    require(
        lexical == private / "runtime/.venv/bin/python"
        and lexical.resolve().is_relative_to((private / "python").resolve()),
        "EXACT_PRIVATE_VENV_AND_MANAGED_TARGET_REQUIRED",
    )
    from importlib.metadata import version

    runtime = {name: version(name) for name in plan["runtime"]}
    require(runtime == plan["runtime"], "LOCKED_OPTIMIZATION_RUNTIME_DIFFERS")
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.snapshot import MarketSnapshot
    from quantlab.data.yahoo import YahooETFRelease
    from quantlab.risk.economic_study import run_economic_study
    from quantlab.risk.optimization import SolverPolicy

    legacy = load_module(
        project / "cloud/risk_forecast_runner.py", "authenticated_legacy_transport"
    )
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    snapshot = MarketSnapshot(stock_root, legacy.MANIFEST_SHA)
    etf = YahooETFRelease(etf_root, legacy.ETF_MANIFEST_SHA, schedule)
    frozen = plan["solver_policy"]
    policy = SolverPolicy(
        tuple(map(tuple, frozen["expected_versions"])), datetime.fromisoformat(frozen["frozen_at"])
    )
    require(policy.policy_id == frozen["policy_id"], "EXTERNAL_FROZEN_SOLVER_POLICY_DIFFERS")
    require(
        Path(sys.modules[run_economic_study.__module__].__file__).resolve()
        == (project / "src/quantlab/risk/economic_study.py").resolve(),
        "PROJECT_IMPORT_PATH_DIFFERS",
    )
    result = run_economic_study(
        snapshot,
        etf,
        schedule,
        forecast_root,
        FORECAST_SHA,
        project / "configs/risk-development.toml",
        project / "configs/risk-economic-development.toml",
        plan["config_sha256"],
        output / "study",
        frozen_solver_policy=policy,
        expected_solver_policy_id=policy.policy_id,
        projection_path=project / "configs/execution-projection.json",
        expected_projection_sha256=plan["projection_sha256"],
        service_deadline_monotonic=float(os.environ["QUANTLAB_SERVICE_DEADLINE"]) - 90,
    )
    for item in plan["package_files"].values():
        read_bound(project / item["path"], item["sha256"], item["bytes"])
    require({name: version(name) for name in plan["runtime"]} == runtime, "RUNTIME_CHANGED")
    MarketSnapshot(stock_root, legacy.MANIFEST_SHA)
    YahooETFRelease(etf_root, legacy.ETF_MANIFEST_SHA, schedule)
    (output / "child-terminal.json").write_bytes(
        encoded(
            {
                "all_economic_paths_complete": result["complete"],
                "attempted_terminal_paths": result["attempted_terminal_paths"],
                "unsupported_paths": result["unsupported_paths"],
                "study_manifest_sha256": file_sha(output / "study/manifest.json"),
                "runtime": runtime,
                "forecast_profile": PROFILE,
                "economic_complete_distinct_from_208_attempted": True,
            }
        )
    )


def run(input_root, output):
    started = time.monotonic()
    require(platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "KAGGLE_ONLY")
    output.mkdir(parents=True, exist_ok=False)
    phase = "startup_plan_discovery"
    (output / "status.json").write_bytes(encoded({"complete": False, "phase": phase}))
    try:
        require(
            ROOT_PLAN_SHA256 != "UNBOUND" and RUNNER_CANONICAL_SHA256 != "UNBOUND",
            "ROOT_APPROVAL_REQUIRED",
        )
        plan_path = discover_bound(input_root, PLAN_NAME, ROOT_PLAN_SHA256)
        plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
        require(
            sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
            "BOUND_KERNEL_SELF_DIFFERS",
        )
        raw = authenticate_package(plan, plan_path.parent)
        deadline = started + plan["service_budget_seconds"]

        def budget():
            if time.monotonic() >= deadline:
                raise TimeoutError("Startup-inclusive service deadline exhausted")

        budget()
        phase = "actual_input_authentication"
        with tempfile.TemporaryDirectory(prefix="quant-risk-economic-", dir="/tmp") as temporary:
            private = Path(temporary)
            project = private / "project"
            for name, data in raw.items():
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            legacy = load_module(
                project / "cloud/risk_forecast_runner.py", "authenticated_extractors"
            )
            stock_freeze = json.loads(raw["provenance/data-freeze.json"])
            etf_freeze = json.loads(raw["provenance/etf-data-freeze.json"])
            forecast_freeze = json.loads(raw["provenance/forecast-data-freeze.json"])
            legacy.validate_data_freeze(stock_freeze)
            legacy.validate_etf_data_freeze(etf_freeze)
            validate_forecast_freeze(forecast_freeze)

            stock_bin = discover_bound(
                input_root,
                "market-v1.bin",
                legacy.DATA_SHA,
                stock_freeze["remote_files"]["market-v1.bin"]["bytes"],
            )
            stock_transport = discover_bound(
                input_root, "transport-manifest.json", legacy.TRANSPORT_SHA
            )
            etf_bin = discover_bound(
                input_root,
                legacy.ETF_CONTAINER_NAME,
                legacy.ETF_DATA_SHA,
                etf_freeze["container"]["bytes"],
            )
            etf_transport = discover_bound(
                input_root, "transport-manifest.json", legacy.ETF_TRANSPORT_SHA
            )
            forecast_bin = discover_bound(
                input_root,
                forecast_freeze["container"]["name"],
                forecast_freeze["container"]["sha256"],
                forecast_freeze["container"]["bytes"],
            )
            forecast_transport = discover_bound(
                input_root, "risk-integration-transport.json", FORECAST_TRANSPORT_SHA
            )
            stock_root, etf_root, forecast_root = [
                private / name for name in ("market", "etf", "forecasts")
            ]
            legacy.extract_data(stock_bin, stock_transport.read_bytes(), stock_root, budget)
            legacy.extract_etf_data(etf_bin, etf_transport.read_bytes(), etf_root, budget)
            extract_forecasts(
                forecast_bin,
                forecast_transport.read_bytes(),
                forecast_root,
                forecast_freeze,
                budget,
            )
            phase = "private_locked_runtime_bootstrap"
            interpreter, environment, runtime_receipt = legacy.bootstrap(private, project, deadline)
            uv = private / "uv-package/bin/uv"
            subprocess.run(
                [
                    str(uv),
                    "sync",
                    "--locked",
                    "--no-install-project",
                    "--no-default-groups",
                    "--group",
                    "optimization",
                    "--project",
                    str(project),
                    "--python",
                    str(interpreter),
                ],
                env=environment,
                check=True,
                timeout=legacy.remaining_budget(deadline),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            environment["QUANTLAB_SERVICE_DEADLINE"] = str(deadline)
            (output / "runtime-bootstrap.json").write_bytes(encoded(runtime_receipt))
            phase = "native_economic_child"
            run_child_with_diagnostics(
                [
                    str(interpreter),
                    str(Path(__file__).resolve()),
                    "--child",
                    str(plan_path),
                    str(project),
                    str(stock_root),
                    str(etf_root),
                    str(forecast_root),
                    str(output),
                    str(private),
                ],
                environment=environment,
                private=private,
                output=output,
                timeout=legacy.remaining_budget(deadline),
            )
            budget()
            phase = "source_runtime_input_postflight"
            for item in plan["package_files"].values():
                read_bound(project / item["path"], item["sha256"], item["bytes"])
            require(
                file_sha(stock_bin) == legacy.DATA_SHA
                and file_sha(etf_bin) == legacy.ETF_DATA_SHA
                and file_sha(forecast_bin) == forecast_freeze["container"]["sha256"],
                "INPUT_BINARY_POSTFLIGHT_DIFFERS",
            )
            budget()
            (output / "transport-terminal.json").write_bytes(
                encoded(
                    {
                        "transport_postflight": "PASS",
                        "all_208_attempted_distinct_from_complete": True,
                        "child_terminal_sha256": file_sha(output / "child-terminal.json"),
                        "no_native_cache_or_raw_inputs_under_working": True,
                    }
                )
            )
    except BaseException as error:
        (output / "FAILURE.json").write_bytes(
            encoded(
                {
                    "complete": False,
                    "error_type": type(error).__name__,
                    "reason": str(error)[:2000],
                    "phase": phase,
                    "partial_paths_preserved": True,
                }
            )
        )
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*(Path(value) for value in sys.argv[2:]))
    else:
        run(Path("/kaggle/input"), Path("/kaggle/working/risk-economic-development"))
