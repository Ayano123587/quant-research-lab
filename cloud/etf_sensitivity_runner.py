"""Private one-change ETF economics; authenticate all bytes before native imports."""

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
from datetime import date
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
RUNNER_CANONICAL_SHA256 = "UNBOUND"
PLAN_NAME = "etf-sensitivity-plan.json"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_bound(path, digest, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "REGULAR_BOUND_FILE_REQUIRED")
    raw = path.read_bytes()
    require(
        sha(raw) == digest and (size is None or len(raw) == size), "BOUND_BYTE_MISMATCH"
    )
    return raw


def safe_name(name):
    p = PurePosixPath(name)
    require(
        bool(name)
        and not p.is_absolute()
        and ".." not in p.parts
        and "\\" not in name
        and ":" not in name
        and p.as_posix() == name,
        "UNSAFE_RELATIVE_MEMBER",
    )
    return p


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
        require(
            not path.is_symlink() and path.is_file(),
            "INPUT_SYMLINK_OR_NONREGULAR_REFUSED",
        )
        require(
            path.resolve(strict=True).is_relative_to(resolved), "INPUT_ESCAPES_ROOT"
        )
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
        raw, count = re.subn(
            rb"(?m)^" + name.encode() + rb' = "[^"\n]+"$',
            name.encode() + b' = "UNBOUND"',
            raw,
        )
        require(count == 1, "SINGLE_RUNNER_ASSIGNMENT_REQUIRED")
    return raw


def validate_plan(plan):
    require(
        plan.get("schema") == "quant_etf_sensitivity_plan_v1"
        and plan.get("scenarios")
        == [
            "reference_sample_target10",
            "target05_only",
            "target15_only",
            "covariance_ledoit_wolf_only",
            "covariance_ewma63_only",
            "weight_inverse_vol_only",
            "weight_risk_budget_only",
        ]
        and plan.get("fee_bps") == 10
        and plan.get("initial_signal") == "2007-02-28"
        and plan.get("end_close") == "2014-12-31"
        and plan.get("measured_sessions") == 1975
        and plan.get("holdout_outcomes") is False
        and plan.get("python") == "3.12.15"
        and plan.get("threads") == 4
        and 600 <= plan.get("service_budget_seconds", 0) <= 36000,
        "FIXED_SEVEN_ONE_CHANGE_ETF_PLAN_REQUIRED",
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_package(plan, mount):
    validate_plan(plan)
    require(
        {p.name for p in mount.iterdir() if p.is_file()}
        == set(plan["package_files"]) | {PLAN_NAME},
        "EXACT_SOURCE_MOUNT_INVENTORY_REQUIRED",
    )
    raw = {}
    for flat, item in plan["package_files"].items():
        require(safe_name(flat).name == flat, "FLAT_SOURCE_NAMES_REQUIRED")
        safe_name(item["path"])
        require(item["path"] not in raw, "UNIQUE_SOURCE_PATH_REQUIRED")
        raw[item["path"]] = read_bound(mount / flat, item["sha256"], item["bytes"])
    for name, digest in plan["root_freeze_sha256"].items():
        require(sha(raw["provenance/" + name]) == digest, "EXTERNAL_FREEZE_DIFFERS")
    require(
        sha(canonical_runner(raw["cloud/etf_sensitivity_runner.py"]))
        == plan["runner_canonical_sha256"],
        "CANONICAL_RUNNER_DIFFERS",
    )
    return raw


def child(plan_path, project, etf_root, output, private):
    plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
    validate_plan(plan)
    require(
        sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
        "CHILD_SOURCE_DIFFERS",
    )
    for item in plan["package_files"].values():
        read_bound(project / item["path"], item["sha256"], item["bytes"])
    lexical = Path(os.path.abspath(sys.executable))
    require(
        platform.python_version() == "3.12.15"
        and lexical == private / "runtime/.venv/bin/python"
        and lexical.resolve().is_relative_to((private / "python").resolve()),
        "PRIVATE_PYTHON_REQUIRED",
    )
    from importlib.metadata import version

    runtime = {name: version(name) for name in plan["runtime"]}
    require(runtime == plan["runtime"], "LOCKED_RUNTIME_DIFFERS")
    from quantlab.allocation.sensitivity_study import run_sensitivity
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.yahoo import YahooETFRelease

    require(
        Path(sys.modules[run_sensitivity.__module__].__file__).resolve()
        == project / "src/quantlab/allocation/sensitivity_study.py",
        "PROJECT_IMPORT_DIFFERS",
    )
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    snapshot = YahooETFRelease(
        etf_root,
        "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3",
        schedule,
    )
    result = run_sensitivity(
        snapshot,
        schedule,
        project / "configs/etf-sensitivity-development.toml",
        output / "study",
        service_deadline_monotonic=float(os.environ["QUANTLAB_SERVICE_DEADLINE"]) - 90,
    )
    for item in plan["package_files"].values():
        read_bound(project / item["path"], item["sha256"], item["bytes"])
    require(
        {name: version(name) for name in plan["runtime"]} == runtime, "RUNTIME_CHANGED"
    )
    YahooETFRelease(
        etf_root,
        "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3",
        schedule,
    )
    (output / "child-terminal.json").write_bytes(
        encoded(
            {
                "all_seven_paths_complete": result["complete"],
                "status": result["status"],
                "paths_preserved": len(result["paths"]),
                "study_manifest_sha256": file_sha(output / "study/manifest.json"),
                "runtime": runtime,
                "holdout_outcomes": False,
                "winner_selected": False,
            }
        )
    )


def run(input_root, output):
    started = time.monotonic()
    require(
        platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "KAGGLE_ONLY"
    )
    output.mkdir(parents=True, exist_ok=False)
    phase = "startup_plan_discovery"
    (output / "status.json").write_bytes(encoded({"complete": False, "phase": phase}))
    try:
        require(
            ROOT_PLAN_SHA256 != "UNBOUND" and RUNNER_CANONICAL_SHA256 != "UNBOUND",
            "ROOT_FREEZE_REQUIRED",
        )
        plan_path = discover_bound(input_root, PLAN_NAME, ROOT_PLAN_SHA256)
        plan = json.loads(read_bound(plan_path, ROOT_PLAN_SHA256))
        require(
            sha(canonical_runner(Path(__file__).read_bytes()))
            == RUNNER_CANONICAL_SHA256,
            "KERNEL_RUNNER_DIFFERS",
        )
        raw = authenticate_package(plan, plan_path.parent)
        deadline = started + plan["service_budget_seconds"]

        def budget():
            if time.monotonic() >= deadline:
                raise TimeoutError("Startup-inclusive ETF sensitivity budget exhausted")

        budget()
        phase = "authenticated_data_extraction"
        with tempfile.TemporaryDirectory(
            prefix="quant-etf-sensitivity-", dir="/tmp"
        ) as temporary:
            private = Path(temporary)
            project = private / "project"
            for name, data in raw.items():
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            market_helper = load_module(
                project / "cloud/risk_forecast_runner.py", "bound_runtime_helper"
            )
            etf_helper = load_module(
                project / "cloud/etf_development_runner.py", "bound_etf_helper"
            )
            etf_freeze = json.loads(raw["provenance/data-freeze.json"])
            etf_helper.validate_data_freeze(etf_freeze)
            etf_bin = discover_bound(
                input_root,
                etf_helper.CONTAINER_NAME,
                etf_helper.DATA_SHA,
                etf_freeze["container"]["bytes"],
            )
            transport = discover_bound(
                input_root, "transport-manifest.json", etf_helper.TRANSPORT_SHA
            )
            etf_root = private / "etf"
            etf_helper.extract_data(etf_bin, transport.read_bytes(), etf_root, budget)
            phase = "locked_private_runtime"
            interpreter, environment, receipt = market_helper.bootstrap(
                private, project, deadline
            )
            environment["QUANTLAB_SERVICE_DEADLINE"] = str(deadline)
            (output / "runtime-bootstrap.json").write_bytes(encoded(receipt))
            phase = "native_economic_child"
            log = private / "child.log"
            try:
                with log.open("xb") as stream:
                    subprocess.run(
                        [
                            str(interpreter),
                            str(Path(__file__).resolve()),
                            "--child",
                            str(plan_path),
                            str(project),
                            str(etf_root),
                            str(output),
                            str(private),
                        ],
                        env=environment,
                        check=True,
                        timeout=market_helper.remaining_budget(deadline),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
            finally:
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 16384))
                        tail = stream.read().decode(errors="replace")
                    (output / "child-diagnostics.json").write_bytes(
                        encoded(
                            {
                                "tail": tail,
                                "native_log_bytes": log.stat().st_size,
                                "tail_limit": 16384,
                            }
                        )
                    )
            phase = "all_input_source_postflight"
            for item in plan["package_files"].values():
                read_bound(project / item["path"], item["sha256"], item["bytes"])
            require(
                file_sha(etf_bin) == etf_helper.DATA_SHA, "INPUT_POSTFLIGHT_DIFFERS"
            )
            budget()
            (output / "transport-terminal.json").write_bytes(
                encoded(
                    {
                        "transport_postflight": "PASS",
                        "child_terminal_sha256": file_sha(
                            output / "child-terminal.json"
                        ),
                        "all_seven_attempted_distinct_from_complete": True,
                        "native_caches_not_published": True,
                    }
                )
            )
    except BaseException as error:
        (output / "FAILURE.json").write_bytes(
            encoded(
                {
                    "complete": False,
                    "error_type": type(error).__name__,
                    "phase": phase,
                    "actual_prefix_outputs_preserved": True,
                }
            )
        )
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*(Path(value) for value in sys.argv[2:]))
    else:
        run(Path("/kaggle/input"), Path("/kaggle/working/etf-sensitivity-development"))
