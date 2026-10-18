"""Private saved-ranking economics; authenticate all bytes before native imports."""

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
from datetime import date
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
RUNNER_CANONICAL_SHA256 = "UNBOUND"
PLAN_NAME = "stock-economic-plan.json"
PREDICTION_MANIFEST = "b46c22e5a6f46330668f6f3870c0c8cc2ddde15c283d3f9af8637f93557492d6"
PREDICTION_BINARY = "78b007543b5c4fe2a576683bff43a69c6f3540a4164ec7ac8126724485116f01"
PREDICTION_TRANSPORT = (
    "c3201833666bfde2f8d017f5cad62f9b2e2a683da234175817be841dbdd3d424"
)


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


def validate_prediction_freeze(value):
    require(
        value.get("schema") == "quant_stock_economic_prediction_data_freeze_v1"
        and value.get("status") == "READY"
        and value.get("private") is True
        and value.get("dataset")
        == "aikyatansinha/portfolio-quant-stock-economic-predictions-v1"
        and value.get("dataset_id") == 12660883
        and value.get("version") == 1,
        "ACTUAL_PRIVATE_READY_PREDICTION_DATA_REQUIRED",
    )
    require(
        value.get("upstream_manifest_sha256") == PREDICTION_MANIFEST
        and value.get("upstream_kernel_id") == 139242106
        and value.get("upstream_version") == 1
        and value.get("prediction_months") == 96
        and value.get("candidate_fit_receipts") == 1536
        and value.get("native_models_packaged") is False
        and value.get("outcome_values_read") is False
        and value.get("container")
        == {
            "name": "stock-economic-predictions-v1.bin",
            "bytes": 21665317,
            "sha256": PREDICTION_BINARY,
        }
        and value.get("transport_manifest_sha256") == PREDICTION_TRANSPORT
        and re.fullmatch(
            r"[0-9a-f]{64}", value.get("remote_verification_receipt_sha256", "")
        ),
        "ACTUAL_COMPLETE_96_MONTH_PREDICTION_PROFILE_REQUIRED",
    )


def validate_plan(plan):
    require(
        plan.get("schema") == "quant_stock_economic_plan_v1"
        and plan.get("paths") == 64
        and plan.get("fees_bps") == [0, 5, 10, 25]
        and plan.get("initial_signal") == "2007-02-28"
        and plan.get("end_close") == "2014-12-31"
        and plan.get("measured_sessions") == 1975
        and plan.get("signal_months_audited") == 96
        and plan.get("economic_signal_months") == 95
        and plan.get("holdout_outcomes") is False
        and plan.get("python") == "3.12.15"
        and plan.get("threads") == 4
        and 600 <= plan.get("service_budget_seconds", 0) <= 36000,
        "FIXED_STOCK_ECONOMIC_PLAN_REQUIRED",
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
        sha(canonical_runner(raw["cloud/stock_economic_runner.py"]))
        == plan["runner_canonical_sha256"],
        "CANONICAL_RUNNER_DIFFERS",
    )
    return raw


def extract_predictions(container, transport_raw, destination, freeze, budget):
    validate_prediction_freeze(freeze)
    require(
        file_sha(container) == PREDICTION_BINARY
        and container.stat().st_size == 21665317
        and sha(transport_raw) == PREDICTION_TRANSPORT,
        "WHOLE_PREDICTION_INPUT_DIFFERS",
    )
    transport = json.loads(transport_raw)
    members = transport["members"]
    require(
        transport["complete"] is True
        and transport["upstream_manifest_sha256"] == PREDICTION_MANIFEST
        and transport["prediction_months"] == 96
        and transport["candidate_fit_receipts"] == 1536
        and transport["native_models_packaged"] is False
        and transport["outcome_values_read"] is False
        and len(members) == 99
        and set(members)
        == {"manifest.json", "COMPLETE", "fit-index.json"}
        | {
            name
            for name in members
            if re.fullmatch(r"predictions/20(?:0[7-9]|1[0-4])-\d\d-\d\d.parquet", name)
        },
        "EXACT_REAL_PREDICTION_MEMBER_PROFILE_REQUIRED",
    )
    with zipfile.ZipFile(container) as archive:
        require(
            len(archive.infolist()) == 99 and set(archive.namelist()) == set(members),
            "EXACT_PREDICTION_ZIP_INVENTORY_REQUIRED",
        )
        for member in archive.infolist():
            budget()
            safe_name(member.filename)
            require(
                not member.is_dir()
                and member.compress_type == zipfile.ZIP_STORED
                and member.file_size == members[member.filename]["bytes"],
                "STORED_MEMBER_REQUIRED",
            )
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(member) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    budget()
                    output.write(block)
                    digest.update(block)
            require(
                digest.hexdigest() == members[member.filename]["sha256"],
                "PREDICTION_MEMBER_DIFFERS",
            )
    require(
        file_sha(destination / "manifest.json") == PREDICTION_MANIFEST
        and (destination / "COMPLETE").read_bytes()
        == (PREDICTION_MANIFEST + "\n").encode(),
        "ORIGINAL_COMPLETE_STUDY_REQUIRED",
    )


def child(plan_path, project, market, prepared, predictions, output, private):
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
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.snapshot import MarketSnapshot
    from quantlab.stocks.economic_study import run_economic_study

    require(
        Path(sys.modules[run_economic_study.__module__].__file__).resolve()
        == project / "src/quantlab/stocks/economic_study.py",
        "PROJECT_IMPORT_DIFFERS",
    )
    snapshot = MarketSnapshot(
        market, "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
    )
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    result = run_economic_study(
        predictions,
        snapshot,
        prepared,
        schedule,
        project / "configs/stock-economic-development.toml",
        output / "study",
        service_deadline_monotonic=float(os.environ["QUANTLAB_SERVICE_DEADLINE"]) - 90,
    )
    for item in plan["package_files"].values():
        read_bound(project / item["path"], item["sha256"], item["bytes"])
    require(
        {name: version(name) for name in plan["runtime"]} == runtime, "RUNTIME_CHANGED"
    )
    (output / "child-terminal.json").write_bytes(
        encoded(
            {
                "all64_paths_complete": result["complete"],
                "status": result["status"],
                "paths_preserved": len(result["paths"]),
                "study_manifest_sha256": file_sha(output / "study/manifest.json"),
                "runtime": runtime,
                "no_holdout_or_model_fit": True,
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
                raise TimeoutError("Startup-inclusive stock economics budget exhausted")

        budget()
        phase = "authenticated_data_extraction"
        with tempfile.TemporaryDirectory(
            prefix="quant-stock-economic-", dir="/tmp"
        ) as temporary:
            private = Path(temporary)
            project = private / "project"
            for name, data in raw.items():
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            market_helper = load_module(
                project / "cloud/risk_forecast_runner.py", "bound_market_helper"
            )
            prepared_helper = load_module(
                project / "cloud/stock_development_runner.py", "bound_prepared_helper"
            )
            market_freeze = json.loads(raw["provenance/data-freeze.json"])
            prepared_freeze = json.loads(raw["provenance/prepared-data-freeze.json"])
            prediction_freeze = json.loads(
                raw["provenance/prediction-data-freeze.json"]
            )
            market_helper.validate_data_freeze(market_freeze)
            prepared_helper.validate_data_freeze(prepared_freeze)
            validate_prediction_freeze(prediction_freeze)

            market_bin = discover_bound(
                input_root,
                "market-v1.bin",
                market_helper.DATA_SHA,
                market_freeze["remote_files"]["market-v1.bin"]["bytes"],
            )
            prepared_bin = discover_bound(
                input_root,
                "stock-panel-v1.bin",
                prepared_helper.DATA_SHA,
                prepared_helper.CONTAINER_BYTES,
            )
            prediction_bin = discover_bound(
                input_root,
                "stock-economic-predictions-v1.bin",
                PREDICTION_BINARY,
                21665317,
            )
            market_transport = discover_bound(
                input_root, "transport-manifest.json", market_helper.TRANSPORT_SHA
            )
            prepared_transport = discover_bound(
                input_root, "transport-manifest.json", prepared_helper.TRANSPORT_SHA
            )
            prediction_transport = discover_bound(
                input_root, "stock-prediction-transport.json", PREDICTION_TRANSPORT
            )
            market, prepared, predictions = [
                private / name for name in ("market", "prepared", "predictions")
            ]
            market_helper.extract_data(
                market_bin,
                market_transport.read_bytes(),
                market,
                budget,
            )
            prepared_helper.extract_data(
                prepared_bin,
                prepared_transport.read_bytes(),
                prepared,
                budget,
            )
            extract_predictions(
                prediction_bin,
                prediction_transport.read_bytes(),
                predictions,
                prediction_freeze,
                budget,
            )
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
                            str(market),
                            str(prepared),
                            str(predictions),
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
                file_sha(market_bin) == market_helper.DATA_SHA
                and file_sha(prepared_bin) == prepared_helper.DATA_SHA
                and file_sha(prediction_bin) == PREDICTION_BINARY,
                "INPUT_POSTFLIGHT_DIFFERS",
            )
            budget()
            (output / "transport-terminal.json").write_bytes(
                encoded(
                    {
                        "transport_postflight": "PASS",
                        "child_terminal_sha256": file_sha(
                            output / "child-terminal.json"
                        ),
                        "all64_attempted_distinct_from_complete": True,
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
        run(Path("/kaggle/input"), Path("/kaggle/working/stock-economic-development"))
