"""Bound source/bootstrap transport for one pre-2015 native adapter check."""

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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
RUNNER_CANONICAL_SHA256 = "UNBOUND"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe_name(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and not {"", ".", ".."} & set(path.parts)
        and "\\" not in name
        and ":" not in name,
        "SAFE_RELATIVE_PATH_REQUIRED",
    )
    return path


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
        raw, count = re.subn(
            rb"^" + name.encode() + rb' = "[^"\n]+"$',
            name.encode() + b' = "UNBOUND"',
            raw,
            flags=re.MULTILINE,
        )
        require(count == 1, "UNAMBIGUOUS_ACTUAL_ASSIGNMENT_REQUIRED")
    return raw


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def authenticate_source(plan, mount):
    require(plan["schema"] == "pre2015_stock_adapter_cloud_transport_v1", "SMOKE_PLAN_REQUIRED")
    require(
        plan["prediction_sessions"] == ["2010-06-30"]
        and plan["candidate_ids"] == ["ridge_alpha_1", "lgbm_regression_l2_leaves7_iterations100"]
        and plan["holdout_outcomes"] is False,
        "FIXED_ENGINEERING_SCOPE_REQUIRED",
    )
    actual = {p.name for p in mount.iterdir() if p.is_file()}
    expected = set(plan["package_files"]) | {"holdout-smoke-plan.json"}
    require(
        actual == expected and all(not p.is_symlink() and p.is_file() for p in mount.iterdir()),
        "EXACT_REGULAR_SOURCE_INVENTORY_REQUIRED",
    )
    paths, authenticated = set(), {}
    for flat, row in plan["package_files"].items():
        require(safe_name(flat).name == flat, "FLAT_SOURCE_NAME_REQUIRED")
        relative = safe_name(row["path"]).as_posix()
        require(relative not in paths, "DUPLICATE_SOURCE_PATH")
        paths.add(relative)
        path = mount / flat
        require(
            path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
            "RAW_SOURCE_BYTE_AUTHENTICATION_FAILED",
        )
        authenticated[relative] = path.read_bytes()
    require(
        hashlib.sha256(authenticated["cloud/holdout_training_smoke_runner.py"]).hexdigest()
        == plan["runner_raw_sha256"],
        "RAW_RUNNER_TEMPLATE_DIFFERS",
    )
    return authenticated


def authenticate_project(project, plan):
    for row in plan["package_files"].values():
        path = project.joinpath(*safe_name(row["path"]).parts)
        require(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == row["bytes"]
            and file_sha(path) == row["sha256"],
            "MATERIALIZED_SOURCE_DRIFT",
        )


def child(plan_path, expected_sha, project, prepared, output):
    require(
        platform.system() == "Linux" and str(Path.cwd()).startswith("/kaggle/working"),
        "KAGGLE_ONLY_ENGINEERING_TRANSPORT",
    )
    require(file_sha(plan_path) == expected_sha, "CHILD_EXTERNAL_PLAN_DIFFERS")
    plan = json.loads(Path(plan_path).read_bytes())
    authenticate_source(plan, Path(plan_path).parent)
    authenticate_project(Path(project), plan)
    from quantlab.evaluation.holdout_training_smoke import run_smoke

    result = run_smoke(
        Path(project) / "engineering-plan.json",
        plan["engineering_plan_sha256"],
        project=project,
        inputs=prepared,
        output=Path(output) / "engineering",
    )
    authenticate_project(Path(project), plan)
    require(result["complete"] is True and result["actual_fits"] == 2, "TWO_ACTUAL_FITS_REQUIRED")


def run(input_root, output):
    require(
        platform.system() == "Linux" and str(Path.cwd()).startswith("/kaggle/working"),
        "KAGGLE_ONLY_ENGINEERING_TRANSPORT",
    )
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "startup.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "phase": "source_authentication",
            "complete": False,
        },
    )
    try:
        require(
            hashlib.sha256(canonical_runner(Path(__file__).read_bytes())).hexdigest()
            == RUNNER_CANONICAL_SHA256,
            "CANONICAL_RUNNER_DIFFERS",
        )
        plan_path = discover_bound(input_root, "holdout-smoke-plan.json", ROOT_PLAN_SHA256)
        plan = json.loads(plan_path.read_bytes())
        authenticated = authenticate_source(plan, plan_path.parent)
        deadline = started + plan["service_budget_seconds"]
        private = Path(tempfile.mkdtemp(prefix="quant-adapter-smoke-", dir="/tmp"))
        project, prepared = private / "project", private / "prepared"
        project.mkdir()
        for name, raw in authenticated.items():
            path = project.joinpath(*safe_name(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        authenticate_project(project, plan)
        helper_path = project / "cloud/stock_development_runner.py"
        spec = importlib.util.spec_from_file_location("authenticated_stock_transport", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        freeze = json.loads(authenticated["prepared-data-freeze.json"])
        helper.validate_data_freeze(freeze)
        transport = discover_bound(
            input_root, "transport-manifest.json", freeze["transport_manifest_sha256"]
        )
        container = discover_bound(
            input_root,
            freeze["container"]["name"],
            freeze["container"]["sha256"],
            freeze["container"]["bytes"],
        )
        proof = helper.extract_data(
            container,
            transport.read_bytes(),
            prepared,
            budget_check=lambda: helper.remaining_budget(deadline),
        )
        write_json(output / "prepared-authentication.json", proof)
        interpreter, environment, bootstrap = helper.bootstrap_with_receipt(
            private, project, output, deadline
        )
        subprocess.run(
            [
                str(interpreter),
                str(project / "cloud/holdout_training_smoke_runner.py"),
                "--child",
                str(plan_path),
                ROOT_PLAN_SHA256,
                str(project),
                str(prepared),
                str(output),
            ],
            check=True,
            env=environment,
            cwd="/kaggle/working",
            timeout=helper.remaining_budget(deadline),
        )
        authenticate_project(project, plan)
        helper.remaining_budget(deadline)
        require(
            (output / "engineering/COMPLETE.json").is_file(), "ACTUAL_ENGINEERING_RECEIPT_REQUIRED"
        )
        write_json(
            output / "COMPLETE.json",
            {
                "complete": True,
                "root_plan_sha256": ROOT_PLAN_SHA256,
                "bootstrap": bootstrap,
                "engineering_receipt_sha256": file_sha(output / "engineering/COMPLETE.json"),
                "scope": "two_pre2015_native_adapter_fits_no_final_evaluation",
            },
        )
    except BaseException as error:
        write_json(
            output / "FAILURE.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "reason": str(error),
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:])
    else:
        run("/kaggle/input", "/kaggle/working/holdout-adapter-smoke")
