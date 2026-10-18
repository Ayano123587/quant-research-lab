"""One externally frozen shared evaluation; stdlib authentication precedes imports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
RUNNER_CANONICAL_SHA256 = "UNBOUND"

WIDER_PARENT = {
    "canonical": "aikyatansinha/quant-stock-wider-main-workflow",
    "kernel_id": 139317315,
    "version": 3,
}
WIDER_METADATA = {
    "panel": (
        "wider-run/wider-panel/manifest.json",
        "86765b77307b3dd44b1b74fdf52ef8ac15cab5e61f5131f4f37c9a74b9535fb4",
    ),
    "study": (
        "wider-run/stock-wider-study/manifest.json",
        "108eb61861248eb22ca811ae71c27faed1cefe83ad1b1ee819e579fd54f0c4af",
    ),
    "workflow": (
        "wider-run/workflow-manifest.json",
        "4c935988d8b01d4e669755d4f5b9ee1c3ae4e9f6622f11fc0e989527f533281a",
    ),
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe_name(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and ":" not in name
        and not {"", ".", ".."} & set(name.split("/")),
        "SAFE_RELATIVE_PATH_REQUIRED",
    )
    return path


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def discover_bound(root, name, digest, size=None):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "REGULAR_INPUT_ROOT_REQUIRED")
    resolved = root.resolve(strict=True)
    require(
        safe_name(name).name == name and re.fullmatch(r"[0-9a-f]{64}", digest),
        "BOUND_BASENAME_AND_SHA_REQUIRED",
    )
    found = []
    for path in root.rglob(name):
        require(path.is_file() and not path.is_symlink(), "REGULAR_INPUT_REQUIRED")
        require(path.resolve(strict=True).is_relative_to(resolved), "INPUT_ESCAPES_ROOT")
        parent = path.parent
        while parent != root:
            require(not parent.is_symlink(), "SYMLINK_ANCESTOR_REFUSED")
            parent = parent.parent
        if (size is None or path.stat().st_size == size) and file_sha(path) == digest:
            found.append(path)
    require(len(found) == 1, "ONE_EXACT_BOUND_INPUT_REQUIRED: " + name)
    return found[0]


def authenticate_package(plan, mount):
    require(plan["schema"] == "shared_quant_final_cloud_transport_v1", "FINAL_PLAN_REQUIRED")
    require(set(plan["data"]) == {"market", "prepared", "etf"}, "ALL_THREE_DATA_SOURCES_REQUIRED")
    require(
        0 < plan["service_budget_seconds"] <= 39000, "BOUNDED_STARTUP_INCLUSIVE_BUDGET_REQUIRED"
    )
    require(
        {p.name for p in mount.iterdir()}
        == set(plan["package_files"]) | {"shared-holdout-plan.json"},
        "EXACT_SOURCE_INVENTORY_REQUIRED",
    )
    result = {}
    for flat, item in plan["package_files"].items():
        require(safe_name(flat).name == flat, "FLAT_SOURCE_REQUIRED")
        name = safe_name(item["path"]).as_posix()
        path = mount / flat
        require(
            path.is_file() and not path.is_symlink() and name not in result,
            "REGULAR_UNIQUE_SOURCE_REQUIRED",
        )
        require(
            path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"],
            "RAW_SOURCE_AUTHENTICATION_FAILED",
        )
        result[name] = path.read_bytes()
    require(
        hashlib.sha256(canonical_runner(result["cloud/shared_holdout_runner.py"])).hexdigest()
        == plan["runner_canonical_sha256"],
        "CANONICAL_SOURCE_DIFFERS",
    )
    return result


def authenticate_project(project, plan):
    for row in plan["package_files"].values():
        path = Path(project).joinpath(*safe_name(row["path"]).parts)
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == row["bytes"]
            and file_sha(path) == row["sha256"],
            "MATERIALIZED_SOURCE_DRIFT",
        )


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validate_descriptor(item, freeze, helper, kind):
    helper.validate_data_freeze(freeze)
    if kind == "market":
        declared = freeze["dataset"]
        dataset, dataset_id, version = declared["canonical"], declared["id"], declared["version"]
        container = {"name": "market-v1.bin", **freeze["remote_files"]["market-v1.bin"]}
        transport = freeze["remote_files"]["transport-manifest.json"]["sha256"]
    else:
        dataset, dataset_id, version = freeze["dataset"], freeze["dataset_id"], freeze["version"]
        container, transport = freeze["container"], freeze["transport_manifest_sha256"]
    require(
        item["dataset"] == dataset
        and item["dataset_id"] == dataset_id
        and item["version"] == version
        and item["container_name"] == container["name"]
        and item["container_sha256"] == container["sha256"]
        and item["container_bytes"] == container["bytes"]
        and item["transport_sha256"] == transport,
        "DESCRIPTOR_MUST_MATCH_ACTUAL_IMMUTABLE_DATA_FREEZE",
    )


def normalize_wider_partitions(panel):
    """Actual parent outputs have no kind/session fields; derive only their role."""
    require(
        panel["complete"] is True and panel["decision_count"] == 180,
        "ACTUAL_COMPLETE_180_DATE_WIDER_PANEL_REQUIRED",
    )
    rows, dates, names = [], set(), set()
    for item in panel["outputs"]:
        require(
            set(item) == {"path", "bytes", "rows", "sha256"},
            "EXACT_WIDER_PARTITION_FIELDS_REQUIRED",
        )
        path = safe_name(item["path"])
        require(
            len(path.parts) == 2 and path.parts[0] == "features" and path.suffix == ".parquet",
            "ACTUAL_WIDER_FEATURE_PATH_REQUIRED",
        )
        day = date.fromisoformat(path.stem)
        require(
            day.isoformat() == path.stem
            and day <= date(2014, 12, 31)
            and day not in dates
            and item["path"] not in names,
            "NO_FUTURE_DUPLICATE_OR_DOUBLE_WIDER_DATE",
        )
        require(
            type(item["bytes"]) is int
            and item["bytes"] > 0
            and type(item["rows"]) is int
            and item["rows"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "ACTUAL_WIDER_BYTES_ROWS_SHA_REQUIRED",
        )
        dates.add(day)
        names.add(item["path"])
        rows.append({**item, "kind": "wider_features", "session": day.isoformat()})
    require(
        len(rows) == 180
        and sum(r["rows"] for r in rows) == panel["feature_rows"] == 71309
        and min(dates) == date(2000, 1, 31)
        and max(dates) == date(2014, 12, 31),
        "EXACT_ORIGINAL_180_DATE_71309_ROW_COHORT_REQUIRED",
    )
    return sorted(rows, key=lambda r: r["session"])


def authenticate_wider_metadata(raw, *, project, expected_runtime):
    """JSON/source authentication only, before any restored Parquet can be opened."""
    require(set(raw) == set(WIDER_METADATA), "ALL_THREE_ACTUAL_PARENT_METADATA_REQUIRED")
    for role, (_, digest) in WIDER_METADATA.items():
        require(
            hashlib.sha256(raw[role]).hexdigest() == digest,
            "ACTUAL_IMMUTABLE_WIDER_METADATA_DIFFERS",
        )
    panel, study, workflow = (json.loads(raw[k]) for k in ("panel", "study", "workflow"))
    require(
        study["complete"] is True
        and study["holdout_evaluated"] is False
        and study["reference_reproduced_exactly"] is True
        and study["pins"]["wider_manifest_sha256"] == WIDER_METADATA["panel"][1]
        and workflow["complete"] is True
        and workflow["holdout_evaluated"] is False
        and workflow["study_manifest_sha256"] == WIDER_METADATA["study"][1],
        "ACTUAL_COMPLETED_PARENT_AND_REFERENCE_REPRODUCTION_REQUIRED",
    )
    inventory = {r["path"]: r for r in workflow["artifacts"]}
    require(
        inventory["stock-wider-study/manifest.json"]["sha256"] == WIDER_METADATA["study"][1]
        and inventory["wider-panel/manifest.json"]["sha256"] == WIDER_METADATA["panel"][1],
        "WORKFLOW_MUST_BIND_ACTUAL_STUDY_AND_PANEL",
    )
    policy = panel["config"]["feature_reproduction_policy"]
    require(
        hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == panel["config"]["feature_reproduction_policy_sha256"]
        and policy["whitelist_sha256"]
        == panel["config"]["source_code_sha256"]["configs/stock-feature-portability-whitelist.json"]
        and policy["parent_prepared_manifest_sha256"]
        == panel["config"]["parent_prepared_manifest_sha256"],
        "EXACT_PARENT_PORTABILITY_POLICY_REQUIRED",
    )
    for name, digest in panel["config"]["source_code_sha256"].items():
        relative = "src/" + name if name.startswith("quantlab/") else name
        path = Path(project).joinpath(*safe_name(relative).parts)
        require(
            path.is_file() and not path.is_symlink() and file_sha(path) == digest,
            "UNCHANGED_PARENT_FEATURE_SOURCE_REQUIRED",
        )
    parent_runtime = study["pins"]["runtime"]
    require(
        parent_runtime["python"]
        == panel["config"]["runtime"]["python"]
        == expected_runtime["python"]
        and parent_runtime["processor"] == expected_runtime["machine"]
        and all(
            expected_runtime["packages"].get(p) == parent_runtime[p]
            for p in ("lightgbm", "numpy", "pyarrow", "scikit-learn", "scipy", "threadpoolctl")
        )
        and panel["config"]["runtime"]["pyarrow"] == expected_runtime["packages"]["pyarrow"],
        "UNCHANGED_PARENT_NUMERICAL_RUNTIME_REQUIRED",
    )
    return normalize_wider_partitions(panel)


def authenticate_wider_review(raw, digest):
    require(hashlib.sha256(raw).hexdigest() == digest, "ACTUAL_ROOT_WIDER_REPLAY_REVIEW_DIFFERS")
    review = json.loads(raw)
    require(
        review.get("schema") == "shared_quant_wider_parent_replay_approval_v1"
        and review.get("parent") == WIDER_PARENT
        and review.get("study_manifest_sha256") == WIDER_METADATA["study"][1]
        and review.get("panel_manifest_sha256") == WIDER_METADATA["panel"][1]
        and review.get("workflow_manifest_sha256") == WIDER_METADATA["workflow"][1]
        and review.get("full_prediction_and_artifact_replay_passed") is True
        and review.get("wider_partition_count") == 180,
        "ACTUAL_FULL_PARENT_REPLAY_APPROVAL_REQUIRED_BEFORE_RESTORE",
    )


def restore_wider_parent(
    input_root, destination, binding, raw, review_raw, *, project, expected_runtime, budget
):
    require(
        binding["parent"] == WIDER_PARENT and binding["destination"] == "wider",
        "ONLY_ACTUAL_VERSION3_WIDER_PARENT_ALLOWED",
    )
    authenticate_wider_review(review_raw, binding["full_replay_review_sha256"])
    expected = authenticate_wider_metadata(raw, project=project, expected_runtime=expected_runtime)
    require(binding["partitions"] == expected, "EXTERNAL_WIDER_INVENTORY_DIFFERS")
    locations = {}
    for role, (relative, digest) in WIDER_METADATA.items():
        path = discover_bound(input_root, Path(relative).name, digest, len(raw[role]))
        require(
            tuple(path.parts[-len(PurePosixPath(relative).parts) :])
            == PurePosixPath(relative).parts,
            "ACTUAL_PARENT_RELATIVE_METADATA_PATH_REQUIRED",
        )
        locations[role] = path
    panel_root = locations["panel"].parent
    feature_root = panel_root / "features"
    require(
        feature_root.is_dir() and not feature_root.is_symlink(),
        "REGULAR_WIDER_FEATURE_ROOT_REQUIRED",
    )
    actual = set()
    for path in feature_root.rglob("*"):
        require(not path.is_symlink(), "WIDER_SYMLINK_REFUSED")
        if path.is_file():
            actual.add(path.relative_to(panel_root).as_posix())
        else:
            require(path.is_dir(), "NONREGULAR_WIDER_MEMBER_REFUSED")
    require(actual == {r["path"] for r in expected}, "EXACT_WIDER_DIRECTORY_INVENTORY_REQUIRED")
    # Authenticate ALL180 native blobs before copying any of them or importing numerical code.
    for row in expected:
        budget()
        path = panel_root.joinpath(*safe_name(row["path"]).parts)
        require(
            path.resolve(strict=True).is_relative_to(panel_root.resolve(strict=True))
            and path.stat().st_size == row["bytes"]
            and file_sha(path) == row["sha256"],
            "WIDER_PARTITION_BYTES_DIFFER_BEFORE_NATIVE_IMPORT",
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for row in expected:
        budget()
        source = panel_root.joinpath(*safe_name(row["path"]).parts)
        target = destination.joinpath(*safe_name(row["path"]).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(
            target.stat().st_size == row["bytes"] and file_sha(target) == row["sha256"],
            "COPIED_WIDER_PARTITION_BYTES_DIFFER",
        )
    (destination / "manifest.json").write_bytes(raw["panel"])
    for role, path in locations.items():
        require(file_sha(path) == WIDER_METADATA[role][1], "PARENT_METADATA_CHANGED_DURING_COPY")
    return {
        "parent": WIDER_PARENT,
        "partitions": expected,
        "opaque_copy_only": True,
        "native_values_parsed": False,
        "total_bytes": sum(r["bytes"] for r in expected),
    }


def runtime(expected):
    from importlib.metadata import version

    actual = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {p: version(p) for p in expected["packages"]},
    }
    require(actual == expected, "EXACT_FROZEN_RUNTIME_REQUIRED")
    return actual


def child(plan_path, expected_plan_sha, project, inputs, output, deadline):
    require(platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "KAGGLE_ONLY")
    require(file_sha(plan_path) == expected_plan_sha, "CHILD_PLAN_DIFFERS")
    plan = json.loads(Path(plan_path).read_bytes())
    authenticate_package(plan, Path(plan_path).parent)
    authenticate_project(project, plan)
    require(time.monotonic() < float(deadline), "STARTUP_INCLUSIVE_DEADLINE_EXHAUSTED")
    actual_runtime = runtime(plan["runtime"])
    release_path = Path(project) / "release/final-release.json"
    require(file_sha(release_path) == plan["release_sha256"], "ACTUAL_RELEASE_DIFFERS")
    release = json.loads(release_path.read_bytes())
    require(release["runtime"] == actual_runtime, "RELEASE_RUNTIME_DIFFERS")
    for item in release["input_files"].values():
        path = Path(inputs).joinpath(*safe_name(item["path"]).parts)
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == item["bytes"]
            and file_sha(path) == item["sha256"],
            "FROZEN_INPUT_DIFFERS_BEFORE_NATIVE_IMPORT",
        )
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.evaluation.holdout_pipeline import run_frozen_pipeline

    calendar = plan["calendar"]
    schedule = XNYSSchedule(
        date.fromisoformat(calendar["start"]), date.fromisoformat(calendar["end"])
    )
    require(schedule.provenance == calendar, "ACTUAL_CALENDAR_DIFFERS")
    write_json(Path(output) / "runtime-before-native.json", actual_runtime)
    result = run_frozen_pipeline(
        release_path,
        plan["release_sha256"],
        project=project,
        inputs=inputs,
        schedule=schedule,
        output=Path(output) / "evaluation",
        dispatch_ticket_path=Path(project) / "release/ticket.json",
        expected_dispatch_ticket_sha256=plan["ticket_sha256"],
    )
    require(time.monotonic() < float(deadline), "FINAL_COMPLETION_DEADLINE_EXHAUSTED")
    authenticate_project(project, plan)
    runtime(plan["runtime"])
    require(result["global_protocol_complete"] is True, "ALL_REGISTERED_CONSUMERS_REQUIRED")


def run(input_root, output):
    started = time.monotonic()
    require(platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "KAGGLE_ONLY")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "startup.json", {"complete": False, "started_at": datetime.now(UTC).isoformat()}
    )
    try:
        require(
            hashlib.sha256(canonical_runner(Path(__file__).read_bytes())).hexdigest()
            == RUNNER_CANONICAL_SHA256,
            "ACTUAL_KERNEL_SOURCE_DIFFERS",
        )
        plan_path = discover_bound(input_root, "shared-holdout-plan.json", ROOT_PLAN_SHA256)
        plan = json.loads(plan_path.read_bytes())
        raw = authenticate_package(plan, plan_path.parent)
        deadline = started + plan["service_budget_seconds"]

        def budget():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("Startup-inclusive shared evaluation deadline exhausted")
            return left

        private = Path(tempfile.mkdtemp(prefix="quant-shared-final-", dir="/tmp"))
        project, inputs = private / "project", private / "inputs"
        for name, data in raw.items():
            path = project.joinpath(*safe_name(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        authenticate_project(project, plan)
        helpers = {
            "market": module(project / "cloud/risk_forecast_runner.py", "bound_market"),
            "prepared": module(project / "cloud/stock_development_runner.py", "bound_prepared"),
            "etf": module(project / "cloud/etf_development_runner.py", "bound_etf"),
        }
        for kind, item in plan["data"].items():
            budget()
            helper = helpers[kind]
            freeze = json.loads(raw[item["freeze_path"]])
            validate_descriptor(item, freeze, helper, kind)
            container = discover_bound(
                input_root,
                item["container_name"],
                item["container_sha256"],
                item["container_bytes"],
            )
            transport = discover_bound(
                input_root, "transport-manifest.json", item["transport_sha256"]
            )
            destination = inputs.joinpath(*safe_name(item["destination"]).parts)
            proof = helper.extract_data(container, transport.read_bytes(), destination, budget)
            write_json(output / (kind + "-authentication.json"), proof)
        for relative, record in plan["auxiliary_inputs"].items():
            destination = inputs.joinpath(*safe_name(relative).parts)
            require(not destination.exists(), "AUXILIARY_INPUT_COLLISION")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw[record["package_path"]])
        if plan.get("wider_parent") is not None:
            binding = plan["wider_parent"]
            metadata = {role: raw[path] for role, path in binding["metadata_package_paths"].items()}
            proof = restore_wider_parent(
                input_root,
                inputs / "wider",
                binding,
                metadata,
                raw[binding["review_package_path"]],
                project=project,
                expected_runtime=plan["runtime"],
                budget=budget,
            )
            write_json(output / "wider-parent-authentication.json", proof)
        budget()
        interpreter, environment, receipt = helpers["prepared"].bootstrap(
            private, project, deadline
        )
        # One locked resolution installs both groups; the second sync must not remove ML.
        subprocess.run(
            [
                str(private / "uv-package/bin/uv"),
                "sync",
                "--locked",
                "--no-install-project",
                "--no-default-groups",
                "--group",
                "ml",
                "--group",
                "optimization",
                "--project",
                str(project),
                "--python",
                str(interpreter),
            ],
            env=environment,
            check=True,
            timeout=budget(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        write_json(output / "bootstrap.json", receipt)
        environment["QUANTLAB_SERVICE_DEADLINE"] = str(deadline)
        log = output / "child.log"
        with log.open("xb") as stream:
            subprocess.run(
                [
                    str(interpreter),
                    str(project / "cloud/shared_holdout_runner.py"),
                    "--child",
                    str(plan_path),
                    ROOT_PLAN_SHA256,
                    str(project),
                    str(inputs),
                    str(output),
                    str(deadline),
                ],
                env=environment,
                cwd="/kaggle/working",
                check=True,
                timeout=budget(),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        budget()
        authenticate_project(project, plan)
        complete = output / "evaluation/COMPLETE"
        require(complete.is_file(), "ACTUAL_PROTOCOL_COMPLETION_RECEIPT_REQUIRED")
        write_json(
            output / "COMPLETE.json",
            {
                "global_protocol_complete": True,
                "release_sha256": plan["release_sha256"],
                "plan_sha256": ROOT_PLAN_SHA256,
                "evaluation_complete_sha256": file_sha(complete),
                "scope": "registered_evaluation_protocol_not_universal_performance_support",
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
                "partial_outputs_preserved": True,
            },
        )
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:])
    else:
        run("/kaggle/input", "/kaggle/working/shared-final-evaluation")
