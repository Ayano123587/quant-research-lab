"""Stdlib-only prospective notebook input authentication; no provider mutations.

The stager must render ROOT_PLAN_SHA256 from an actual reviewed source/data
freeze. Unbound source refuses before package extraction or native imports.
All four whole input bundles and every member are authenticated before the
locked environment helper or any project module is imported.
"""

import hashlib
import importlib.util
import json
import math
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
INPUT_ROLES = ("original_market", "original_prepared_panel", "original_raw_reference")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _budget(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("Startup-inclusive wider notebook service deadline")


def _sha(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            _budget(deadline)
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name):
    path = PurePosixPath(name)
    _require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and not path.is_absolute()
        and all(p not in ("", ".", "..") for p in path.parts)
        and ":" not in name,
        "Unsafe bound member path",
    )
    return path


def _regular(path, root):
    _require(path.resolve().is_relative_to(root.resolve()), "Input path escapes root")
    _require(
        path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
        "Regular contained input file required",
    )


def discover_bound(root, *, sha256, size=None, suffix=None, filename=None, deadline):
    """Recursive exact SHA/size unique discovery, never first-match mounting."""
    root, matches = Path(root), []
    _require(
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(c in "0123456789abcdef" for c in sha256),
        "Externally bound SHA required",
    )
    for path in root.rglob("*"):
        _budget(deadline)
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        if filename is not None and path.name != filename:
            continue
        if size is not None and path.stat().st_size != size:
            continue
        _regular(path, root)
        if _sha(path, deadline) == sha256:
            matches.append(path)
    _require(len(matches) == 1, "Exactly one bound input required")
    return matches[0]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _inspect(container, members, deadline):
    expected = {r["path"]: r for r in members}
    _require(len(expected) == len(members), "Duplicate bound input members")
    with zipfile.ZipFile(container) as archive:
        entries = archive.infolist()
        _require(
            len(entries) == len(expected) and {e.filename for e in entries} == set(expected),
            "Exact whole bundle inventory required",
        )
        for entry in entries:
            _budget(deadline)
            _safe_relative(entry.filename)
            _require(
                not entry.is_dir()
                and not stat.S_ISLNK(entry.external_attr >> 16)
                and not entry.flag_bits & 1,
                "Regular unencrypted bundle members required",
            )
            item = expected[entry.filename]
            _require(entry.file_size == item["bytes"], "Bundle member size mismatch")
            digest = hashlib.sha256()
            with archive.open(entry) as stream:
                while block := stream.read(4 * 1024 * 1024):
                    _budget(deadline)
                    digest.update(block)
            _require(digest.hexdigest() == item["sha256"], "Bundle member SHA mismatch")


def _extract(container, members, root, deadline):
    root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(container) as archive:
        for item in members:
            target = root / _safe_relative(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item["path"]) as source, target.open("xb") as out:
                while block := source.read(4 * 1024 * 1024):
                    _budget(deadline)
                    out.write(block)
            _require(_sha(target, deadline) == item["sha256"], "Extracted input SHA mismatch")


def run(input_root, output_root):
    """Authenticate inputs, create locked interpreter/context; no prepare or fit."""
    started, output = time.monotonic(), Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    _write(
        output / "STARTED.json",
        {"actual_started_at": datetime.now(UTC).isoformat(), "complete": False},
    )
    try:
        deadline = started + 36000
        plan_path = discover_bound(
            input_root, sha256=ROOT_PLAN_SHA256, suffix=".json", deadline=deadline
        )
        plan = json.loads(plan_path.read_bytes())
        _require(set(plan["roots"]) == set(INPUT_ROLES), "All three exact input roots required")
        for value in plan["roots"].values():
            if value != ".":
                _safe_relative(value)
        _require(
            plan["schema"] == "quant_stock_wider_notebook_plan_v1"
            and plan["private"] is True
            and plan["service_seconds"] == 36000
            and plan["holdout_evaluated"] is False,
            "Frozen private wider notebook plan required",
        )
        _require(
            set(plan["data_inputs"]) == set(INPUT_ROLES),
            "Exactly original market/prepared/reference plus source dataset required",
        )
        _require(
            plan["source_approval"]["status"] == "READY"
            and plan["source_approval"]["schema"] == "quant_stock_wider_source_freeze_v1",
            "Actual root source approval required",
        )
        production = {
            item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
            for item in plan["source_files"]
            if item["path"] != "reference/metrics.json"
        }
        _require(
            production == plan["source_approval"]["production_inventory"]
            and plan["runtime"] == plan["source_approval"]["runtime"]
            and plan["data_freeze_sha256"] == plan["source_approval"]["data_freeze_sha256"],
            "Approved exact source/runtime/data closure required",
        )
        _require(
            all(
                plan["data_freezes"][role]["status"] == "READY"
                and plan["data_bindings"][role]["private"] is True
                and type(plan["data_bindings"][role]["dataset_id"]) is int
                and plan["data_bindings"][role]["dataset_id"] > 0
                and plan["data_bindings"][role]["version"] == 1
                for role in INPUT_ROLES
            ),
            "Actual positive private numeric-version data freezes required",
        )
        source_paths = {}
        for item in plan["source_files"]:
            name = str(_safe_relative(item["path"]))
            _require(name not in source_paths, "Duplicate source package path")
            source_paths[name] = discover_bound(
                input_root,
                sha256=item["sha256"],
                size=item["bytes"],
                filename=item["package_name"],
                deadline=deadline,
            )
        containers = {}
        for role, item in plan["data_inputs"].items():
            container = discover_bound(
                input_root, sha256=item["sha256"], size=item["bytes"], deadline=deadline
            )
            _inspect(container, item["members"], deadline)
            containers[role] = container
        # Every whole BIN and every member passed before project/native imports.
        required_tmp = (
            sum(m["bytes"] for item in plan["data_inputs"].values() for m in item["members"])
            + 8 * 1024**3
        )
        _require(
            plan["minimum_free_tmp_bytes"] == required_tmp
            and shutil.disk_usage("/tmp").free >= required_tmp,
            "Insufficient verified private input/runtime disk budget",
        )
        private = Path(tempfile.mkdtemp(prefix="quantlab-wider-", dir="/tmp"))
        project = private / "project"
        for name, path in source_paths.items():
            destination = project / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        for item in plan["source_files"]:
            _require(
                _sha(project / item["path"], deadline) == item["sha256"],
                "Materialized source changed",
            )
        for role, item in plan["data_inputs"].items():
            _extract(containers[role], item["members"], private / role, deadline)
        derived = plan["derived_reference_metrics"]
        metrics_source = project / _safe_relative(derived["source_path"])
        _require(
            derived["source_path"] in source_paths
            and _sha(metrics_source, deadline) == derived["sha256"]
            and metrics_source.stat().st_size == derived["bytes"],
            "Original hash-bound derived reference metrics required",
        )
        reference_root = (
            private / "original_raw_reference" / plan["roots"]["original_raw_reference"]
        )
        reference_manifest = reference_root / "manifest.json"
        _require(
            _sha(reference_manifest, deadline) == plan["reference_manifest_sha256"],
            "Original b46 reference manifest required before metrics restoration",
        )
        reference_inventory = json.loads(reference_manifest.read_bytes())["artifacts"]
        _require(
            any(
                item["path"] == "metrics.json" and item["sha256"] == derived["sha256"]
                for item in reference_inventory
            ),
            "Derived metrics differ from b46 inventory",
        )
        with (reference_root / "metrics.json").open("xb") as stream:
            stream.write(metrics_source.read_bytes())
        helper_path = project / "cloud/stock_development_runner.py"
        _require(
            "cloud/stock_development_runner.py" in source_paths,
            "Hash-bound locked-runtime helper required",
        )
        spec = importlib.util.spec_from_file_location("wider_locked_runtime", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        locked, environment, actual_runtime = helper.bootstrap_with_receipt(
            private, project, output, deadline
        )
        _require(Path(locked).is_file(), "Actual locked interpreter path required")
        # Notebook plotting is a declared locked research dependency, not an
        # opportunistic import from the provider's unrelated base environment.
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
                "research",
                "--project",
                str(project),
                "--python",
                actual_runtime["managed_interpreter"],
            ],
            check=True,
            env=environment,
            timeout=helper.remaining_budget(deadline),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        context = {
            "project": str(project),
            "startup_deadline_monotonic": deadline,
            "normalized_manifest_sha256": plan["normalized_manifest_sha256"],
            "source_plan_sha256": ROOT_PLAN_SHA256,
            "runtime": plan["runtime"],
            "locked_interpreter": str(locked),
            "private_managed_root": str(private / "python"),
            "project_source_sha256": {
                item["path"]: item["sha256"] for item in plan["source_files"]
            },
            "input_container_sha256": {
                str(containers[role]): item["sha256"] for role, item in plan["data_inputs"].items()
            },
            "market_root": str(private / "original_market" / plan["roots"]["original_market"]),
            "prepared_root": str(
                private / "original_prepared_panel" / plan["roots"]["original_prepared_panel"]
            ),
            "reference_root": str(
                private / "original_raw_reference" / plan["roots"]["original_raw_reference"]
            ),
            "wider_root": str(output / "wider-panel"),
            "study_root": str(output / "stock-wider-study"),
            "artifact_root": str(output / "figures"),
        }
        for value in plan["roots"].values():
            if value != ".":
                _safe_relative(value)
        Path(context["artifact_root"]).mkdir()
        context_path = private / "notebook-context.json"
        _write(context_path, context)
        _budget(deadline)
        _write(
            output / "notebook-bootstrap-receipt.json",
            {
                "complete": False,
                "input_authentication_complete": True,
                "source_plan_sha256": ROOT_PLAN_SHA256,
                "roles": list(INPUT_ROLES),
                "runtime_bootstrap": actual_runtime,
                "scientific_study_complete": False,
            },
        )
        return str(locked), str(context_path), environment
    except BaseException as error:
        _write(
            output / "FAILURE.json",
            {
                "complete": False,
                "phase": "wider_notebook_bootstrap",
                "error_type": type(error).__name__,
                "message": str(error)[:1000],
            },
        )
        raise
