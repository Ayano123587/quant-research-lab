"""Authenticated cloud cost study, reusing the reviewed source/data transport."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path, PurePosixPath

ROOT_PLAN_SHA256 = "UNBOUND"
PLAN_NAME = "etf-cost-plan.json"
SELF = "cloud/etf_cost_runner.py"
HELPER = "cloud/etf_development_runner.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(raw):
    # Rendered binding uses JSON double quotes, as does the unbound template.
    line = ("ROOT_PLAN_SHA256 = " + json.dumps(ROOT_PLAN_SHA256)).encode()
    if raw.count(line) != 1:
        raise ValueError("One literal externally bound plan required")
    return raw.replace(line, b"ROOT_PLAN_SHA256" + b' = "UNBOUND"')


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name or p.as_posix() != name:
        raise ValueError("Safe relative source path required")
    return p


def read(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular authenticated source required")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("Authenticated bytes differ")
    return raw


def authenticate(plan_path, project=None):
    raw = read(plan_path, ROOT_PLAN_SHA256)
    plan = json.loads(raw)
    if (
        plan["schema"] != "quant_etf_cost_transport_v1"
        or plan["costs_bps"] != [0.0, 5.0, 10.0, 25.0]
        or plan["max_seconds"] != 36000
        or plan["full_measured_sessions"] != 1975
        or plan["holdout_evaluated"] is not False
        or not plan["files"]
    ):
        raise ValueError("Full four-fee development-only plan required")
    files, flat_names = {}, set()
    for name, entry in plan["files"].items():
        safe(name)
        flat = entry["upload_name"]
        if Path(flat).name != flat or flat in flat_names or "/" in flat or "\\" in flat:
            raise ValueError("Unique flat source inventory required")
        flat_names.add(flat)
        path = Path(project) / name if project is not None else Path(plan_path).parent / flat
        value = read(path, entry["sha256"])
        if len(value) != entry["bytes"]:
            raise ValueError("Source length differs")
        files[name] = value
    if project is None:
        present = {p.name for p in Path(plan_path).parent.iterdir() if p.is_file()}
        if present != flat_names | {PLAN_NAME}:
            raise ValueError("Exact remote package inventory required")
    if canonical(Path(__file__).read_bytes()) != files[SELF].replace(b"\r\n", b"\n"):
        raise ValueError("Executed runner differs from reviewed source")
    return plan, files


def helper_from(project, plan):
    path = Path(project) / HELPER
    read(path, plan["files"][HELPER]["sha256"])
    spec = importlib.util.spec_from_file_location("reviewed_etf_transport", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def verify_extracted(snapshot, transport):
    manifest = json.loads(transport)
    for name, entry in manifest["members"].items():
        path = Path(snapshot).joinpath(*safe(name).parts)
        raw = read(path, entry["sha256"])
        if len(raw) != entry["bytes"]:
            raise ValueError("Extracted actual data member length differs")


def bootstrap_interpreter(private, bootstrap):
    """Allow the venv symlink only to the exact private managed interpreter."""
    private = Path(private)
    python = Path(bootstrap["python"])
    managed = Path(bootstrap["evidence"]["managed_interpreter"])
    expected = private / "runtime/.venv/bin/python"
    if (
        python.absolute() != expected.absolute()
        or not python.is_file()
        or python.resolve() != managed.resolve()
        or not managed.resolve().is_relative_to((private / "python").resolve())
    ):
        raise ValueError("Exact private venv and managed interpreter required")
    return python


def child(mode, plan_path, project, private, output, deadline):
    plan, _ = authenticate(plan_path, project)
    helper = helper_from(project, plan)
    if mode == "bootstrap":
        interpreter, _, evidence = helper.bootstrap_with_receipt(
            Path(private), Path(project), Path(output)
        )
        helper.write_json(
            Path(output) / "bootstrap-result.json",
            {"python": str(interpreter), "evidence": evidence},
        )
        return
    from importlib.metadata import version

    runtime = {name: version(name) for name in plan["runtime"]}
    if platform.python_version() != "3.12.15" or runtime != plan["runtime"]:
        raise ValueError("Exact locked study runtime required")
    snapshot = Path(private) / "data"
    transport = read(Path(private) / "transport-manifest.json", helper.TRANSPORT_SHA)
    verify_extracted(snapshot, transport)
    # All source and all actual data bytes are authenticated before these imports.
    from quantlab.allocation.robustness import run_cost_study
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.provenance import identity_from_source
    from quantlab.data.yahoo import YahooETFRelease

    if (
        Path(sys.modules[run_cost_study.__module__].__file__).resolve()
        != (Path(project) / "src/quantlab/allocation/robustness.py").resolve()
    ):
        raise ValueError("Exact materialized scientific import path required")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    release = YahooETFRelease(snapshot, helper.MANIFEST_SHA, schedule)
    histories = {asset: release.bars("etf", asset.split(":")[1]) for asset in release.asset_ids}
    run_cost_study(
        histories,
        schedule,
        source_identity=identity_from_source(release, schedule),
        output=Path(output) / "study",
        service_deadline_monotonic=deadline - 60,
    )
    authenticate(plan_path, project)
    verify_extracted(snapshot, transport)
    if {name: version(name) for name in plan["runtime"]} != runtime:
        raise ValueError("Study runtime changed")
    helper.write_json(
        Path(output) / "child-complete.json",
        {
            "complete": True,
            "runtime": runtime,
            "source_plan_sha256": ROOT_PLAN_SHA256,
            "all_actual_data_bytes_postverified": True,
        },
    )


def run(input_root, output):
    deadline = time.monotonic() + 36000
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = {"complete": False, "stage": "startup", "plan_sha256": ROOT_PLAN_SHA256}

    def status():
        (output / "status.json").write_text(json.dumps(state, indent=2) + "\n")

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Cloud service budget exhausted")
        return value

    status()
    try:
        matches = [
            p for p in Path(input_root).rglob(PLAN_NAME) if sha(p.read_bytes()) == ROOT_PLAN_SHA256
        ]
        if len(matches) != 1:
            raise ValueError("One externally bound source package required")
        plan_path = matches[0]
        plan, files = authenticate(plan_path)
        private = Path(tempfile.mkdtemp(prefix="quant-etf-cost-", dir="/tmp"))
        project = private / "project"
        project.mkdir()
        for name, raw in files.items():
            path = project.joinpath(*safe(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        helper = helper_from(project, plan)
        helper.validate_data_freeze(json.loads(files["provenance/data-freeze.json"]))
        transports = [
            p
            for p in Path(input_root).rglob("transport-manifest.json")
            if sha(p.read_bytes()) == helper.TRANSPORT_SHA
        ]
        if len(transports) != 1:
            raise ValueError("One exact original approved Yahoo data transport required")
        transport = read(transports[0], helper.TRANSPORT_SHA)
        (private / "transport-manifest.json").write_bytes(transport)
        proof = helper.extract_data(
            transports[0].parent / helper.CONTAINER_NAME, transport, private / "data", remaining
        )
        helper.write_json(output / "data-proof.json", proof)
        env = helper.private_environment(private, project)
        env.update(
            {
                name: "4"
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
        )
        state["stage"] = "bootstrap"
        status()
        arguments = [str(plan_path), str(project), str(private), str(output), str(deadline)]
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--bootstrap", *arguments],
            check=True,
            env=env,
            cwd=project,
            timeout=remaining(),
        )
        bootstrap = json.loads((output / "bootstrap-result.json").read_bytes())
        python = bootstrap_interpreter(private, bootstrap)
        state["stage"] = "full_cost_family"
        status()
        subprocess.run(
            [str(python), "-s", "-P", str(Path(__file__).resolve()), "--study", *arguments],
            check=True,
            env=env,
            cwd=project,
            timeout=remaining(),
        )
        authenticate(plan_path)
        authenticate(plan_path, project)
        verify_extracted(private / "data", transport)
        read(transports[0].parent / helper.CONTAINER_NAME, helper.DATA_SHA)
        if json.loads((output / "child-complete.json").read_bytes())["complete"] is not True:
            raise ValueError("Scientific child did not complete")
        state.update(complete=True, stage="complete")
        status()
    except BaseException as error:
        state.update(
            complete=False, stage="failed", error_type=type(error).__name__, reason=str(error)[:700]
        )
        status()
        raise


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] in ("--bootstrap", "--study"):
        child(sys.argv[1][2:], *sys.argv[2:6], float(sys.argv[6]))
    else:
        run("/kaggle/input", "/kaggle/working/etf-cost-study")
