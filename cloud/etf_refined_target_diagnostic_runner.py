"""Prospective one-refinement target diagnostic; authenticate bytes before native imports."""

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
PLAN_NAME = "etf-refined-target-diagnostic-plan.json"
PARENT_RUNNER_SHA = "29998d97ce32afbaa34da8f80a423319f986bbf4806a3b18184d0a17669cd5f0"


def validate_plan(plan):
    require(
        plan.get("schema") == "quant_etf_refined_target_diagnostic_plan_v1"
        and plan.get("action")
        == "refined_risk_budget_targets_original_KKT_stop_first_exception_no_backtest"
        and plan.get("scenario") == "weight_risk_budget_only"
        and plan.get("initial_signal") == "2007-02-28"
        and plan.get("end_signal") == "2014-12-31"
        and plan.get("scheduled_signals") == 95
        and plan.get("parent_kernel_id") == 139275071
        and plan.get("parent_kernel_version") == 1
        and plan.get("parent_runner_sha256") == PARENT_RUNNER_SHA
        and plan.get("original_diagnostic_report_sha256")
        == "4b04679506bdaa8b27f02531f11f67a526cc061806aaa197975634094eb7d14f"
        and plan.get("refinement_review_sha256")
        == "d76d9e29b95fda0bab6e0a8bceaba37533ae8abeeef2723eacf8f3e5d8862d8e"
        and plan.get("numerical_amendment")
        == {
            "path": "src/quantlab/allocation/weighting.py",
            "old_sha256": "d090c64bccc1a96b8257f2e5b4573d46314635f19c8a69fef97675da60e8457a",
            "new_sha256": "6407da185bd1c89798e75ad438bff78ef4fd95a16a9ca3e07ec2c5328e0b2971",
            "policy": (
                "one_original_successful_feasible_KKT_failure_same_problem_"
                "warmstart_ftol1e-14_strict_objective_nonincrease_original_gates"
            ),
        }
        and plan.get("holdout_outcomes") is False
        and plan.get("backtest") is False
        and plan.get("scores") is False
        and plan.get("python") == "3.12.15"
        and plan.get("threads") == 4
        and 600 <= plan.get("service_budget_seconds", 0) <= 36000,
        "EXACT_TARGET_DIAGNOSTIC_ACTION_REQUIRED",
    )


def plain(value):
    """Only explicit primitive numeric values, never frame objects or arbitrary locals."""
    import math

    import numpy as np

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), "FINITE_DIAGNOSTIC_SCALAR_REQUIRED")
        return value
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    raise ValueError("Unallowlisted diagnostic value type")


def capture_solver_failure(error, weighting_path, expected_sha):
    require(file_sha(weighting_path) == expected_sha, "FROZEN_WEIGHTING_SOURCE_REQUIRED")
    current, matches = error.__traceback__, []
    while current is not None:
        frame = current.tb_frame
        if (
            frame.f_code.co_name == "strategic_weights"
            and Path(frame.f_code.co_filename).resolve() == weighting_path.resolve()
        ):
            matches.append(current)
        current = current.tb_next
    if not matches:
        return None
    require(len(matches) == 1, "ONE_EXACT_PRODUCTION_SOLVER_FRAME_REQUIRED")
    tb = matches[0]
    local, observed = tb.tb_frame.f_locals, {}
    allowed = (
        "matrix",
        "scaled",
        "n",
        "cap",
        "initial",
        "cap_constraint",
        "slack",
        "active",
        "grad",
        "multipliers",
        "stationarity",
        "stationarity_residual",
        "complementarity",
    )
    for name in allowed:
        if name in local:
            observed[name] = plain(local[name])
    if "result" in local:
        result = local["result"]
        observed["scipy_result"] = {
            name: plain(result[name])
            for name in (
                "x",
                "fun",
                "status",
                "success",
                "message",
                "nit",
                "nfev",
                "njev",
                "multipliers",
            )
            if name in result
        }
    estimate = local["estimate"]
    observed["estimate"] = {
        "window_hash": estimate.window_hash,
        "asset_ids": list(estimate.asset_ids),
        "units": estimate.units,
        "matrix": plain(estimate.matrix),
        "training_start": estimate.training_start.isoformat(),
        "training_end": estimate.training_end.isoformat(),
        "max_available_at": estimate.max_available_at.isoformat(),
    }
    return {
        "error_type": type(error).__name__,
        "message": str(error)[:2000],
        "source_sha256": expected_sha,
        "function": "strategic_weights",
        "line": tb.tb_lineno,
        "observed_allowlisted_locals": observed,
        "not_present_at_exception": [name for name in allowed if name not in local],
    }


def diagnostic_signal_dates(schedule):
    days = tuple(
        schedule.month_end(y, m).session
        for y in range(2007, 2015)
        for m in range(1, 13)
        if date(2007, 2, 1) <= date(y, m, 1) <= date(2014, 12, 1)
    )
    require(
        len(days) == 95
        and days[0] == date(2007, 2, 28)
        and days[-1] == date(2014, 12, 31)
        and all(type(day) is date for day in days),
        "EXACT_95_REAL_SIGNAL_DATES_REQUIRED",
    )
    return days


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
    from bisect import bisect_right
    from importlib.metadata import version

    from quantlab.allocation.policies import ASSET_IDS
    from quantlab.allocation.sensitivity import SCENARIOS, sensitivity_target
    from quantlab.backtest.engine import HistoricalView
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.provenance import identity_from_source
    from quantlab.data.yahoo import YahooETFRelease

    require(
        Path(sys.modules[sensitivity_target.__module__].__file__).resolve()
        == project / "src/quantlab/allocation/sensitivity.py",
        "PROJECT_IMPORT_DIFFERS",
    )
    runtime = {name: version(name) for name in plan["runtime"]}
    require(runtime == plan["runtime"], "LOCKED_RUNTIME_DIFFERS")
    deadline = float(os.environ["QUANTLAB_SERVICE_DEADLINE"]) - 90

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("Startup-inclusive target diagnostic budget exhausted")

    budget()
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    release = YahooETFRelease(
        etf_root, "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3", schedule
    )
    identity = identity_from_source(release, schedule)
    histories = {a: release.bars("etf", a.split(":")[1]) for a in ASSET_IDS}
    indices = {a: tuple(b.session for b in rows) for a, rows in histories.items()}
    days = diagnostic_signal_dates(schedule)
    require(
        len(days) == 95
        and str(days[0]) == plan["initial_signal"]
        and str(days[-1]) == plan["end_signal"],
        "EXACT_95_REAL_MONTH_ENDS_REQUIRED",
    )
    scenario = next(s for s in SCENARIOS if s.scenario_id == "weight_risk_budget_only")
    weighting = project / "src/quantlab/allocation/weighting.py"
    weighting_sha = plan["approved_project_files"]["src/quantlab/allocation/weighting.py"]
    successes, failure = [], None
    for day in days:
        budget()
        view = HistoricalView(
            tuple((a, rows[: bisect_right(indices[a], day)]) for a, rows in histories.items()),
            day,
            schedule.session(day).decision_at,
            schedule.next_session(day).open_at,
        )
        try:
            target = sensitivity_target(view, schedule, identity, scenario)
        except Exception as error:
            captured = capture_solver_failure(error, weighting, weighting_sha)
            if captured is None:
                raise
            failure = {
                "signal_session": str(day),
                "signal_at": view.cutoff.isoformat(),
                "next_scheduled_open_at": view.next_open_at.isoformat(),
                **captured,
            }
            break
        successes.append(
            {
                "signal_session": str(day),
                "signal_at": view.cutoff.isoformat(),
                "covariance_window_sha256": target.source_target.covariance.window_hash,
                "strategic_weights": list(target.strategic_weights),
                "applied_weights": list(target.weights),
                "cash_weight": target.cash_weight,
                "solver": target.strategic_certificate.solver,
            }
        )
    for item in plan["package_files"].values():
        read_bound(project / item["path"], item["sha256"], item["bytes"])
    require({name: version(name) for name in plan["runtime"]} == runtime, "RUNTIME_CHANGED")
    YahooETFRelease(
        etf_root, "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3", schedule
    )
    budget()
    report = {
        "action": plan["action"],
        "diagnostic_finished": True,
        "status": "first_actual_production_exception_captured"
        if failure
        else "all95_targets_passed",
        "planned_signals": len(days),
        "successful_target_count": len(successes),
        "successful_targets": successes,
        "first_exception": failure,
        "parent_kernel_id": 139275071,
        "parent_kernel_version": 1,
        "parent_runner_sha256": PARENT_RUNNER_SHA,
        "source_identity": identity.__dict__,
        "runtime": runtime,
        "backtest_run": False,
        "performance_scores_computed": False,
        "holdout_outcomes_constructed": False,
        "production_algorithm_changed": True,
        "numerical_amendment": plan["numerical_amendment"],
        "original_diagnostic_report_sha256": plan["original_diagnostic_report_sha256"],
    }
    (output / "target-diagnostic.json").write_bytes(encoded(report))
    report_sha = file_sha(output / "target-diagnostic.json")
    budget()
    (output / "child-terminal.json").write_bytes(
        encoded(
            {
                "diagnostic_finished": True,
                "status": report["status"],
                "report_sha256": report_sha,
                "economic_study_complete": False,
                "backtest_or_scores_run": False,
            }
        )
    )


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_bound(path, digest, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "REGULAR_BOUND_FILE_REQUIRED")
    raw = path.read_bytes()
    require(sha(raw) == digest and (size is None or len(raw) == size), "BOUND_BYTE_MISMATCH")
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
            rb"(?m)^" + name.encode() + rb' = "[^"\n]+"$',
            name.encode() + b' = "UNBOUND"',
            raw,
        )
        require(count == 1, "SINGLE_RUNNER_ASSIGNMENT_REQUIRED")
    return raw


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
        sha(canonical_runner(raw["cloud/etf_refined_target_diagnostic_runner.py"]))
        == plan["runner_canonical_sha256"],
        "CANONICAL_RUNNER_DIFFERS",
    )
    return raw


def run(input_root, output):
    started = time.monotonic()
    require(platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "KAGGLE_ONLY")
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
            sha(canonical_runner(Path(__file__).read_bytes())) == RUNNER_CANONICAL_SHA256,
            "KERNEL_RUNNER_DIFFERS",
        )
        raw = authenticate_package(plan, plan_path.parent)
        deadline = started + plan["service_budget_seconds"]

        def budget():
            if time.monotonic() >= deadline:
                raise TimeoutError("Startup-inclusive ETF target diagnostic budget exhausted")

        budget()
        phase = "authenticated_data_extraction"
        with tempfile.TemporaryDirectory(
            prefix="quant-etf-refined-target-diagnostic-", dir="/tmp"
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
            interpreter, environment, receipt = market_helper.bootstrap(private, project, deadline)
            environment["QUANTLAB_SERVICE_DEADLINE"] = str(deadline)
            (output / "runtime-bootstrap.json").write_bytes(encoded(receipt))
            phase = "native_target_diagnostic_child"
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
            require(file_sha(etf_bin) == etf_helper.DATA_SHA, "INPUT_POSTFLIGHT_DIFFERS")
            budget()
            (output / "transport-terminal.json").write_bytes(
                encoded(
                    {
                        "transport_postflight": "PASS",
                        "child_terminal_sha256": file_sha(output / "child-terminal.json"),
                        "target_diagnostic_only_no_backtest_or_score": True,
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
        run(Path("/kaggle/input"), Path("/kaggle/working/etf-refined-target-diagnostic"))
