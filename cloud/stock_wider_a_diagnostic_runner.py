"""Source-only bounded cloud A diagnostic using the immutable original bootstrap."""

import hashlib
import importlib.util
import json
import math
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT_APPROVAL_SHA256 = "UNBOUND"
PARENT_PLAN_SHA256 = "548f88e827c32933bd9db1cd301babd842056726ca309d58477d36f88fee544d"
CHILD_SHA256 = "UNBOUND"
SERVICE_SECONDS = 1800


def sha(path, deadline):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError("A diagnostic deadline")
            h.update(block)
    return h.hexdigest()


def discover(root, expected, deadline, *, name=None, size=None):
    root = Path(root)
    matches = []
    for path in root.rglob(name or "*"):
        if time.monotonic() >= deadline:
            raise TimeoutError("A diagnostic discovery deadline")
        if not path.is_file():
            continue
        if size is not None and path.stat().st_size != size:
            continue
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Regular non-symlink input required")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Input escapes root")
        if sha(path, deadline) == expected:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("Exactly one SHA-bound input required")
    return matches[0]


def run(input_root="/kaggle/input", output_root="/kaggle/working/a-diagnostic"):
    if ROOT_APPROVAL_SHA256 == "UNBOUND" or CHILD_SHA256 == "UNBOUND":
        raise ValueError("Unbound source-only diagnostic")
    if not Path("/kaggle/input").is_dir():
        raise ValueError("Cloud-only diagnostic")
    started = time.monotonic()
    deadline = started + SERVICE_SECONDS
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "STARTED.json").write_text(
        json.dumps(
            {
                "actual_started_at": datetime.now(UTC).isoformat(),
                "diagnostic_complete": False,
                "root_approval_sha256": ROOT_APPROVAL_SHA256,
                "parent_plan_sha256": PARENT_PLAN_SHA256,
                "fits": 0,
                "service_seconds": SERVICE_SECONDS,
            }
        )
        + "\n"
    )

    def expired(_signum, _frame):
        raise TimeoutError("Startup-inclusive A diagnostic deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, SERVICE_SECONDS)
    try:
        approval_path = discover(input_root, ROOT_APPROVAL_SHA256, deadline, name="*.json")
        approval = json.loads(approval_path.read_bytes())
        if (
            approval["schema"] != "quant_stock_wider_a_diagnostic_source_freeze_v1"
            or approval["status"] != "READY"
            or approval["parent_plan_sha256"] != PARENT_PLAN_SHA256
            or approval["child_sha256"] != CHILD_SHA256
            or approval["service_seconds"] != SERVICE_SECONDS
        ):
            raise ValueError("Root diagnostic approval contract mismatch")
        plan_path = discover(input_root, PARENT_PLAN_SHA256, deadline, name="*.json")
        plan = json.loads(plan_path.read_bytes())
        entry = next(
            i
            for i in plan["source_files"]
            if i["path"] == "cloud/stock_wider_notebook_bootstrap.py"
        )
        bootstrap_path = discover(
            input_root, entry["sha256"], deadline, name=entry["package_name"], size=entry["bytes"]
        )
        child = discover(
            input_root, CHILD_SHA256, deadline, name="stock_wider_a_diagnostic_child.py"
        )
        spec = importlib.util.spec_from_file_location("frozen_wider_bootstrap", bootstrap_path)
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        bootstrap.ROOT_PLAN_SHA256 = PARENT_PLAN_SHA256
        locked, context, env = bootstrap.run(input_root, output / "bootstrap")
        env.update(
            QUANTLAB_A_CONTEXT=str(context),
            QUANTLAB_A_CONTEXT_SHA=sha(context, deadline),
            QUANTLAB_A_OUTPUT=str(output),
            QUANTLAB_A_DEADLINE=str(deadline),
        )
        remaining = deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError("No A diagnostic child budget")
        subprocess.run([locked, str(child)], env=env, check=True, timeout=remaining)
        if sha(child, deadline) != CHILD_SHA256:
            raise ValueError("Diagnostic child changed")
        if (output / "DIAGNOSTIC_COMPLETE").read_text().strip() != sha(
            output / "report.json", deadline
        ):
            raise ValueError("Diagnostic report completion mismatch")
        if time.monotonic() >= deadline:
            raise TimeoutError("A diagnostic final budget")
    except BaseException as error:
        (output / "FAILURE.json").write_text(
            json.dumps(
                {"diagnostic_complete": False, "fits": 0, "error_type": type(error).__name__}
            )
            + "\n"
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    run()
