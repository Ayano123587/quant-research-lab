"""Opaque, cloud-only export of completed wider-stock outputs; no fitting."""

import hashlib
import json
import os
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

PARENT_KERNEL_ID = 139317315
PARENT_VERSION = 3
PARENT_CANONICAL = "aikyatansinha/quant-stock-wider-main-workflow"
SOURCE_PLAN = "7c4f4661d6a85e6afc649e928dc44217cdfcf2b37d2f17e7e7fb0b1f90c43a8a"
WORKFLOW = "4c935988d8b01d4e669755d4f5b9ee1c3ae4e9f6622f11fc0e989527f533281a"
STUDY = "108eb61861248eb22ca811ae71c27faed1cefe83ad1b1ee819e579fd54f0c4af"
PANEL = "86765b77307b3dd44b1b74fdf52ef8ac15cab5e61f5131f4f37c9a74b9535fb4"
NOTEBOOK = "8316650cfd47715c2b3d2637d41be21ec6a85ff5325c2aa6b3a95ae56a905f8a"
STARTED = time.monotonic()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path, root):
    """Reject links at every level, escapes, devices, and nonregular files."""
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("Contained parent output required")
    current = path
    while True:
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("Symlinks are forbidden")
        if current == root:
            break
        current = current.parent
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Regular parent output required")


def budget():
    if time.monotonic() - STARTED >= 1800:
        raise TimeoutError("Whole opaque export service budget exhausted")


def read_bound(path, root, digest, size):
    budget()
    regular(path, root)
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Externally bound parent bytes changed")
    return raw


def export(root, output):
    root, output = Path(root), Path(output)
    if sys.platform != "linux" or root.resolve() != Path("/kaggle/input"):
        raise ValueError("Actual private cloud parent mount required")
    output.mkdir(parents=True, exist_ok=False)
    matches = []
    for path in root.rglob("workflow-manifest.json"):
        budget()
        regular(path, root)
        if path.stat().st_size == 347137 and sha(path) == WORKFLOW:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("Exactly one pinned completed parent workflow required")
    parent = matches[0].parent
    workflow = json.loads(read_bound(matches[0], parent, WORKFLOW, 347137))
    marker = parent / "NOTEBOOK_COMPLETE"
    if (
        read_bound(
            marker, parent, "c9a0a5c2fc68486ca005bc6acd3be1a425917c91a86db28b2ca7e3b63614be76", 65
        )
        .decode()
        .strip()
        != WORKFLOW
    ):
        raise ValueError("Actual parent workflow completion marker required")
    study_path = parent / "stock-wider-study/manifest.json"
    panel_path = parent / "wider-panel/manifest.json"
    study = json.loads(read_bound(study_path, parent, STUDY, 288363))
    panel = json.loads(read_bound(panel_path, parent, PANEL, 656651))
    if not (
        workflow["complete"] is True
        and workflow["cells"] == 384
        and workflow["paired_mature_months"] == 94
        and workflow["holdout_evaluated"] is False
        and workflow["source_plan_sha256"] == SOURCE_PLAN
        and workflow["study_manifest_sha256"] == STUDY
        and study["complete"] is True
        and study["cells"] == 384
        and study["common_mature_months"] == 94
        and study["reference_reproduced_exactly"] is True
        and study["holdout_evaluated"] is False
        and panel["complete"] is True
        and panel["decision_count"] == 180
        and panel["feature_rows"] == 71309
        and len(panel["accepted_feature_representation_differences"]) == 896
    ):
        raise ValueError("Exact full completed development scope required")
    entries = list(workflow["artifacts"])
    entries.extend(
        {"path": name, "bytes": (parent / name).stat().st_size, "sha256": sha(parent / name)}
        for name in ("workflow-manifest.json", "NOTEBOOK_COMPLETE")
    )
    inventory = {entry["path"]: entry for entry in entries}
    if len(entries) != len(inventory) or len(entries) != 1827:
        raise ValueError("All 1827 unique original workflow files required")
    for name in inventory:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Safe original relative artifact identities required")
        if relative.suffix not in (".json", ".png", ".parquet", ".txt") and relative.name not in (
            "COMPLETE",
            "NOTEBOOK_COMPLETE",
        ):
            raise ValueError("No raw input, cache or undeclared artifact types")
    for entry in study["artifacts"]:
        if inventory["stock-wider-study/" + entry["path"]] != {
            **entry,
            "path": "stock-wider-study/" + entry["path"],
        }:
            raise ValueError("Whole study inventory must equal workflow inventory")
    for entry in panel["outputs"]:
        match = inventory["wider-panel/" + entry["path"]]
        if any(match[k] != entry[k] for k in ("bytes", "sha256")):
            raise ValueError("All 180 actual panel partitions required")
    counts = {
        "predictions": sum(
            n.startswith("stock-wider-study/cells/") and n.endswith(".parquet") for n in inventory
        ),
        "fit_receipts": sum(
            n.startswith("stock-wider-study/cells/") and n.endswith(".json") for n in inventory
        ),
        "opaque_native_models": sum(
            n.startswith("stock-wider-study/cells/") and n.endswith(".txt") for n in inventory
        ),
        "checkpoints": sum(n.startswith("stock-wider-study/checkpoints/") for n in inventory),
        "controls": sum(n.startswith("stock-wider-study/controls/") for n in inventory),
        "panel_partitions": sum(n.startswith("wider-panel/features/") for n in inventory),
    }
    if counts != dict(
        predictions=384,
        fit_receipts=384,
        opaque_native_models=384,
        checkpoints=384,
        controls=96,
        panel_partitions=180,
    ):
        raise ValueError("Exact full family artifact counts required")
    if sum(e["bytes"] for e in entries) != 123746295:
        raise ValueError("Exact registered original full output bytes required")
    # The generated special notebook may not be exposed in a parent mount.
    # Its numeric-v3 raw bytes have separately been authenticated on the host.
    notebook_included = False
    for path in parent.rglob("*"):
        budget()
        if path.is_symlink():
            raise ValueError("No symlinks in parent output")
        if not path.is_file():
            continue
        regular(path, parent)
        name = path.relative_to(parent).as_posix()
        if name in inventory:
            continue
        if name == "__notebook__.ipynb" and path.stat().st_size == 426014 and sha(path) == NOTEBOOK:
            inventory[name] = {"path": name, "bytes": 426014, "sha256": NOTEBOOK}
            notebook_included = True
        else:
            raise ValueError("Undeclared parent file is forbidden")

    def guard():
        budget()
        for name, entry in inventory.items():
            budget()
            path = parent / name
            regular(path, parent)
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError("Actual parent artifact byte authentication failed")

    guard()
    temporary = output / "stock-wider-audit.partial"
    archive = output / "stock-wider-audit.zip"
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as bundle:
        for name in sorted(inventory):
            budget()
            bundle.write(parent / name, arcname=name)
    guard()
    with zipfile.ZipFile(temporary) as bundle:
        if set(bundle.namelist()) != set(inventory) or len(bundle.infolist()) != len(inventory):
            raise ValueError("Exact archive member inventory required")
        for name, entry in inventory.items():
            budget()
            with bundle.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise ValueError("Full opaque archive roundtrip failed")
    os.replace(temporary, archive)
    result = {
        "schema": "quant_stock_wider_opaque_audit_transport_v1",
        "export_complete": True,
        "parent_kernel_id": PARENT_KERNEL_ID,
        "parent_version": PARENT_VERSION,
        "parent_canonical": PARENT_CANONICAL,
        "source_plan_sha256": SOURCE_PLAN,
        "workflow_manifest_sha256": WORKFLOW,
        "study_manifest_sha256": STUDY,
        "panel_manifest_sha256": PANEL,
        "members": inventory,
        "counts": counts,
        "executed_notebook_included": notebook_included,
        "separate_executed_notebook_sha256": NOTEBOOK,
        "scientific_completion_replayed": False,
        "opaque_bytes_only": True,
        "native_model_loading": False,
        "models_refitted": False,
        "raw_prices_read": False,
        "holdout_evaluated": False,
        "export_source_sha256": sha(Path(__file__)),
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive)},
    }
    guard()
    budget()
    target = output / "transport-manifest.json"
    target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    budget()
    (output / "EXPORT_COMPLETE").write_text(sha(target) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    destination = Path("/kaggle/working/stock-wider-audit-export")
    try:
        export(Path("/kaggle/input"), destination)
    except BaseException as error:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "FAILURE.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error)[:1000],
                    "export_complete": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise
