"""Bundle authenticated completed ablation predictions and receipts, never models."""

import hashlib
import json
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

PARENT = "d6196c7a3b73a375c2e0fe5368f36b539e06d64989e12239d76cc1a488bd64fe"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(root, output):
    started = time.monotonic()
    if sys.platform != "linux" or Path(root).resolve() != Path("/kaggle/input"):
        raise ValueError("Only the actual cloud mount may supply completed predictions")
    matches = [p for p in Path(root).rglob("manifest.json") if sha(p) == PARENT]
    if len(matches) != 1:
        raise ValueError("One exact independently frozen complete ablation parent required")
    parent = matches[0].parent
    manifest = json.loads(matches[0].read_bytes())
    if not (
        manifest["complete"] is True
        and manifest["cells"] == 480
        and manifest["reference_reproduced_exactly"] is True
        and manifest["holdout_evaluated"] is False
        and (parent / "COMPLETE").read_text().strip() == PARENT
    ):
        raise ValueError("Full original ablation completion required")
    chosen = []
    for entry in manifest["artifacts"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe parent artifact path")
        if relative.parts[0] == "cells" and relative.suffix in (".json", ".parquet"):
            chosen.append(entry)
        elif name in ("config.json", "metrics.json", "paired-comparison.json"):
            chosen.append(entry)
    names = [entry["path"] for entry in chosen]
    if not (
        len(set(names)) == len(names) == 963
        and sum(name.endswith(".parquet") for name in names) == 480
        and sum(name.startswith("cells/") and name.endswith(".json") for name in names) == 480
    ):
        raise ValueError("Exactly 480 prediction files and 480 fit receipts required")

    def guard():
        if (
            sha(parent / "manifest.json") != PARENT
            or (parent / "COMPLETE").read_text().strip() != PARENT
        ):
            raise ValueError("Parent completion changed")
        for entry in chosen:
            path = parent / entry["path"]
            if not (
                path.resolve().is_relative_to(parent.resolve())
                and all(not stat.S_ISLNK(p.lstat().st_mode) for p in (path, *path.parents))
                and path.stat().st_size == entry["bytes"]
                and sha(path) == entry["sha256"]
            ):
                raise ValueError("Externally pinned prediction or receipt changed")
        if time.monotonic() - started >= 1800:
            raise TimeoutError("Original whole-service budget exhausted")

    guard()
    output = Path(output)
    output.mkdir(exist_ok=False)
    archive = output / "ablation-audit.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for name in sorted([*names, "manifest.json", "COMPLETE"]):
            bundle.write(parent / name, arcname=name)
    guard()
    result = {
        "complete": True,
        "parent_manifest_sha256": PARENT,
        "parent_kernel_id": 139273894,
        "parent_kernel_version": 1,
        "prediction_files": 480,
        "fit_receipts": 480,
        "model_files_included": 0,
        "predictions_recomputed": False,
        "models_refitted": False,
        "holdout_evaluated": False,
        "source_sha256": sha(Path(__file__)),
        "files": {archive.name: {"sha256": sha(archive), "bytes": archive.stat().st_size}},
    }
    guard()
    target = output / "export-manifest.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETE").write_text(sha(target) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    export("/kaggle/input", "/kaggle/working/stock-ablation-audit-export")
