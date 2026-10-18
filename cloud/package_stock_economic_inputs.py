"""Package authenticated saved predictions; no native model or outcome reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST_SHA = "b46c22e5a6f46330668f6f3870c0c8cc2ddde15c283d3f9af8637f93557492d6"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    root = Path(root).resolve()
    if digest(root / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Exact completed initial stock study required")
    if (root / "COMPLETE").read_text().strip() != MANIFEST_SHA:
        raise ValueError("Actual completion marker required")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if not manifest["complete"] or manifest["final_holdout_evaluation"] or manifest["errors"]:
        raise ValueError("Complete development-only study required")
    source = {row["path"]: row for row in manifest["artifacts"]}
    predictions = {name for name in source if name.startswith("predictions/")}
    if len(predictions) != 96 or len(manifest["candidate_ids"]) != 16:
        raise ValueError("Entire sixteen-candidate monthly prediction grid required")
    result = {}
    for name in sorted(predictions | {"fit-index.json", "manifest.json", "COMPLETE"}):
        member = PurePosixPath(name)
        path = root / name
        if (
            member.is_absolute()
            or ".." in member.parts
            or "\\" in name
            or ":" in name
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("Regular confined source artifact required")
        sha, size = digest(path), path.stat().st_size
        if name in source and (sha != source[name]["sha256"] or size != source[name]["bytes"]):
            raise ValueError("Actual prediction metadata bytes differ: " + name)
        result[name] = {"sha256": sha, "bytes": size}
    fits = json.loads((root / "fit-index.json").read_bytes())
    if len(fits) != 1536 or len({(r["signal_session"], r["candidate_id"]) for r in fits}) != 1536:
        raise ValueError("All original fit receipts required")
    return result


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    before = inventory(root)
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Fresh output outside immutable inputs required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        binary = output / "stock-economic-predictions-v1.bin"
        with zipfile.ZipFile(binary, "x", compression=zipfile.ZIP_STORED) as archive:
            for name in before:
                archive.write(root / name, name)
        with zipfile.ZipFile(binary) as archive:
            if set(archive.namelist()) != set(before) or len(archive.infolist()) != 99:
                raise ValueError("Exact 99-member prediction profile required")
            for name, pin in before.items():
                with archive.open(name) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != pin["sha256"] or archive.getinfo(name).file_size != pin["bytes"]:
                    raise ValueError("Full member roundtrip differs")
        if inventory(root) != before:
            raise ValueError("Source changed while packaging")
        transport = {
            "schema": "stock_economic_prediction_transport_v1",
            "complete": True,
            "upstream_kernel_id": 139242106,
            "upstream_version": 1,
            "upstream_manifest_sha256": MANIFEST_SHA,
            "prediction_months": 96,
            "candidate_fit_receipts": 1536,
            "native_models_packaged": False,
            "outcome_values_read": False,
            "profile": "all_saved_predictions_and_fit_receipts_not_all_upstream_artifacts",
            "container": {
                "name": binary.name,
                "bytes": binary.stat().st_size,
                "sha256": digest(binary),
            },
            "members": before,
            "published": False,
        }
        path = output / "stock-prediction-transport.json"
        path.write_text(json.dumps(transport, indent=2) + "\n")
        (output / "COMPLETE").write_text(digest(path) + "\n")
        print(
            json.dumps(
                {
                    "PASS": True,
                    "members": 99,
                    "container": transport["container"],
                    "transport_sha256": digest(path),
                }
            )
        )
    except BaseException as error:
        (output / "FAILURE.json").write_text(
            json.dumps({"complete": False, "error_type": type(error).__name__})
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    package(**vars(parser.parse_args()))
