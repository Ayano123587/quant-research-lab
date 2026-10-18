"""Package only authenticated actual forecast integration inputs; no publication.

The complete upstream manifest covers 9,120 cells. This fixed 1,526-member
transport contains 1,520 forecasts, four source/support/summary artifacts, and
the original manifest/COMPLETE. It never claims all upstream bytes were copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

UPSTREAM_SHA = "ea7a801782345b38d2f3e0a1be038c5212b00263360eb78c1b910bfeba65fc8a"
SOURCE_SHA = "388af1862482f8fe33fb37085fc08b70115ce8ae165b62c23a8953dcf3d87c3f"
PROFILE = ("etf8_T63", "stock100_T252")
RECIPES = (
    "sample_ddof1",
    "ewma_halflife63",
    "ledoit_wolf",
    "oas",
    "constant_correlation_fixed0p5",
    "pca_K1",
    "pca_K3",
    "pca_K5",
)
METADATA = (
    "config.json",
    "input-support.json",
    "candidate-audit.jsonl",
    "forecast-comparison.json",
)
CONTAINER = "risk-integration-forecast-v1.bin"
TRANSPORT = "risk-integration-transport.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def safe_name(name):
    require(isinstance(name, str) and bool(name), "Regular relative member name required")
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and ":" not in name
        and path.as_posix() == name,
        "Unsafe member path",
    )
    return path


def profile_inventory(root, expected_manifest_sha256):
    """Read no forecast/proxy values: validate source pins, roles and every copied SHA."""
    root = Path(root).resolve()
    require(
        expected_manifest_sha256 == UPSTREAM_SHA,
        "Externally pinned actual forecast manifest required",
    )
    require(
        digest(root / "manifest.json") == expected_manifest_sha256,
        "Upstream forecast manifest SHA differs",
    )
    require(
        (root / "COMPLETE").read_bytes() == (UPSTREAM_SHA + "\n").encode()
        and not (root / "FAILURE.json").exists(),
        "Completed upstream forecast study required",
    )
    manifest = json.loads((root / "manifest.json").read_bytes())
    require(
        manifest["schema"] == "risk_common_probe_development_v1"
        and manifest["complete"] is True
        and manifest["actual_cells"] == manifest["expected_cells"] == 9120
        and manifest["holdout_outcomes_constructed"] is False,
        "Complete 9120-cell development-only forecast contract required",
    )
    dates = manifest["pins"]["dates"]
    require(
        len(dates) == len(set(dates)) == 95
        and dates == sorted(dates)
        and dates[0] == "2007-02-28"
        and dates[-1] == "2014-12-31"
        and all(re.fullmatch(r"20(?:0[7-9]|1[0-4])-\d\d-\d\d", d) for d in dates),
        "Exact original 95 declared development dates required",
    )
    upstream = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    require(len(upstream) == len(manifest["artifacts"]), "Duplicate upstream artifact")
    names = {
        f"cells/{day}/{panel}/{recipe}.json"
        for day in dates
        for panel in PROFILE
        for recipe in RECIPES
    } | set(METADATA)
    require(len(names) == 1524 and names <= set(upstream), "Exact integration profile missing")
    names |= {"manifest.json", "COMPLETE"}
    require(
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == names,
        "Exact fixed-profile input inventory required",
    )
    result = {}
    for name in sorted(names):
        safe_name(name)
        path = root / name
        require(
            path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "Regular authenticated source file required",
        )
        actual = digest(path)
        if name not in {"manifest.json", "COMPLETE"}:
            require(actual == upstream[name], "Input artifact SHA differs: " + name)
        result[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return result


def package_profile(root, output, *, expected_manifest_sha256):
    """Fresh immutable ZIP_STORED .bin with round-trip member authentication."""
    root, output = Path(root).resolve(), Path(output).resolve()
    inventory = profile_inventory(root, expected_manifest_sha256)
    require(
        not output.is_relative_to(root) and not root.is_relative_to(output),
        "Output must be outside authenticated inputs",
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        container = output / CONTAINER
        with zipfile.ZipFile(
            container, "x", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for name in inventory:
                archive.write(root / name, name)
        with zipfile.ZipFile(container) as archive:
            require(
                set(archive.namelist()) == set(inventory) and len(archive.infolist()) == 1526,
                "Roundtrip member inventory differs",
            )
            for item in archive.infolist():
                safe_name(item.filename)
                require(
                    item.compress_type == zipfile.ZIP_STORED
                    and not stat.S_ISLNK(item.external_attr >> 16),
                    "Stored regular members required",
                )
                with archive.open(item) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                require(
                    actual == inventory[item.filename]["sha256"]
                    and item.file_size == inventory[item.filename]["bytes"],
                    "Roundtrip SHA/size differs",
                )
        require(
            profile_inventory(root, expected_manifest_sha256) == inventory,
            "Forecast inputs changed while packaging",
        )
        result = {
            "schema": "risk_economic_forecast_transport_v1",
            "complete": True,
            "upstream_kernel_id": 139237851,
            "upstream_version": 1,
            "upstream_source_sha256": SOURCE_SHA,
            "upstream_manifest_sha256": UPSTREAM_SHA,
            "upstream_completed_cells": 9120,
            "copied_forecast_cells": 1520,
            "full_grid_bytes_independently_reviewed": False,
            "profile": list(PROFILE),
            "container": {
                "name": CONTAINER,
                "sha256": digest(container),
                "bytes": container.stat().st_size,
            },
            "members": [{"path": name, **details} for name, details in inventory.items()],
            "published": False,
        }
        (output / TRANSPORT).write_bytes(encoded(result))
        (output / "COMPLETE").write_text(digest(output / TRANSPORT) + "\n")
        return result
    except BaseException as error:
        (output / "FAILURE.json").write_bytes(
            encoded({"complete": False, "error_type": type(error).__name__})
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    package_profile(**vars(parser.parse_args()))
