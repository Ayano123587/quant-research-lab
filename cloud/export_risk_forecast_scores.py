"""Export every authenticated completed forecast score without refitting any model."""

import csv
import hashlib
import json
import math
import time
from pathlib import Path, PurePosixPath

MANIFEST_SHA = "ea7a801782345b38d2f3e0a1be038c5212b00263360eb78c1b910bfeba65fc8a"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(root, name, expected):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe upstream artifact path")
    path = root.joinpath(*relative.parts)
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or sha(path) != expected
    ):
        raise ValueError("Upstream artifact bytes differ")
    return path


def export(inputs, output):
    if not Path("/kaggle/working").is_dir():
        raise ValueError("Metadata consolidation runs beside the actual cloud artifacts")
    started = time.monotonic()
    matches = [p for p in inputs.rglob("manifest.json") if p.is_file() and sha(p) == MANIFEST_SHA]
    if len(matches) != 1:
        raise ValueError("One exact completed upstream study required")
    root = matches[0].parent
    manifest = json.loads(matches[0].read_bytes())
    if (
        not manifest["complete"]
        or manifest["actual_cells"] != 9120
        or manifest["holdout_outcomes_constructed"]
        or (root / "COMPLETE").read_text().strip() != MANIFEST_SHA
    ):
        raise ValueError("Completed development-only full grid required")
    inventory = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    dates = manifest["pins"]["dates"]
    recipes = [recipe["recipe_id"] for recipe in manifest["pins"]["recipes"]]
    if len(set(dates)) != 95 or len(set(recipes)) != 8:
        raise ValueError("Exact 95-date eight-recipe schedule required")
    expected_grid = {
        (day, panel, lookback, recipe)
        for day in dates
        for panel in ("etf8", "stock50", "stock100", "stock150")
        for lookback in (63, 126, 252)
        for recipe in recipes
    }
    names = sorted(name for name in inventory if name.startswith("scores/"))
    if len(names) != 9120:
        raise ValueError("All 9120 score artifacts required")
    rows, identities = [], set()
    for name in names:
        if time.monotonic() - started > 1700:
            raise TimeoutError("Metadata export budget exhausted")
        score = json.loads(bound(root, name, inventory[name]).read_bytes())
        forecast_name = name.replace("scores/", "cells/", 1)
        if score["forecast_sha256"] != inventory[forecast_name]:
            raise ValueError("Score-to-forecast hash differs")
        forecast = json.loads(bound(root, forecast_name, inventory[forecast_name]).read_bytes())
        key = (
            score["signal_session"],
            score["panel_id"],
            score["observations"],
            score["recipe_id"],
        )
        forecast_key = (
            forecast["signal_session"],
            forecast["panel_id"],
            forecast["observations"],
            forecast["recipe"]["recipe_id"],
        )
        if key != forecast_key or score["proxy"]["signal_session"] != key[0]:
            raise ValueError("Forecast, score and proxy identities differ")
        if key in identities or key[0] >= "2015-01-01":
            raise ValueError("Duplicate or final-period score refused")
        identities.add(key)
        predicted, proxy = forecast["variance"], score["proxy"]
        valid = score["score"]["status"] == "valid"
        if valid:
            observed = proxy["value"]
            if not (
                forecast["status"] == proxy["status"] == "valid"
                and math.isfinite(predicted)
                and predicted > 0
                and math.isfinite(observed)
                and observed >= 0
                and proxy["observations"] == 21
                and proxy["planned_available_at"][:10] <= "2014-12-31"
            ):
                raise ValueError("Valid full-window mature probe required")
            loss = math.log(predicted) + observed / predicted
            squared = (predicted - observed) ** 2
            if not (
                math.isclose(loss, score["score"]["qlike"], rel_tol=1e-13, abs_tol=1e-13)
                and math.isclose(
                    squared, score["score"]["squared_variance_error"], rel_tol=1e-13, abs_tol=1e-25
                )
            ):
                raise ValueError("Saved forecast loss does not replay")
        rows.append(
            {
                "signal_session": key[0],
                "panel": key[1],
                "lookback": key[2],
                "recipe": key[3],
                "forecast_status": forecast["status"],
                "score_status": score["score"]["status"],
                "forecast_variance": predicted,
                "observed_probe_variance": proxy["value"] if valid else None,
                "qlike": score["score"]["qlike"] if valid else None,
                "squared_variance_error": score["score"]["squared_variance_error"]
                if valid
                else None,
                "proxy_end_session": proxy["end_session"],
                "proxy_available_at": proxy["planned_available_at"],
                "forecast_sha256": inventory[forecast_name],
                "score_sha256": inventory[name],
            }
        )
    if identities != expected_grid:
        raise ValueError("Export does not cover the complete declared grid")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "all_forecast_scores.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "complete": True,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_kernel_id": 139237851,
        "source_kernel_version": 1,
        "scheduled_cells": len(rows),
        "valid_scored_cells": sum(r["qlike"] is not None for r in rows),
        "all_forecast_and_score_artifacts_authenticated": 18240,
        "saved_scalar_losses_replayed": True,
        "future_prices_independently_reconstructed": False,
        "models_refitted": False,
        "holdout_evaluated": False,
        "source_sha256": sha(Path(__file__)),
        "outputs_sha256": {path.name: sha(path)},
    }
    (output / "export-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "COMPLETE").write_text(sha(output / "export-manifest.json") + "\n")


if __name__ == "__main__":
    working = Path("/kaggle/working")
    if not working.is_dir():
        raise ValueError("Cloud-only metadata consolidation")
    status = working / "risk-score-export-status.json"
    status.write_text(json.dumps({"complete": False, "phase": "authenticate_upstream"}))
    try:
        export(Path("/kaggle/input"), working / "risk-full-score-export")
    except Exception as error:
        status.write_text(json.dumps({"complete": False, "error_type": type(error).__name__}))
        raise
    else:
        status.write_text(json.dumps({"complete": True, "models_refitted": False}))
