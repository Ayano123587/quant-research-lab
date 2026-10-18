"""Compare all completed ETF controls using matched observed return blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.evaluation import uncertainty

POLICIES = ("strategic_equal", "trend_252", "volatility_63", "trend_252_volatility_63")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(study, manifest_sha256, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise ValueError("Preserve previous uncertainty results; use a new destination")
    if file_hash(study / "manifest.json") != manifest_sha256:
        raise ValueError("Externally authenticated completed study manifest required")
    manifest = json.loads((study / "manifest.json").read_bytes())
    if manifest["complete"] is not True or manifest["status"] != "complete":
        raise ValueError("A scientifically complete study is required")
    for name, expected in manifest["outputs_sha256"].items():
        if Path(name).name != name or file_hash(study / name) != expected:
            raise ValueError("Completed study output differs")
    contract = manifest["contract"]
    if (
        set(contract["policies"]) != set(POLICIES)
        or contract["phase"] != "development_only_not_final_selection_or_test"
        or contract["measurement_start"] != "2007-03-01"
        or contract["end_close"] != "2014-12-31"
        or contract["initial_cash"] != 100_000
        or contract["fee_bps_per_traded_side"] != 10
    ):
        raise ValueError("Full declared development comparison required")
    frames = {name: pd.read_parquet(study / f"{name}.parquet") for name in POLICIES}
    dates = frames[POLICIES[0]]["session"]
    if len(dates) != 1975 or any(not frame["session"].equals(dates) for frame in frames.values()):
        raise ValueError("All four controls require the entire same 1,975-session cohort")
    returns = {}
    for name, frame in frames.items():
        nav = frame["close_nav"].to_numpy(dtype=np.float64)
        if not np.isfinite(nav).all() or np.any(nav <= 0):
            raise ValueError("Positive finite observed portfolio NAV required")
        returns[name] = nav / np.r_[contract["initial_cash"], nav[:-1]] - 1
    results = {
        name: uncertainty.paired_block_intervals(
            dates.tolist(), returns[name], returns[POLICIES[0]]
        )
        for name in POLICIES[1:]
    }
    record = {
        "schema": "etf_control_uncertainty_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "study_manifest_sha256": manifest_sha256,
        "source_sha256": {
            "evaluation/uncertainty.py": file_hash(uncertainty.__file__),
            "scripts/evaluate_etf_uncertainty.py": file_hash(__file__),
        },
        "scope": "full_development_observed_after_control_scores_not_preregistered_inference",
        "reference": POLICIES[0],
        "comparisons": results,
        "interpretation": {
            "selection_adjusted": False,
            "generated_market_data": False,
            "fees": "original_executed_ledger_fees_preserved_no_reexecution",
            "block_sensitivity": "all_21_63_126_session_intervals_reported_without_selection",
            "limits": "Historical resampling uncertainty is not a guarantee of future performance.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create avoids overwriting another research result.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.study, args.manifest_sha256, args.output)
    for name, comparison in result["comparisons"].items():
        print(
            json.dumps(
                {
                    "policy": name,
                    "reference": result["reference"],
                    "observed_sharpe_difference": comparison["observed_annual_sharpe_difference"],
                    "block_sensitivity": comparison["block_sensitivity"],
                }
            )
        )
