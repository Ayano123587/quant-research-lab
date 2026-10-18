"""Descriptive paired time-block intervals from all audited ETF development paths."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/risk_economic_development_v1"
    manifest_path = source / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert (source / "COMPLETE").read_text().strip() == digest(manifest_path)
    for name, pin in manifest["outputs_sha256"].items():
        assert digest(source / name) == pin
    daily = pd.read_parquet(source / "daily-paths.parquet")
    nav = daily.pivot(index="session", columns="path_id", values="close_nav").sort_index()
    assert nav.shape == (1975, 104) and nav.notna().all().all()
    values = nav.to_numpy()
    returns = values / np.vstack([np.full((1, 104), 100000.0), values[:-1]]) - 1
    differences = returns.copy()
    for j, name in enumerate(nav.columns):
        fee = name.split("/")[1]
        baseline = nav.columns.get_loc(f"etf8/{fee}/equal_weight")
        differences[:, j] -= returns[:, baseline]
    n = len(values)
    # One common set of circular observed-date blocks for all arms and fees.
    doubled = np.concatenate([differences, differences], axis=0)
    cumulative = np.vstack([np.zeros((1, 104)), doubled.cumsum(axis=0)])
    records = []
    for block in [21, 63, 126]:
        rng = np.random.default_rng(20261018 + block)
        number = int(np.ceil(n / block))
        lengths = np.full(number, block)
        lengths[-1] = n - block * (number - 1)
        samples = []
        for _ in range(20):
            starts = rng.integers(0, n, size=(100, number))
            sums = cumulative[starts + lengths] - cumulative[starts]
            samples.append(sums.sum(axis=1) / n * 252)
        sampled = np.concatenate(samples)
        assert sampled.shape == (2000, 104)
        low, high = np.quantile(sampled, [0.025, 0.975], axis=0)
        for j, name in enumerate(nav.columns):
            records.append(
                {
                    "path_id": name,
                    "block_sessions": block,
                    "paired_sessions": n,
                    "resamples": 2000,
                    "seed": 20261018 + block,
                    "annualized_mean_return_difference": differences[:, j].mean() * 252,
                    "lower": low[j],
                    "upper": high[j],
                }
            )
    output = root / "reports/risk_economic_uncertainty_v1"
    output.mkdir(exist_ok=False)
    pd.DataFrame(records).to_csv(output / "paired-mean-return-intervals.csv", index=False)
    metadata = {
        "complete": True,
        "holdout_evaluated": False,
        "source_export_sha256": digest(manifest_path),
        "comparisons": 312,
        "reference": "equal_weight at same fee",
        "interval_target": "252 times paired daily arithmetic mean return difference",
        "interpretation": (
            "Descriptive reused-development circular block intervals; "
            "no multiplicity adjustment or winner selection. Not CAGR or Sharpe intervals."
        ),
        "market_data_generated": False,
        "outputs_sha256": {
            "paired-mean-return-intervals.csv": digest(output / "paired-mean-return-intervals.csv")
        },
    }
    (output / "export-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "COMPLETE").write_text(digest(output / "export-manifest.json") + "\n")
    print(output)


if __name__ == "__main__":
    main()

