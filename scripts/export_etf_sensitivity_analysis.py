"""Export the seven authenticated development sensitivity paths for analysis."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, output):
    source, output = Path(source), Path(output)
    study = source / "study"
    manifest_path = study / "manifest.json"
    assert digest(manifest_path) == (
        "44655a41e80bcf1f6f81d598b747e77387e7d1a16a856918e74a3a553388d6ec"
    )
    manifest = json.loads(manifest_path.read_text())
    audit_path = source / "independent-seven-ledger-replay-v1.json"
    audit = json.loads(audit_path.read_text())
    assert audit["PASS"] and audit["manifest_sha256"] == digest(manifest_path)
    assert audit["measured_rows"] == 13825 and audit["maximum_metric_error"] == 0
    assert audit["reference_exact_float64_cells"] == 31600
    for name, pin in manifest["artifacts"].items():
        path = study / name
        assert path.stat().st_size == pin["bytes"] and digest(path) == pin["sha256"]
    frames = []
    for arm in sorted(audit["metrics"]):
        rows = [
            json.loads(line)
            for line in (study / arm / "money-ledger.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1976 and rows[0]["state"]["nav"] == 100000
        nav = np.array([row["state"]["nav"] for row in rows])
        returns = nav[1:] / nav[:-1] - 1
        metrics = audit["metrics"][arm]
        np.testing.assert_allclose(
            returns.mean() / returns.std(ddof=1) * np.sqrt(252),
            metrics["annual_sharpe_zero_cash_reference"],
            rtol=0,
            atol=1e-12,
        )
        frames.append(
            pd.DataFrame(
                [
                    dict(
                        arm=arm,
                        session=row["session"],
                        close_nav=row["state"]["nav"],
                        cash=row["state"]["cash"],
                        fee=row["execution"]["fee"],
                        turnover=row["execution"]["full_turnover"],
                        daily_return=ret,
                    )
                    for row, ret in zip(rows[1:], returns, strict=True)
                ]
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(frames, ignore_index=True).to_parquet(output / "daily-paths.parquet", index=False)
    pd.DataFrame.from_dict(audit["metrics"], orient="index").rename_axis("arm").to_csv(
        output / "path-metrics.csv"
    )
    result = dict(
        schema="etf_sensitivity_portable_analysis_v1",
        complete=True,
        holdout_evaluated=False,
        fee_bps=10,
        paths=7,
        sessions_per_path=1975,
        source_manifest_sha256=digest(manifest_path),
        audit_sha256=digest(audit_path),
        scope=audit["scope"],
        outputs_sha256={p.name: digest(p) for p in output.iterdir()},
    )
    (output / "export-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETE").write_text(digest(output / "export-manifest.json") + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    export(**vars(parser.parse_args()))
