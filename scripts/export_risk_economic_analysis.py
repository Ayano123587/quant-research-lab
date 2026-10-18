"""Export authenticated full ETF risk-economic paths and stock support records."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, output):
    source, output = Path(source), Path(output)
    study = source / "study"
    audit_path = source / "independent-full-ledger-replay-v1.json"
    audit = json.loads(audit_path.read_text())
    review = json.loads((source / "root-full-ledger-review.json").read_text())
    assert review["pass"] and review["full_replay_receipt_sha256"] == digest(audit_path)
    assert audit["PASS"] and audit["ledger_rows"] == 205400
    manifest = json.loads((study / "manifest.json").read_text())
    assert (
        digest(study / "manifest.json")
        == "077d7aa85065d31172194468f0781fe11de71e2e1d084504b612b9c3e0dfae04"
    )
    pins = {row["path"]: row["sha256"] for row in manifest["artifacts"]}
    output.mkdir(parents=True, exist_ok=False)
    paths, metrics = [], []
    for path_id, receipt in sorted(audit["paths"].items()):
        rel = "paths/" + path_id + "/ledger.parquet"
        path = study / rel
        assert digest(path) == pins[rel]
        frame = pd.read_parquet(
            path, columns=["session", "close_nav", "cash", "fee", "full_turnover"]
        )
        assert len(frame) == receipt["rows"] == 1975
        frame.insert(0, "path_id", path_id)
        paths.append(frame)
        panel, fee, arm = path_id.split("/")
        metrics.append(
            {
                "path_id": path_id,
                "panel": panel,
                "fee_bps": int(fee[4:]),
                "arm": arm,
                **receipt["replayed_metrics"],
            }
        )
    daily = pd.concat(paths, ignore_index=True)
    assert len(daily) == 205400 and len(metrics) == 104
    daily.to_parquet(output / "daily-paths.parquet", index=False)
    pd.DataFrame(metrics).to_csv(output / "path-metrics.csv", index=False)
    support = []
    for row in audit["unsupported"]:
        failure = row["failure"]
        support.append(
            {
                "path_id": row["path_id"],
                "status": "unsupported",
                "failure_session": failure["session"],
                "reason": failure["reason"],
                "unresolved_assets": failure["unresolved_assets"],
            }
        )
    assert len(support) == 104
    (output / "stock-support.json").write_text(json.dumps(support, indent=2) + "\n")
    files = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    result = {
        "schema": "risk_economic_portable_analysis_v1",
        "complete": True,
        "scientific_family_complete": False,
        "holdout_evaluated": False,
        "source_manifest_sha256": digest(study / "manifest.json"),
        "source_audit_sha256": digest(audit_path),
        "complete_etf_paths": 104,
        "unsupported_stock_paths": 104,
        "daily_rows": 205400,
        "outputs_sha256": files,
    }
    (output / "export-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETE").write_text(digest(output / "export-manifest.json") + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    export(**vars(parser.parse_args()))
