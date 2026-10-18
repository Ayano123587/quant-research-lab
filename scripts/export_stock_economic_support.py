"""Publish complete coverage of the actual 64-path stock execution attempt."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, output):
    source, output = Path(source), Path(output)
    parent = source / "study/manifest.json"
    expected = "be9bbe820ffa16ec917434f8f6bb6ee18533425ec330b79f05a2e76f3ee607a7"
    assert sha(parent) == expected
    study = json.loads(parent.read_bytes())
    audit_path = source / "root-independent-stock64-source-failures-v1.json"
    audit = json.loads(audit_path.read_bytes())
    assert audit["PASS"] and audit["parent_manifest_sha256"] == expected
    assert audit["attempted_paths"] == audit["unsupported_paths"] == 64
    assert audit["complete_paths"] == 0 and not audit["aggregate_performance_authorized"]
    assert study["complete"] is False and study["metrics"] is None
    assert study["status"] == "unsupported_full_comparison" and len(study["paths"]) == 64
    accounting_path = source / "root-independent-stock64-saved-accounting-v1.json"
    accounting = json.loads(accounting_path.read_bytes())
    assert accounting["PASS"] and accounting["parent_manifest_sha256"] == expected
    assert accounting["inventory_files_verified"] == 128 and accounting["paths_audited"] == 64
    assert accounting["complete_performance_paths"] == 0
    assert accounting["aggregate_performance_authorized"] is False
    rows = []
    for path in study["paths"]:
        assert path["economic_support"] == "UNSUPPORTED_VALID_PREFIX"
        assert path["complete_metrics"] is None
        failure = path["failure"]
        rows.append({
            "candidate": path["candidate_id"],
            "fee_bps": path["fee_bps"],
            "status": path["economic_support"],
            "valid_measured_sessions": path["valid_measured_sessions"],
            "stopped_session": failure["session"],
            "stopped_event": failure["event"],
            "unresolved_assets": ";".join(failure["unresolved_assets"]),
        })
    assert len({(r["candidate"], r["fee_bps"]) for r in rows}) == 64
    output.mkdir(parents=True, exist_ok=False)
    with (output / "coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accounting_summary = {
        key: accounting[key]
        for key in (
            "actual_utc", "parent_manifest_sha256", "inventory_files_verified", "paths_audited",
            "saved_daily_rows", "maximum_money_identity_error", "complete_performance_paths",
            "aggregate_performance_authorized", "scope",
        )
    }
    accounting_summary["audit_sha256"] = sha(accounting_path)
    (output / "accounting-audit.json").write_text(
        json.dumps(accounting_summary, indent=2) + "\n"
    )
    summary = {
        "complete": True,
        "scope": "complete_coverage_of_all64_attempted_paths_not_complete_performance",
        "source_kernel_id": 139269458,
        "source_kernel_version": 1,
        "source_manifest_sha256": expected,
        "source_failure_audit_sha256": sha(audit_path),
        "attempted_paths": 64,
        "complete_paths": 0,
        "unsupported_paths": 64,
        "aggregate_performance_authorized": False,
        "holdout_evaluated": False,
        "source_check_scope": audit["scope"],
        "unresolved_exposure_checks": audit["unresolved_exposure_checks"],
        "observed_unique_asset_dates": audit["observed_unique_asset_dates"],
        "reason_counts": audit["reason_counts"],
        "outputs_sha256": {
            name: sha(output / name) for name in ("coverage.csv", "accounting-audit.json")
        },
    }
    (output / "export-manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "COMPLETE").write_text(sha(output / "export-manifest.json") + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    export(**vars(parser.parse_args()))
