"""Development comparison figures from audited, observed monthly IC results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def export_development_metrics(study, expected_manifest_sha256, intervals_path, output):
    """Authenticate completed cloud metrics before exporting derived results."""
    study, output = Path(study), Path(output)
    manifest_raw = (study / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest_raw).hexdigest()
    if digest != expected_manifest_sha256 or (study / "COMPLETE").read_text().strip() != digest:
        raise ValueError("Pinned completed stock study required")
    manifest = json.loads(manifest_raw)
    if not manifest["complete"] or manifest["final_holdout_evaluation"]:
        raise ValueError("Completed development study without holdout required")
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    copied = {}
    for name in ("metrics.json", "development-selection.json", "config.json"):
        raw = (study / name).read_bytes()
        pin = artifacts[name]
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("Cloud metric artifact differs: " + name)
        copied[name] = raw
    metrics = json.loads(copied["metrics.json"])
    selection = json.loads(copied["development-selection.json"])
    if len(metrics) != 1536 or len(selection["common_dates"]) != 94:
        raise ValueError("Entire registered initial development grid required")
    intervals_raw = Path(intervals_path).read_bytes()
    intervals = json.loads(intervals_raw)
    if (
        intervals["authenticated_metrics_sha256"]
        != hashlib.sha256(copied["metrics.json"]).hexdigest()
    ):
        raise ValueError("Uncertainty must describe this exact metric file")
    output.mkdir(parents=True, exist_ok=False)
    copied["uncertainty.json"] = intervals_raw
    for name, raw in copied.items():
        (output / name).write_bytes(raw)
    provenance = {
        "complete": True,
        "cloud_study_manifest_sha256": digest,
        "metric_cells": len(metrics),
        "common_mature_months": 94,
        "holdout_evaluated": False,
        "model_fit_here": False,
        "outputs_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in copied.items()},
    }
    raw = (json.dumps(provenance, indent=2) + "\n").encode()
    (output / "export-manifest.json").write_bytes(raw)
    (output / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    return provenance


def plot_ic_comparison(intervals_path, output):
    """Plot every supplied candidate; never filter to a favorable confidence band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(intervals_path).read_bytes()
    record = json.loads(source)
    if (
        record.get("PASS") is not True
        or record.get("holdout_read") is not False
        or record.get("selection_adjusted") is not False
    ):
        raise ValueError("Audited development-only exploratory uncertainty required")
    reports = record["intervals"]
    names = [row["candidate_id"] for row in reports[0]["candidates"]]
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Unique complete candidate comparison required")
    for report in reports:
        if (
            report["months"] != reports[0]["months"]
            or report["input_sha256"] != reports[0]["input_sha256"]
            or [row["candidate_id"] for row in report["candidates"]] != names
        ):
            raise ValueError("All block lengths must use the identical candidate/month cohort")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def short(name):
        return (
            name.replace("control_", "").replace("regression_", "").replace("iterations", "trees")
        )

    rows = sorted(reports[0]["candidates"], key=lambda row: row["mean_monthly_IC"])
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    ax.barh(
        [short(row["candidate_id"]) for row in rows],
        [row["mean_monthly_IC"] for row in rows],
        color=[
            "#386b9b" if row["candidate_id"].startswith("control_") else "#bd7628" for row in rows
        ],
    )
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set(
        xlabel="Mean monthly Spearman rank correlation",
        title=(
            f"Stock development: all {len(names)} candidates, {reports[0]['months']} mature months"
        ),
    )
    ax.grid(axis="x", alpha=0.2)
    artifacts = []
    for extension in ("png", "svg"):
        path = output / ("mean_rank_ic." + extension)
        fig.savefig(path, dpi=160)
        artifacts.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(reports), figsize=(18, 9), sharey=True, layout="constrained")
    order = [row["candidate_id"] for row in rows]
    for ax, report in zip(axes, reports, strict=True):
        indexed = {row["candidate_id"]: row for row in report["candidates"]}
        for index, name in enumerate(order):
            row = indexed[name]
            lower, upper = row["paired_difference_interval"]
            ax.plot([lower, upper], [index, index], color="#386b9b", linewidth=2)
            ax.scatter([row["mean_IC_difference"]], [index], color="#bd7628", s=18, zorder=3)
        ax.axvline(0, color="0.35", linewidth=0.8)
        ax.set(
            title=f"{report['block_months']}-month blocks",
            xlabel="IC difference versus fixed composite",
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(order)), [short(name) for name in order])
    fig.suptitle("Paired 95% exploratory intervals; candidate search is not adjusted", fontsize=14)
    for extension in ("png", "svg"):
        path = output / ("paired_ic_uncertainty." + extension)
        fig.savefig(path, dpi=160)
        artifacts.append(path)
    plt.close(fig)
    provenance = {
        "input_file_sha256": hashlib.sha256(source).hexdigest(),
        "observed_IC_input_sha256": reports[0]["input_sha256"],
        "candidates": len(names),
        "months": reports[0]["months"],
        "holdout_read": False,
        "selection_adjusted": False,
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts
        },
    }
    (output / "figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Verify and export a completed development study")
    export.add_argument("--study", required=True)
    export.add_argument("--expected-manifest-sha256", required=True)
    export.add_argument("--intervals-path", required=True)
    export.add_argument("--output", required=True)
    plots = commands.add_parser("plot", help="Plot every audited candidate and block-length choice")
    plots.add_argument("--intervals-path", required=True)
    plots.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = (
        export_development_metrics(**args) if command == "export" else plot_ic_comparison(**args)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
