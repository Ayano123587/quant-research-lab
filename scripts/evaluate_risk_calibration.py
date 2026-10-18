"""Describe authenticated full-grid variance forecasts; never fit or select models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED = "f0e4467650aaf008786acdc8be875a5bb4f2498272b597a325c74ebe65c37e69"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(source, output):
    csv = source / "all_forecast_scores.csv"
    assert sha(csv) == EXPECTED
    provenance = json.loads((source / "export-manifest.json").read_bytes())
    assert provenance["complete"] and not provenance["holdout_evaluated"]
    assert (source / "COMPLETE").read_text().strip() == sha(source / "export-manifest.json")
    frame = pd.read_csv(csv, float_precision="round_trip")
    keys = ["signal_session", "panel", "lookback", "recipe"]
    assert len(frame) == 9120 and not frame.duplicated(keys).any()
    dates = sorted(frame.signal_session.unique())
    recipes = sorted(frame.recipe.unique())
    assert len(dates) == 95 and len(recipes) == 8
    valid = frame[frame.score_status.eq("valid")].copy()
    assert len(valid) == 8808
    assert np.allclose(
        valid.qlike,
        np.log(valid.forecast_variance) + valid.observed_probe_variance / valid.forecast_variance,
        rtol=1e-12,
        atol=1e-12,
    )
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "all_forecast_scores.csv", index=False)
    summaries, intervals, bins, annual = [], [], [], []
    rng = np.random.default_rng(20261017)
    # Shared scheduled-month indices preserve gaps and pair all recipes exactly.
    draws = {}
    for length in (3, 6, 12):
        starts = rng.integers(0, len(dates), size=(2000, int(np.ceil(len(dates) / length))))
        draws[length] = ((starts[:, :, None] + np.arange(length)) % len(dates)).reshape(2000, -1)[
            :, : len(dates)
        ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, ((panel, lookback), group) in zip(
        axes.flat, frame.groupby(["panel", "lookback"]), strict=True
    ):
        wide = group.pivot(index="signal_session", columns="recipe", values="qlike").reindex(dates)
        paired = wide.notna().all(axis=1)
        assert len(group) == 760
        for recipe in recipes:
            rows = valid[
                (valid.panel == panel) & (valid.lookback == lookback) & (valid.recipe == recipe)
            ].copy()
            summaries.append(
                dict(
                    panel=panel,
                    lookback=int(lookback),
                    recipe=recipe,
                    valid_months=len(rows),
                    paired_months=int(paired.sum()),
                    mean_qlike=float(wide.loc[paired, recipe].mean()),
                    mean_squared_variance_error=float(rows.squared_variance_error.mean()),
                    observed_to_forecast_ratio=float(
                        rows.observed_probe_variance.sum() / rows.forecast_variance.sum()
                    ),
                )
            )
            difference = (wide[recipe] - wide["sample_ddof1"]).where(paired).to_numpy()
            calendar_year = pd.Index(dates).str[:4]
            for year in sorted(set(calendar_year)):
                observed = difference[calendar_year == year]
                annual.append(
                    dict(
                        panel=panel,
                        lookback=int(lookback),
                        recipe=recipe,
                        year=int(year),
                        paired_months=int(np.isfinite(observed).sum()),
                        mean_difference=float(np.nanmean(observed)),
                    )
                )
            for length, indices in draws.items():
                samples = np.nanmean(difference[indices], axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
                intervals.append(
                    dict(
                        panel=panel,
                        lookback=int(lookback),
                        recipe=recipe,
                        block_months=length,
                        paired_months=int(paired.sum()),
                        mean_difference=float(np.nanmean(difference)),
                        lower=float(low),
                        upper=float(high),
                    )
                )
            # Descriptive forecast quantiles; these bins never calibrate a model.
            rows["bin"] = pd.qcut(rows.forecast_variance, 5, labels=False, duplicates="drop")
            binned = (
                rows.groupby("bin")
                .agg(
                    predicted=("forecast_variance", "mean"),
                    observed=("observed_probe_variance", "mean"),
                    months=("qlike", "size"),
                )
                .reset_index()
            )
            for item in binned.to_dict("records"):
                bins.append(dict(panel=panel, lookback=int(lookback), recipe=recipe, **item))
            ax.plot(binned.predicted, binned.observed, "o-", ms=3, label=recipe)
        limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, limit], [0, limit], "k--", lw=0.7)
        ax.set(
            title=f"{panel}, T={lookback}",
            xlabel="Forecast daily variance",
            ylabel="Observed probe variance",
        )
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    axes.flat[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "variance_calibration.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(output / "forecast_summary.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "paired_loss_intervals.csv", index=False)
    pd.DataFrame(bins).to_csv(output / "calibration_bins.csv", index=False)
    pd.DataFrame(annual).to_csv(output / "annual_paired_losses.csv", index=False)
    manifest = dict(
        complete=True,
        source_csv_sha256=EXPECTED,
        scheduled_cells=9120,
        valid_cells=8808,
        unsupported_or_immature_cells=312,
        holdout_evaluated=False,
        models_refitted=False,
        future_prices_independently_reconstructed=False,
        uncertainty=(
            "2000 circular scheduled-month block resamples; 3/6/12 months; "
            "shared indices; missing cells retained; descriptive unadjusted intervals"
        ),
        calibration=(
            "Five within-recipe forecast quantiles; descriptive development analysis, "
            "no calibration fit"
        ),
        global_winner=None,
        outputs_sha256={p.name: sha(p) for p in sorted(output.iterdir())},
    )
    (output / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "COMPLETE").write_text(sha(output / "export-manifest.json") + "\n")
    print(
        json.dumps(dict(scheduled_cells=9120, valid_cells=len(valid), comparisons=len(intervals)))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.source, args.output)
