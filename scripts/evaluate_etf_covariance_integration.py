"""Connect the registered covariance choice to the matched ETF economic paths."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate(report):
    path = report / "export-manifest.json"
    metadata = json.loads(path.read_text())
    assert metadata["complete"] and not metadata["holdout_evaluated"]
    assert (report / "COMPLETE").read_text().strip() == digest(path)
    for name, pin in metadata["outputs_sha256"].items():
        assert digest(report / name) == pin
    return metadata


def evaluate(source):
    root = Path(__file__).resolve().parents[1]
    forecasts = root / "reports/risk_forecast_v1"
    calibration = root / "reports/risk_forecast_calibration_v1"
    sensitivity = root / "reports/etf_sensitivity_v3"
    for report in (forecasts, calibration, sensitivity):
        authenticate(report)
    comparison = json.loads((forecasts / "forecast-comparison.json").read_text())
    cell = next(
        p for p in comparison["panels"] if p["panel_id"] == "etf8" and p["observations"] == 63
    )
    scores = pd.read_csv(calibration / "all_forecast_scores.csv", float_precision="round_trip")
    scores = scores[(scores.panel == "etf8") & (scores.lookback == 63)]
    paired = (
        scores[scores.signal_session.isin(cell["common_dates"])]
        .pivot(index="signal_session", columns="recipe", values="qlike")
        .sort_index()
    )
    assert paired.shape == (94, 8) and paired.notna().all().all()
    means = paired.mean()
    for name, value in cell["mean_qlike_common"].items():
        np.testing.assert_allclose(means[name], value, rtol=0, atol=1e-12)
    selected = min(means.index, key=lambda name: (means[name], name))
    assert selected == cell["selected_recipe_id"] == "ewma_halflife63"
    study = Path(source) / "study"
    assert digest(study / "manifest.json") == (
        "44655a41e80bcf1f6f81d598b747e77387e7d1a16a856918e74a3a553388d6ec"
    )
    manifest = json.loads((study / "manifest.json").read_text())
    names = ("reference_sample_target10", "covariance_ewma63_only")
    targets = []
    for arm in names:
        name = arm + "/monthly-targets.json"
        assert digest(study / name) == manifest["artifacts"][name]["sha256"]
        targets.append(json.loads((study / name).read_text()))
    source_identity = json.loads((forecasts / "config.json").read_text())["pins"][
        "etf_source_identity"
    ]
    matched = []
    for reference, candidate in zip(*targets, strict=True):
        for key in (
            "active",
            "strategic_weights",
            "input_row_identity_sha256",
            "max_input_available_at",
            "source_identity",
            "input_rows",
        ):
            assert reference[key] == candidate[key]
        assert reference["source_identity"] == source_identity
        for key in ("signal_at", "intended_execution_at", "action"):
            assert reference["decision"][key] == candidate["decision"][key]
        assert (
            reference["scenario"]["annual_target"] == candidate["scenario"]["annual_target"] == 0.1
        )
        assert reference["strategic_weights"] == [0.125] * 8
        matched.append(reference["decision"]["signal_at"])
    assert len(matched) == 95
    daily = pd.read_parquet(sensitivity / "daily-paths.parquet")
    returns = (
        daily[daily.arm.isin(names)]
        .pivot(index="session", columns="arm", values="daily_return")
        .sort_index()
    )
    assert returns.shape == (1975, 2) and returns.notna().all().all()
    metrics = pd.read_csv(
        sensitivity / "path-metrics.csv", index_col="arm", float_precision="round_trip"
    ).loc[list(names)]
    output = root / "reports/etf_covariance_integration_v1"
    output.mkdir(exist_ok=False)
    paired.to_csv(output / "paired-forecast-losses.csv")
    metrics.to_csv(output / "matched-path-metrics.csv")
    result = dict(
        complete=True,
        holdout_evaluated=False,
        strategy_rerun=False,
        selection_rule="minimum mean QLIKE on all94 common mature ETF8 T63 dates; fixed id tie",
        selected_covariance_recipe=selected,
        covariance_reference="sample_ddof1",
        matched_signal_targets=95,
        matched_economic_sessions=1975,
        fee_bps=10,
        selection_based_on_portfolio_scores=False,
        forecast_source_manifest_sha256=authenticate(forecasts)["source_manifest_sha256"],
        sensitivity_source_manifest_sha256=digest(study / "manifest.json"),
        forecast_mean_qlike=means.to_dict(),
        annualized_paired_arithmetic_mean_return_difference=float(
            (returns[names[1]] - returns[names[0]]).mean() * 252
        ),
        interpretation=(
            "The forecast-selected EWMA has lower development Sharpe than sample. "
            "Forecast selection and economic evaluation answer different questions; "
            "this is reused-development integration, not an independent test."
        ),
        outputs_sha256={p.name: digest(p) for p in output.iterdir()},
    )
    (output / "export-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETE").write_text(digest(output / "export-manifest.json") + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    evaluate(**vars(parser.parse_args()))
