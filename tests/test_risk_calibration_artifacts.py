"""Replay the complete real development export without fitting any models."""

from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1] / "reports/risk_forecast_calibration_v1"


def test_full_grid_losses_and_common_probe_identity():
    rows = pd.read_csv(REPORT / "all_forecast_scores.csv", float_precision="round_trip")
    assert len(rows) == 9120
    assert not rows.duplicated(["signal_session", "panel", "lookback", "recipe"]).any()
    assert rows.signal_session.nunique() == 95
    assert rows.recipe.nunique() == 8
    assert set(rows.lookback) == {63, 126, 252}
    assert set(rows.panel) == {"etf8", "stock50", "stock100", "stock150"}
    assert rows.groupby(["signal_session", "panel", "lookback"]).size().eq(8).all()
    valid = rows[rows.score_status.eq("valid")]
    assert len(valid) == 8808
    probe = valid.groupby(["signal_session", "panel", "lookback"])
    assert probe.observed_probe_variance.nunique().eq(1).all()
    assert probe.proxy_end_session.nunique().eq(1).all()
    assert probe.proxy_available_at.nunique().eq(1).all()
    np.testing.assert_allclose(
        valid.qlike,
        np.log(valid.forecast_variance) + valid.observed_probe_variance / valid.forecast_variance,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        valid.squared_variance_error,
        (valid.observed_probe_variance - valid.forecast_variance) ** 2,
        rtol=1e-12,
        atol=1e-25,
    )


def test_all_paired_means_and_interval_comparisons_use_same_dates():
    rows = pd.read_csv(REPORT / "all_forecast_scores.csv", float_precision="round_trip")
    summary = pd.read_csv(REPORT / "forecast_summary.csv", float_precision="round_trip")
    intervals = pd.read_csv(REPORT / "paired_loss_intervals.csv", float_precision="round_trip")
    assert len(summary) == 96 and len(intervals) == 288
    for (panel, lookback), group in rows.groupby(["panel", "lookback"]):
        wide = group.pivot(index="signal_session", columns="recipe", values="qlike")
        paired = wide.dropna()
        observed = summary[(summary.panel == panel) & (summary.lookback == lookback)]
        assert observed.paired_months.eq(len(paired)).all()
        for item in observed.itertuples():
            np.testing.assert_allclose(item.mean_qlike, paired[item.recipe].mean(), atol=1e-12)
            selected = intervals[
                (intervals.panel == panel)
                & (intervals.lookback == lookback)
                & (intervals.recipe == item.recipe)
            ]
            assert set(selected.block_months) == {3, 6, 12}
            np.testing.assert_allclose(
                selected.mean_difference,
                (paired[item.recipe] - paired.sample_ddof1).mean(),
                atol=1e-12,
            )
            assert (selected.lower <= selected.upper).all()
    baseline = intervals[intervals.recipe.eq("sample_ddof1")]
    assert baseline[["mean_difference", "lower", "upper"]].eq(0).all().all()


def test_saved_uncertainty_replays_observed_month_blocks_with_missing_support():
    rows = pd.read_csv(REPORT / "all_forecast_scores.csv", float_precision="round_trip")
    intervals = pd.read_csv(REPORT / "paired_loss_intervals.csv", float_precision="round_trip")
    dates = sorted(rows.signal_session.unique())
    rng = np.random.default_rng(20261017)
    for length in (3, 6, 12):
        starts = rng.integers(0, 95, size=(2000, int(np.ceil(95 / length))))
        indices = ((starts[:, :, None] + np.arange(length)) % 95).reshape(2000, -1)[:, :95]
        for (panel, lookback), group in rows.groupby(["panel", "lookback"]):
            wide = group.pivot(index="signal_session", columns="recipe", values="qlike")
            wide = wide.reindex(dates)
            paired = wide.notna().all(axis=1)
            for recipe in sorted(rows.recipe.unique()):
                differences = (wide[recipe] - wide.sample_ddof1).where(paired).to_numpy()
                # Missing support retains its scheduled calendar position in every block.
                draws = differences[indices]
                estimates = np.nansum(draws, axis=1) / np.isfinite(draws).sum(axis=1)
                expected = np.quantile(estimates, [0.025, 0.975])
                saved = intervals[
                    (intervals.panel == panel)
                    & (intervals.lookback == lookback)
                    & (intervals.recipe == recipe)
                    & (intervals.block_months == length)
                ]
                assert len(saved) == 1
                np.testing.assert_allclose(saved[["lower", "upper"]].iloc[0], expected, atol=1e-12)


def test_calendar_year_slices_cover_every_paired_month():
    annual = pd.read_csv(REPORT / "annual_paired_losses.csv", float_precision="round_trip")
    intervals = pd.read_csv(REPORT / "paired_loss_intervals.csv", float_precision="round_trip")
    assert len(annual) == 768
    assert set(annual.year) == set(range(2007, 2015))
    for (panel, lookback, recipe), group in annual.groupby(["panel", "lookback", "recipe"]):
        saved = intervals[
            (intervals.panel == panel)
            & (intervals.lookback == lookback)
            & (intervals.recipe == recipe)
            & intervals.block_months.eq(6)
        ].iloc[0]
        assert group.paired_months.sum() == saved.paired_months
        np.testing.assert_allclose(
            np.average(group.mean_difference, weights=group.paired_months),
            saved.mean_difference,
            atol=1e-12,
        )
