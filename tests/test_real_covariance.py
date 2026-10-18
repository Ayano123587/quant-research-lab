"""Estimator checks on authenticated ETF prices, not simulated matrices."""

import hashlib
import math
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.returns import trailing_returns
from quantlab.risk.covariance import ReturnWindow, estimate_covariance

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def real_windows():
    source = Path(
        os.environ.get(
            "QUANTLAB_ARCHIVE", str(Path(__file__).resolve().parents[1] / "data/raw/archive.zip")
        )
    )
    if not source.exists():
        pytest.skip("Authenticated supplied archive required")
    schedule = XNYSSchedule(date(2015, 1, 1), date(2018, 1, 31))
    symbols = ("SPY", "IEF", "EFA", "TLT", "GLD", "DBC")
    end = date(2017, 11, 10)
    with MarketArchive(source, SUPPLIED_ARCHIVE_SHA256) as archive:
        bars = [archive.bars("etf", symbol) for symbol in symbols]
        intervals = [
            trailing_returns(rows, schedule, end, 126, archive_endpoint=end) for rows in bars
        ]
    assert all(
        tuple(x.start_at for x in col) == tuple(x.start_at for x in intervals[0])
        for col in intervals
    )
    windows = {}
    for count in (2, 63, 126):
        selected = [col[-count:] for col in intervals]
        values = np.array([[col[i].return_value for col in selected] for i in range(count)])
        row_hash = hashlib.sha256(
            repr([[x.start_row for x in col] + [col[-1].end_row] for col in selected]).encode()
        ).hexdigest()
        times = selected[0]
        windows[count] = ReturnWindow(
            values,
            tuple(rows[0].asset_id for rows in bars),
            tuple(x.start_at for x in times),
            tuple(x.end_at for x in times),
            tuple(schedule.session(x.end_at.date()).decision_at for x in times),
            schedule.session(end).decision_at,
            SUPPLIED_ARCHIVE_SHA256,
            "source_close_to_close_adjustment_unverified",
            "fixed_six_ETF_software_check",
            "all_requested_intervals_valid",
            {
                "source_rows_sha256": row_hash,
                "schedule_sha256": schedule.provenance["schedule_sha256"],
            },
        )
    return windows


@pytest.mark.parametrize("count", [63, 126])
def test_sample_hand_reference_daily_units_and_order(real_windows, count):
    window = real_windows[count]
    result = estimate_covariance(window)
    values = window.values
    for i in range(values.shape[1]):
        mean_i = math.fsum(float(v) for v in values[:, i]) / count
        for j in range(values.shape[1]):
            mean_j = math.fsum(float(v) for v in values[:, j]) / count
            reference = math.fsum(
                (float(row[i]) - mean_i) * (float(row[j]) - mean_j) for row in values
            ) / (count - 1)
            assert result.matrix[i, j] == pytest.approx(reference, rel=1e-12, abs=1e-18)
    assert result.units == "decimal_return_squared_per_session"
    assert result.asset_ids == window.asset_ids
    assert result.window_hash == window.window_hash
    assert result.max_available_at <= result.signal_at
    assert result.training_end == window.interval_end[-1]
    assert result.source_provenance["source_snapshot_id"] == SUPPLIED_ARCHIVE_SHA256
    assert result.source_provenance["return_convention"] == window.return_convention
    assert not result.matrix.flags.writeable
    mle = estimate_covariance(window, "sample_mle")
    np.testing.assert_allclose(mle.matrix, result.matrix * (count - 1) / count, rtol=1e-13)
    weights = np.full(len(window.asset_ids), 1 / len(window.asset_ids))
    assert result.portfolio_variance(weights, window.asset_ids) >= 0
    with pytest.raises(ValueError, match="order mismatch"):
        result.portfolio_variance(weights, tuple(reversed(window.asset_ids)))
    with pytest.raises(ValueError, match="Minimum observation"):
        estimate_covariance(window, min_observations=count + 1)


def test_ewma_actual_weighted_reference(real_windows):
    window = real_windows[63]
    decay = 2 ** (-1 / 21)
    result = estimate_covariance(window, "ewma", ewma_lambda=decay)
    weights = decay ** np.arange(62, -1, -1, dtype=float)
    weights /= weights.sum()
    location = np.sum(window.values * weights[:, None], axis=0)
    reference = sum(
        weights[i] * np.outer(row - location, row - location) for i, row in enumerate(window.values)
    ) / (1 - np.sum(weights**2))
    np.testing.assert_allclose(result.matrix, reference, rtol=1e-12, atol=1e-18)
    assert result.diagnostics["effective_weight_count"] == pytest.approx(1 / np.sum(weights**2))
    assert result.parameters["centering"] == "weighted_mean"


@pytest.mark.parametrize("method", ["ledoit_wolf", "oas", "pca_factor", "constant_correlation"])
def test_actual_estimators_psd_and_definitions(real_windows, method):
    window = real_windows[126]
    result = estimate_covariance(window, method, n_factors=2, constant_correlation_intensity=0.5)
    assert result.status == "valid"
    assert result.diagnostics["minimum_eigenvalue"] >= -result.diagnostics["rank_tolerance"]
    np.testing.assert_allclose(result.matrix, result.matrix.T, rtol=0, atol=1e-18)
    assert np.min(np.diag(result.matrix)) >= 0
    assert result.diagnostics["repair_amount"] == 0 or method == "pca_factor"
    sample = estimate_covariance(window)
    if method in ("ledoit_wolf", "oas"):
        from sklearn.covariance import OAS, LedoitWolf

        reference = (LedoitWolf if method == "ledoit_wolf" else OAS)(store_precision=False).fit(
            window.values
        )
        np.testing.assert_allclose(result.matrix, reference.covariance_, rtol=1e-13)
        assert result.parameters["shrinkage"] == reference.shrinkage_
        assert result.parameters["target"] == "scaled_identity"
        assert result.parameters["denominator"] == 126
    elif method == "pca_factor":
        np.testing.assert_allclose(np.diag(result.matrix), np.diag(sample.matrix), rtol=1e-12)
        assert result.parameters["residual_floor"] == 0
        assert len(result.parameters["factor_eigenvalues"]) == 2
    else:
        np.testing.assert_allclose(np.diag(result.matrix), np.diag(sample.matrix), rtol=1e-13)
        correlation = np.corrcoef(window.values, rowvar=False)
        p = len(window.asset_ids)
        average = math.fsum(correlation[i, j] for i in range(p) for j in range(p) if i != j) / (
            p * (p - 1)
        )
        assert result.parameters["average_correlation"] == pytest.approx(average)
        assert result.parameters["intensity_rule"] == "fixed_configuration_not_estimated_optimum"
        for i in range(p):
            for j in range(p):
                if i != j:
                    target = average * math.sqrt(sample.matrix[i, i] * sample.matrix[j, j])
                    assert result.matrix[i, j] == pytest.approx(
                        0.5 * sample.matrix[i, j] + 0.5 * target, rel=1e-12, abs=1e-18
                    )


def test_real_short_window_singular_psd_is_accepted(real_windows):
    result = estimate_covariance(real_windows[2])
    assert result.status == "valid"
    assert result.diagnostics["rank"] <= 1
    assert result.diagnostics["singular"]
    assert result.diagnostics["condition_number"] is None
    assert result.diagnostics["repair_amount"] == 0


def test_actual_window_cannot_be_used_before_available(real_windows):
    window = real_windows[63]
    with pytest.raises(ValueError, match="availability violates"):
        replace(window, signal_at=window.available_at[-2])
