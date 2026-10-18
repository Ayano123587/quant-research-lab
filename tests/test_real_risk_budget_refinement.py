"""Two observed Yahoo covariance windows only, no strategy or target-date scan."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import test_real_source_integration as actual_source

from quantlab.allocation.policies import allocate
from quantlab.allocation.weighting import strategic_weights
from quantlab.backtest.engine import HistoricalView

source = actual_source.source
PRIVATE = Path(__file__).resolve().parents[1].parent / "quant-research-lab-private"


def observed_estimate(source, day):
    _, schedule, identity, histories = source
    view = HistoricalView(
        tuple((a, tuple(b for b in rows if b.session <= day)) for a, rows in histories.items()),
        day,
        schedule.session(day).decision_at,
        schedule.next_session(day).open_at,
    )
    return allocate(
        view, schedule, policy="trend_252_volatility_63", source_identity=identity
    ).covariance


def test_actual_may31_failed_original_certificate_refines_same_problem_once(source):
    report = (
        PRIVATE
        / "etf-target-diagnostic-publication-v1/scientific-small-output-v1"
        / "etf-target-diagnostic/target-diagnostic.json"
    )
    if not report.exists():
        pytest.skip("Authenticated actual cloud failure report required")
    raw = report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "4b04679506bdaa8b27f02531f11f67a526cc061806aaa197975634094eb7d14f"
    )
    failure = json.loads(raw)["first_exception"]
    estimate = observed_estimate(source, date(2007, 5, 31))
    assert estimate.window_hash == failure["observed_allowlisted_locals"]["estimate"]["window_hash"]
    np.testing.assert_array_equal(estimate.matrix, failure["observed_allowlisted_locals"]["matrix"])
    result = strategic_weights(estimate, "capped_risk_budget")
    initial = result.solver["initial_attempt"]
    refined = result.solver["refinement_attempt"]
    assert initial["success"] and initial["stationarity_residual"] > 1e-6
    assert initial["stationarity_residual"] == pytest.approx(
        failure["observed_allowlisted_locals"]["stationarity_residual"], abs=1e-14
    )
    assert refined["success"] and refined["ftol"] == 1e-14
    assert refined["stationarity_residual"] <= 1e-6
    assert refined["complementarity_residual"] <= 1e-7
    assert refined["objective"] <= initial["objective"]
    assert max(result.weights) <= 0.25 and math.fsum(result.weights) <= 1
    assert result.covariance_window_sha256 == estimate.window_hash
    assert abs(math.fsum(result.risk_shares) - 1) <= 1e-12


def test_actual_feb_first_signal_keeps_original_iterate_without_refinement(source):
    estimate = observed_estimate(source, date(2007, 2, 28))
    assert (
        estimate.window_hash == "915f6914ee7276b1a13ef198029e275b2ad76e0703031f2d191b235b1f50bca3"
    )
    result = strategic_weights(estimate, "capped_risk_budget")
    assert result.solver["refinement_attempt"] is None
    assert result.solver["initial_attempt"]["ftol"] == result.solver["ftol"] == 1e-12
    assert result.solver["stationarity_residual"] <= 1e-6
    assert result.solver["complementarity_residual"] <= 1e-7
    assert max(result.weights) <= 0.25 and math.fsum(result.weights) <= 1
