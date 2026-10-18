"""Readable exports and fixed intervals checked against actual development results."""

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quantlab.evaluation.holdout_tables import (
    _mean_intervals,
    comparison_intervals,
    comparison_tables,
)
from quantlab.evaluation.uncertainty import paired_ic_intervals

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.real_data


def test_actual94_month_stock_projection_preserves_values_and_existing_paired_intervals():
    path = PROJECT.parent / (
        "quant-research-lab-private/stock-model-publication-v1/scientific-small-output-v1/"
        "stock-development/study/metrics.json"
    )
    if not path.exists():
        pytest.skip("Supply actual completed stock development metrics")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a4ac02351747c5085b73a92d9e313b14b389b9b3c0b20ed00046bded7881af43"
    )
    arms = ["control_fixed_composite", "control_momentum_F01"]
    actual = [
        r
        for r in json.loads(path.read_bytes())
        if r["candidate_id"] in arms and r["status"] == "valid"
    ]
    indexed = {(r["candidate_id"], date.fromisoformat(r["signal_session"])): r for r in actual}
    days = sorted({d for _, d in indexed})
    assert len(days) == 94 and len(indexed) == 188
    result = {"per_arm_date": indexed, "outcome_status_by_session": {d: "complete" for d in days}}
    spec = {
        "kind": "stock_prediction",
        "comparison_id": "actual_development_replay",
        "arm_ids": arms,
        "analysis_policy": {"paired_block_lengths": [3, 6, 12]},
    }
    original = json.dumps(actual, sort_keys=True)
    rows = comparison_tables(spec, result)["stock_date_metrics"]
    assert len(rows) == 188
    assert all(r["ic"] == indexed[r["arm_id"], r["signal_session"]]["ic"] for r in rows)
    intervals = comparison_intervals(spec, result, {})
    direct = paired_ic_intervals(
        days,
        {a: [indexed[a, d]["ic"] for d in days] for a in arms},
        reference=arms[0],
        block_months=6,
    )
    for expected in direct["candidates"]:
        observed = next(
            r
            for r in intervals
            if r["block_length"] == 6 and r["arm_id"] == expected["candidate_id"]
        )
        assert [observed["lower"], observed["upper"]] == expected["paired_difference_interval"]
    assert json.dumps(actual, sort_keys=True) == original


def test_original_covariance95_position_bootstrap_matches_published288_interval_export():
    source = PROJECT / "reports/risk_forecast_calibration_v1/all_forecast_scores.csv"
    reference = PROJECT / "reports/risk_forecast_calibration_v1/paired_loss_intervals.csv"
    if not source.exists() or not reference.exists():
        pytest.skip("Supply actual completed risk calibration outputs")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "f0e4467650aaf008786acdc8be875a5bb4f2498272b597a325c74ebe65c37e69"
    )
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
        "138fa81d0014a1d528c36b2831226e63e9eb1dbc5d3e264514892f01d66032ce"
    )
    with source.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["panel"] == "etf8" and r["lookback"] == "63"]
    indexed = {(r["signal_session"], r["recipe"]): r for r in rows}
    days = sorted({r["signal_session"] for r in rows})
    assert len(days) == 95
    values = []
    for day in days:
        a, b = indexed[day, "ewma_halflife63"]["qlike"], indexed[day, "sample_ddof1"]["qlike"]
        values.append(float(a) - float(b) if a and b else None)
    assert any(value is None for value in values)  # Genuine retained censor positions.
    actual = _mean_intervals([date.fromisoformat(d) for d in days], values, [3, 6, 12])
    with reference.open(newline="") as stream:
        expected = [
            r
            for r in csv.DictReader(stream)
            if r["panel"] == "etf8" and r["lookback"] == "63" and r["recipe"] == "ewma_halflife63"
        ]
    assert len(actual) == len(expected) == 3
    for row in actual:
        original = next(r for r in expected if int(r["block_months"]) == row["block_length"])
        assert row["lower"] == pytest.approx(float(original["lower"]), abs=1e-14)
        assert row["upper"] == pytest.approx(float(original["upper"]), abs=1e-14)
        assert row["mean_difference"] == pytest.approx(
            float(original["mean_difference"]), abs=1e-14
        )
