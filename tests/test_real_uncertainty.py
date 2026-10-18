"""Uncertainty checks on complete, externally authenticated cloud ledgers."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.evaluation.uncertainty import paired_block_intervals


@pytest.fixture(scope="module")
def observed():
    root = os.environ.get("QUANT_ETF_COMPLETED_STUDY")
    expected = os.environ.get("QUANT_ETF_COMPLETED_MANIFEST_SHA256")
    if not root or not expected:
        pytest.skip("Provide the externally authenticated completed ETF study")
    root = Path(root)
    assert hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == expected
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert manifest["complete"] is True and manifest["status"] == "complete"
    for name, digest in manifest["outputs_sha256"].items():
        assert Path(name).name == name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    frames = [
        pd.read_parquet(root / f"{name}.parquet")
        for name in ("trend_252_volatility_63", "strategic_equal")
    ]
    assert len(frames[0]) == len(frames[1]) == 1975
    assert frames[0]["session"].equals(frames[1]["session"])
    values = []
    for frame in frames:
        nav = frame["close_nav"].to_numpy()
        values.append(nav / np.r_[100_000.0, nav[:-1]] - 1)
    return frames[0]["session"].tolist(), *values


@pytest.mark.real_data
def test_observed_difference_matches_verified_full_cloud_metrics(observed):
    sessions, candidate, baseline = observed
    report = paired_block_intervals(sessions, candidate, baseline, block_lengths=(63,), draws=100)
    assert report["sessions"] == 1975
    assert report["observed_annual_sharpe_difference"] == pytest.approx(
        0.6971547593351592 - 0.42023705898644265, abs=1e-12
    )
    assert report["observed_annual_arithmetic_mean_difference"] == pytest.approx(
        sum(float(a - b) for a, b in zip(candidate, baseline, strict=True)) / 1975 * 252,
        abs=1e-14,
    )


@pytest.mark.real_data
def test_same_actual_arm_has_exactly_zero_paired_uncertainty(observed):
    sessions, candidate, _ = observed
    report = paired_block_intervals(sessions, candidate, candidate, block_lengths=(21,), draws=100)
    block = report["block_sensitivity"][0]
    assert block["annual_arithmetic_mean_difference_interval"] == [0.0, 0.0]
    assert block["annual_sharpe_difference_interval"] == [0.0, 0.0]
    assert block["undefined_sharpe_draws"] == 0


@pytest.mark.real_data
def test_swap_sign_and_block_order_invariance(observed):
    sessions, candidate, baseline = observed
    left = paired_block_intervals(sessions, candidate, baseline, block_lengths=(21, 63), draws=100)
    right = paired_block_intervals(sessions, baseline, candidate, block_lengths=(63, 21), draws=100)
    for forward, reverse in zip(
        left["block_sensitivity"], right["block_sensitivity"][::-1], strict=True
    ):
        assert forward["block_sessions"] == reverse["block_sessions"]
        for key in (
            "annual_arithmetic_mean_difference_interval",
            "annual_sharpe_difference_interval",
        ):
            assert forward[key] == pytest.approx([-v for v in reverse[key][::-1]], abs=1e-14)


@pytest.mark.real_data
def test_refuse_unmatched_or_reordered_actual_observations(observed):
    sessions, candidate, baseline = observed
    with pytest.raises(ValueError, match="matched session"):
        paired_block_intervals(sessions, candidate[:-1], baseline)
    with pytest.raises(ValueError, match="increasing"):
        paired_block_intervals(sessions[::-1], candidate[::-1], baseline[::-1])
