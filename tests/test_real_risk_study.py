"""Actual-source engineering arithmetic only; no full study or model selection."""

import os
from bisect import bisect_left, bisect_right
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.risk import study
from quantlab.risk.covariance import estimate_covariance

pytestmark = pytest.mark.real_data
MANIFEST_SHA256 = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"


@pytest.mark.parametrize(
    "before,after",
    [
        (
            'selection_scope = "separate_per_panel_and_observation_window_'
            'no_pooling_unlike_variance_scales"',
            'selection_scope = "pool_unlike_panels"',
        ),
        (
            'probe_interpretation = "constant_weight_daily_return_risk_probe_'
            'not_a_tradable_holdings_ledger"',
            'probe_interpretation = "holdings_ledger"',
        ),
        ("stock_asset_count = 100", "stock_asset_count = 150"),
        ("etf_return_observations = 63", "etf_return_observations = 252"),
    ],
)
def test_actual_config_refuses_changed_scoring_or_integration(tmp_path, before, after):
    # Invalid software configuration probes; no market records are altered.
    actual = Path("configs/risk-development.toml").read_text()
    assert actual.count(before) == 1
    path = tmp_path / "rejected-risk-config.toml"
    path.write_text(actual.replace(before, after))
    with pytest.raises(ValueError, match="protocol required before scores"):
        study.load_spec(path)


@pytest.fixture(scope="module")
def source():
    root = Path(os.environ.get("QUANTLAB_SNAPSHOT", "data/processed/market-v1"))
    if not (root / "COMPLETE").exists():
        pytest.skip("Authenticated real snapshot absent; set QUANTLAB_SNAPSHOT")
    snapshot = MarketSnapshot(root, MANIFEST_SHA256)
    schedule = XNYSSchedule(date(2000, 1, 1), date(2018, 12, 31))
    spec = study.load_spec("configs/risk-development.toml")
    histories = {}
    with snapshot.verified_read_session() as verified:
        for kind, symbols in (
            ("etf", spec["panels"]["etf_symbols"]),
            ("stock", ("IBM", "GE", "AAPL", "TSLA")),
        ):
            for symbol in symbols:
                bars = verified.bars(kind, symbol)
                histories[bars[0].asset_id] = bars
    return snapshot, schedule, spec, histories


def _candidate(asset, history, schedule, day):
    days = tuple(b.session for b in history)
    position = bisect_left(schedule.sessions, day)
    past = history[bisect_left(days, schedule.sessions[position - 252]) : bisect_right(days, day)]
    future_days, _ = study.proxy_schedule(schedule, day)
    future = history[bisect_left(days, future_days[0]) : bisect_right(days, future_days[-1])]
    return study.asset_window(asset, past, schedule, signal_session=day, future_rows=future)


def test_registered_grid_and_host_full_study_guard(source, tmp_path, monkeypatch):
    snapshot, schedule, spec, _ = source
    assert len(study.recipes(spec)) == 8
    assert len(study.panel_grid(spec)) == 12
    assert len(study.recipes(spec)) * len(study.panel_grid(spec)) == 96
    assert len(study._calendar_dates(schedule, spec)) == 95
    with monkeypatch.context() as host:
        host.chdir(tmp_path)
        actual_is_dir = Path.is_dir
        host.setattr(
            Path,
            "is_dir",
            lambda path: False
            if path in (Path("/kaggle/working"), Path("/kaggle/input"))
            else actual_is_dir(path),
        )
        with pytest.raises(ValueError, match="Kaggle"):
            study.run_covariance_study(
                snapshot, schedule, "configs/risk-development.toml", tmp_path / "no-run"
            )
    assert not (tmp_path / "no-run").exists()


def test_actual_nested_order_never_uses_future_support(source):
    _, schedule, _, histories = source
    day = date(2010, 6, 30)
    candidates = []
    for asset in ("stock:ibm.us", "stock:ge.us", "stock:aapl.us"):
        candidate, reasons = _candidate(asset, histories[asset], schedule, day)
        assert candidate is not None and not reasons
        candidates.append(candidate)
    actual = study.select_nested(candidates, sizes=(1, 2, 3, 50))
    ordered = sorted(candidates, key=lambda c: (-c.activity, c.asset_id))
    assert actual[1] == tuple(ordered[:1])
    assert actual[2][:1] == actual[1] and actual[3][:2] == actual[2]
    assert actual[50] == ()
    assert study.select_nested(reversed(candidates), sizes=(1, 2, 3)) == {
        k: actual[k] for k in (1, 2, 3)
    }
    # Natural actual inception, with no altered rows or generated observations.
    tsla, reasons = _candidate("stock:tsla.us", histories["stock:tsla.us"], schedule, day)
    assert tsla is None and "MISSING_EXACT_253_SCHEDULED_MARKS" in reasons


def test_actual_etf_sample_ewma_and_factor_arithmetic(source):
    snapshot, schedule, spec, histories = source
    day = date(2010, 6, 30)
    panel = tuple(
        _candidate(
            "etf:" + symbol.lower() + ".us",
            histories["etf:" + symbol.lower() + ".us"],
            schedule,
            day,
        )[0]
        for symbol in spec["panels"]["etf_symbols"]
    )
    window = study.return_window(
        panel,
        schedule,
        63,
        signal_session=day,
        source_snapshot_id=snapshot.manifest_sha256,
        archive_sha256=snapshot.archive_sha256,
    )
    assert window.values.shape == (63, 8)
    assert max(window.available_at) == schedule.session(day).decision_at
    assert not window.values.flags.writeable
    q = np.full(8, 1 / 8)
    sample = estimate_covariance(window)
    np.testing.assert_allclose(
        sample.matrix, np.cov(window.values, rowvar=False, ddof=1), rtol=1e-12, atol=1e-18
    )
    assert sample.portfolio_variance(q, window.asset_ids) == pytest.approx(
        np.var(window.values @ q, ddof=1)
    )
    weights = (2 ** (-1 / 63)) ** np.arange(62, -1, -1)
    weights /= weights.sum()
    centered = window.values - weights @ window.values
    expected = (centered.T * weights) @ centered / (1 - weights @ weights)
    ewma = estimate_covariance(window, "ewma")
    np.testing.assert_allclose(ewma.matrix, expected, rtol=1e-12, atol=1e-18)
    factor = estimate_covariance(window, "pca_factor", n_factors=3)
    assert factor.asset_ids == window.asset_ids and factor.parameters["n_factors"] == 3
    assert factor.diagnostics["minimum_eigenvalue"] >= -factor.diagnostics["rank_tolerance"]


def test_actual_common_probe_exact_horizon_and_losses(source):
    snapshot, schedule, _, histories = source
    day = date(2010, 6, 30)
    panel = tuple(
        _candidate(asset, histories[asset], schedule, day)[0]
        for asset in ("stock:ibm.us", "stock:ge.us")
    )
    proxy = study.future_proxy(
        panel,
        schedule,
        signal_session=day,
        information_cutoff=schedule.session(date(2014, 12, 31)).decision_at,
    )
    dates, maturity = study.proxy_schedule(schedule, day)
    assert len(dates) == 22 and dates[0] == schedule.next_session(day).session
    assert proxy["planned_available_at"] == maturity and proxy["status"] == "valid"
    observed = np.asarray(
        [
            [
                b.close / a.close - 1
                for a, b in zip(c.future_rows[:-1], c.future_rows[1:], strict=True)
            ]
            for c in panel
        ]
    ).mean(axis=0)
    assert proxy["value"] == pytest.approx(np.var(observed, ddof=1))
    window = study.return_window(
        panel,
        schedule,
        63,
        signal_session=day,
        source_snapshot_id=MANIFEST_SHA256,
        archive_sha256=snapshot.archive_sha256,
    )
    forecast = estimate_covariance(window).portfolio_variance((0.5, 0.5), window.asset_ids)
    measured = study.score_forecast(forecast, proxy)
    assert measured["qlike"] == pytest.approx(np.log(forecast) + proxy["value"] / forecast)
    assert measured["squared_variance_error"] == pytest.approx((forecast - proxy["value"]) ** 2)
    assert study.score_forecast(0, proxy)["status"] == "nonpositive_or_invalid_forecast"


def test_spilled_proxy_clock_blocks_future_arithmetic(source, monkeypatch):
    _, schedule, _, histories = source
    day = date(2014, 12, 31)
    candidate, reasons = _candidate("stock:ibm.us", histories["stock:ibm.us"], schedule, day)
    assert candidate and not reasons
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Future proxy arithmetic before maturity")

    monkeypatch.setattr(study.np, "var", forbidden)
    result = study.future_proxy(
        (candidate,),
        schedule,
        signal_session=day,
        information_cutoff=schedule.session(date(2014, 12, 31)).decision_at,
    )
    assert result["status"] == "not_mature_by_information_cutoff"
    assert result["value"] is None and not calls


def test_actual_payload_checkpoint_and_partial_grid_refusal(source, tmp_path):
    _, schedule, spec, histories = source
    day = date(2010, 6, 30)
    candidate, _ = _candidate("stock:ibm.us", histories["stock:ibm.us"], schedule, day)
    origin = tmp_path / "interrupted"
    artifact = origin / "cells" / "actual-input.json"
    study._write(
        artifact, {"input_sha256": candidate.input_sha256, "returns252": candidate.returns252}
    )
    pins = {"actual_input_sha256": candidate.input_sha256}
    cell = {
        "signal_session": day.isoformat(),
        "panel_id": "stock50",
        "observations": 63,
        "recipe_id": "sample_ddof1",
        "artifacts": [{"path": "cells/actual-input.json", "sha256": study.file_sha256(artifact)}],
    }
    checkpoint = origin / "checkpoints" / "000001.json"
    study._write(checkpoint, {"pins": pins, "complete": False, "cells": [cell], "parent": None})
    continued = tmp_path / "continued"
    continued.mkdir()
    assert study._restore(checkpoint, study.file_sha256(checkpoint), continued, pins) == [cell]
    assert study.file_sha256(continued / "cells" / "actual-input.json") == study.file_sha256(
        artifact
    )
    assert not (continued / "COMPLETE").exists()
    with pytest.raises(ValueError, match="pins mismatch"):
        study._restore(
            checkpoint, study.file_sha256(checkpoint), tmp_path / "refused", {"changed": True}
        )
    with pytest.raises(ValueError, match="Partial research grid"):
        study.paired_summary([], (day,), spec)
