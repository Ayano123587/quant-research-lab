"""Observed feature projections and saved predictions only; no host model fit."""

import os
from datetime import date
from pathlib import Path

import pytest

from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks import ablation, development
from quantlab.stocks.prepare import SOURCE_MANIFEST_SHA256, prepare_stock_panel
from quantlab.stocks.training_targets import RAW, training_targets

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def actual_data(tmp_path_factory):
    source_path = Path(os.environ.get("QUANTLAB_SNAPSHOT", "data/processed/market-v1"))
    if not (source_path / "COMPLETE").exists():
        pytest.skip("Authenticated real snapshot absent; set QUANTLAB_SNAPSHOT")
    schedule = XNYSSchedule(date(2008, 1, 1), date(2018, 12, 31))
    source = MarketSnapshot(source_path, SOURCE_MANIFEST_SHA256)
    output = tmp_path_factory.mktemp("actual-ablation-feature-columns") / "scoped-inputs"
    receipt = prepare_stock_panel(
        source,
        schedule,
        output,
        start=date(2010, 1, 1),
        end=date(2014, 12, 31),
        scoped_test_assets=("stock:ibm.us", "stock:ge.us", "stock:aapl.us", "stock:tsla.us"),
    )
    spec = development.load_spec("configs/stock-development.toml")
    data = development.read_development_inputs(
        output, receipt["manifest_sha256"], spec, schedule, allow_scoped_test=True
    )
    return data, spec


def test_registered_five_real_column_projections(actual_data):
    data, _ = actual_data
    before = tuple(row.values for row in data.features)
    arms = ablation.projections()
    assert len(arms) == 5
    assert [len(arm.feature_ids) for arm in arms] == [6, 3, 5, 5, 5]
    for arm in arms:
        projected = ablation.project_values(data.features, arm)
        assert projected == tuple(
            tuple(row.values[i] for i in arm.indices) for row in data.features
        )
        assert len(projected) == len(data.features)
    assert tuple(row.values for row in data.features) == before


def test_projected_hash_binds_actual_raw_batch_without_target_change(actual_data):
    data, spec = actual_data
    day, cutoff = data.decision_times[-1]
    batch = development.training_batch(data, day, cutoff, spec)
    original = training_targets(batch, RAW)
    hashes = [ablation.projected_input_sha(batch, arm) for arm in ablation.projections()]
    assert len(set(hashes)) == 5
    assert training_targets(batch, RAW) == original
    assert all(label.available_at <= cutoff for label in batch.outcomes)
    assert all(label.signal_session < day for label in batch.outcomes)


def test_actual_saved_reference_full_96_date_score_and_fit_authentication():
    path = Path(os.environ.get("QUANTLAB_STOCK_REFERENCE", "data/results/stock-development"))
    if not (path / "COMPLETE").exists():
        pytest.skip("Complete original real cloud prediction reference absent")
    spec = development.load_spec("configs/stock-development.toml")
    predictions, metrics, fits = ablation.authenticate_reference(path, spec)
    assert len(predictions) == len(metrics) == len(fits) == 96
    assert sum(row["status"] == "valid" for row in metrics.values()) == 94
    for day, rows in predictions.items():
        scores = tuple((row["asset_id"], row["score"]) for row in rows)
        assert ablation.score_bytes(scores) == ablation.score_bytes(tuple(reversed(scores)))
        assert day < date(2015, 1, 1)
        assert fits[day]["status"] == "fitted"


def test_actual_registration_and_host_native_training_refusal(tmp_path, monkeypatch):
    spec = ablation.load_spec("configs/stock-feature-ablation-development.toml")
    assert spec["historical_preregistration"] is False
    assert spec["candidate_id"] == ablation.CANDIDATE_ID
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
        with pytest.raises(ValueError, match="restricted to Kaggle"):
            ablation.run_feature_ablation(
                None, None, None, None, None, None, tmp_path / "forbidden"
            )
    assert not (tmp_path / "forbidden").exists()
