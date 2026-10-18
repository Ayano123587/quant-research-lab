"""Actual registered metadata checks; no final choice, release, or estimator fit."""

import json
from pathlib import Path

import pytest

from quantlab.evaluation.holdout_release import (
    compile_documents,
    covariance_parameters,
    prepared_metadata,
)
from quantlab.risk.study import load_spec, recipes

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.real_data


def test_unresolved_actual_wider_gate_refuses_before_any_metadata_or_release(monkeypatch):
    import quantlab.evaluation.holdout_release as compiler

    def forbidden(*args, **kwargs):
        raise AssertionError("Unresolved decision must not open release inputs")

    monkeypatch.setattr(compiler, "_bound_json", forbidden)
    with pytest.raises(ValueError, match="Wider selection unresolved"):
        compile_documents(
            PROJECT,
            root_decision={"wider_resolution": "UNRESOLVED"},
            descriptors=None,
            prepared_manifest_path=None,
            market_manifest_path=None,
            yahoo_manifest_path=None,
            development_evidence=None,
            input_files=None,
            source_sha256=None,
            runtime=None,
            calendar=None,
            solver_policy=None,
            selectors=None,
        )


def test_actual_original214_partitions_filter180_each_without_native_columns():
    path = PROJECT / "data/processed/stock-monthly-v2/manifest.json"
    if not path.exists():
        pytest.skip("Supply actual completed6273 metadata")
    manifest, rows = prepared_metadata(
        path, "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    )
    assert manifest["complete"] is True
    assert len(rows) == 360 and all(r["session"] <= "2014-12-31" for r in rows)
    assert all(sum(r["kind"] == kind for r in rows) == 180 for kind in ("features", "labels"))
    assert all(r["sha256"] and r["bytes"] > 0 for r in rows)
    with pytest.raises(ValueError, match="actual metadata SHA"):
        prepared_metadata(path, manifest["source_manifest_sha256"])


def test_all_actual_registered_covariance_windows_expand_exact_adapter_parameters():
    spec = load_spec(PROJECT / "configs/risk-development.toml")
    expected_fields = {
        "method",
        "min_observations",
        "ewma_lambda",
        "n_factors",
        "residual_floor",
        "constant_correlation_intensity",
        "relative_tolerance",
        "absolute_tolerance",
    }
    for count in spec["panels"]["return_observations"]:
        parameters = covariance_parameters(spec, count)
        assert len(parameters) == 8
        for recipe in recipes(spec):
            actual = parameters[recipe.recipe_id]
            assert set(actual) == expected_fields and actual["min_observations"] == count
            assert actual["method"] == recipe.method
            assert all(actual[k] == v for k, v in recipe.parameters)
            assert actual["relative_tolerance"] == 1e-10 and actual["absolute_tolerance"] == 1e-15
        assert set(json.loads(json.dumps(parameters))) == set(parameters)


def test_actual2010_forecast_only_stream_replays_verified_cells_without_solver_or_ledger(
    monkeypatch, tmp_path
):
    import hashlib
    import os
    import zipfile
    from datetime import date

    from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.provenance import SourceIdentity
    from quantlab.evaluation.holdout import calendar_window, file_sha
    from quantlab.evaluation.holdout_adapters import ArchiveProvider
    from quantlab.evaluation.holdout_registry import (
        RegisteredProvider,
        SharedHistory,
        construct_risk_provider,
    )

    publication = (
        PROJECT.parent / "quant-research-lab-private/risk-economic-forecast-publication-v1"
    )
    freeze_path = publication / "root-data-freeze-v1.json"
    archive = Path(os.environ.get("QUANTLAB_ARCHIVE", PROJECT / "data/raw/archive.zip"))
    if not freeze_path.exists() or not archive.exists():
        pytest.skip("Supply actual verified saved development forecasts and original archive")
    freeze = json.loads(freeze_path.read_bytes())
    transport_path = publication / "upload/risk-integration-transport.json"
    assert file_sha(transport_path) == freeze["transport_manifest_sha256"]
    transport = json.loads(transport_path.read_bytes())
    members = {r["path"]: r for r in transport["members"]}
    count, day = 63, date(2010, 6, 30)
    spec = load_spec(PROJECT / "configs/risk-development.toml")
    parameters = covariance_parameters(spec, count)
    saved = {}
    with zipfile.ZipFile(publication / "upload" / freeze["container"]["name"]) as container:
        for recipe_id in parameters:
            name = "cells/2010-06-30/etf8_T63/" + recipe_id + ".json"
            raw = container.read(name)
            assert len(raw) == members[name]["bytes"]
            assert hashlib.sha256(raw).hexdigest() == members[name]["sha256"]
            row = json.loads(raw)
            assert row["signal_session"] == str(day) and row["observations"] == count
            saved[recipe_id] = {k: v for k, v in row.items() if k != "matrix"}
    schedule = XNYSSchedule(date(2008, 1, 1), date(2011, 1, 31))
    identity = SourceIdentity(
        "original_archive",
        schedule.provenance["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    history = SharedHistory(
        ArchiveProvider(
            archive,
            SUPPLIED_ARCHIVE_SHA256,
            tuple(("etf", s) for s in spec["panels"]["etf_symbols"]),
            identity,
        )
    )
    provider = construct_risk_provider(
        {
            "kind": "risk_forecast",
            "observations": count,
            "estimator_parameters": parameters,
            "panel": {"kind": "etf8_fixed"},
        },
        freeze={},
        inputs=PROJECT,
        schedule=schedule,
        identity=identity,
        signals=(day,),
    )
    assert provider.arm is None and provider.solver_policy is None

    def replay(view, reference):
        assert reference.asset_ids == tuple(saved["sample_ddof1"]["asset_ids"])
        assert reference.weights == (1 / 8,) * 8
        assert reference.known_at == reference.max_input_available_at == view.cutoff
        provider.forecast_provider.references[day] = reference
        provider.forecast_provider.forecasts[day] = saved
        return saved

    def no_economics(*args):
        raise AssertionError("Forecast-only streams cannot invoke an economic solver")

    monkeypatch.setattr(provider.forecast_provider, "forecast", replay)
    monkeypatch.setattr(provider, "decide", no_economics)
    wrapper = RegisteredProvider(
        "actual",
        history,
        provider,
        arm_id="actual2010_forecast_only",
        schedule=schedule,
        refit_policy="none",
    )
    wrapper.export_root = tmp_path / "stream"
    window = calendar_window(schedule, day, date(2010, 7, 2))
    loaded = wrapper.load("actual", window)
    wrapper.prepare_stream(loaded, window)
    assert set(wrapper.stream_exports) == {day} and not wrapper.targets and not wrapper.fit_receipts
    record = json.loads(Path(wrapper.stream_exports[day]["path"]).read_bytes())
    assert set(record["forecasts"]) == set(parameters) and record["target"] is None
    assert not list(tmp_path.rglob("*ledger*"))
