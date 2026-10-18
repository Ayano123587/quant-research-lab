"""Real source-clock/stream regressions, without a final release or learned fit."""

import json
import os
from datetime import date
from pathlib import Path

import pytest

from quantlab.backtest.engine import Decision, HistoricalView, run_history
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity
from quantlab.evaluation.holdout import calendar_window, file_sha, mature_training_keys
from quantlab.evaluation.holdout_adapters import ArchiveProvider, StockPanelProvider
from quantlab.evaluation.holdout_registry import (
    RegisteredProvider,
    SharedHistory,
    authenticate_factory_inputs,
)

pytestmark = pytest.mark.real_data
PROJECT = Path(__file__).resolve().parents[1]


def test_stock_signal_memo_replays_actual_cloud_fit_once_without_host_native_fit(monkeypatch):
    from dataclasses import replace
    from datetime import datetime
    from types import SimpleNamespace

    from quantlab.evaluation.holdout import FitReceipt, MatureKeys
    from quantlab.evaluation.holdout_preparation import DeferredStockProvider
    from quantlab.evaluation.holdout_registry import StockSignalMemo

    path = PROJECT.parent / (
        "quant-research-lab-private/holdout-adapter-smoke-publication-v1/terminal-small-v1/"
        "holdout-adapter-smoke/engineering/ridge_alpha_1.json"
    )
    if not path.exists():
        pytest.skip("Supply the actual completed native2010 smoke receipt")
    assert file_sha(path) == "13e6922dd41c0bf75c18be897bd5035c33540a8ca546ed84d05229a737e1bced"
    row = json.loads(path.read_bytes())["records"][0]
    raw = row["fit"]
    receipt = FitReceipt(
        raw["fit_id"],
        datetime.fromisoformat(raw["cutoff"]),
        tuple((date.fromisoformat(d), a) for d, a in raw["used_keys"]),
        datetime.fromisoformat(raw["maximum_feature_available_at"]),
    )
    selected = MatureKeys(receipt.used_keys, (), None, ())
    decision_raw = row["decision"]
    decision = Decision(
        decision_raw["action"],
        tuple(tuple(r) for r in decision_raw["target"]),
        datetime.fromisoformat(decision_raw["signal_at"]),
        datetime.fromisoformat(decision_raw["intended_execution_at"]),
        decision_raw["reason"],
    )
    view = HistoricalView(
        (), date.fromisoformat(row["session"]), receipt.cutoff, decision.intended_execution_at
    )
    provider = DeferredStockProvider(
        {}, inputs=PROJECT, schedule=None, signals=(), release={}, release_sha256=file_sha(path)
    )
    calls = {"fit": 0, "decide": 0, "bind": 0}
    predictions = []

    def replay_fit(view, approved):
        calls["fit"] += 1
        assert approved.keys == receipt.used_keys
        return receipt

    def replay_decide(view, state):
        calls["decide"] += 1
        predictions.append(row["predictions"])
        return decision

    def replay_bind(loaded, window, **context):
        calls["bind"] += 1
        provider.root = context["export_root"]

    # Saved genuine cloud results test memo mechanics; no estimator is constructed.
    monkeypatch.setattr(provider, "fit", replay_fit)
    monkeypatch.setattr(provider, "decide", replay_decide)
    monkeypatch.setattr(provider, "bind_history", replay_bind)
    memo = StockSignalMemo(provider, {"actual_receipt_sha256": file_sha(path)})
    loaded, source, window = object(), object(), SimpleNamespace(start=view.session)
    memo.bind_history(loaded, window, source_loader=source, export_root="first-owner")
    memo.bind_history(loaded, window, source_loader=source, export_root="second-arm")
    assert provider.root == "first-owner"
    assert memo.fit(view, selected) is memo.fit(view, selected) is receipt
    assert memo.decide(view, None) is memo.decide(view, None) is decision
    assert calls == {"fit": 1, "decide": 1, "bind": 1}
    assert len(receipt.used_keys) == 44472 and len(predictions) == 1
    assert len(row["predictions"]["scores"]) == 500 and len(decision.target) == 50
    with pytest.raises(ValueError, match="cached mature"):
        memo.fit(view, replace(selected, keys=selected.keys[:-1]))
    with pytest.raises(ValueError, match="pure signals"):
        memo.decide(view, object())
    with pytest.raises(ValueError, match="history/source/window"):
        memo.bind_history(object(), window, source_loader=source, export_root="third-arm")
    with pytest.raises(ValueError, match="history/source/window"):
        memo.bind_history(loaded, window, source_loader=object(), export_root="third-arm")


def test_stock_memo_binding_keeps_full_candidate_source_runtime_and_refit_identity(monkeypatch):
    import copy

    import quantlab.evaluation.holdout_registry as registry
    from quantlab.evaluation.holdout_preparation import DeferredStockProvider

    # Binding mechanics use actual executed source/runtime metadata, without market fabrication.
    complete = PROJECT.parent / (
        "quant-research-lab-private/holdout-adapter-smoke-publication-v1/terminal-small-v1/"
        "holdout-adapter-smoke/engineering/COMPLETE.json"
    )
    if not complete.exists():
        pytest.skip("Supply actual completed smoke metadata")
    actual = json.loads(complete.read_bytes())
    identity = SourceIdentity(
        "original_archive",
        actual["source_sha256"]["src/quantlab/data/calendar.py"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    created = []

    def construct(spec, **context):
        result = DeferredStockProvider(
            spec,
            inputs=PROJECT,
            schedule=None,
            signals=(),
            release={},
            release_sha256=file_sha(complete),
        )
        created.append(result)
        return result

    monkeypatch.setattr(registry, "construct_stock_provider", construct)
    spec = {
        "candidate_id": "ridge_alpha_1",
        "candidate_kind": "ridge",
        "candidate_parameters": {"alpha": 1.0},
        "projection": "original_six",
    }
    context = dict(
        refit_policy="monthly_expanding_mature",
        history={"archive_sha256": SUPPLIED_ARCHIVE_SHA256},
        identity=identity,
        release_sha=file_sha(complete),
        release_context=({"runtime": actual["runtime"]}, file_sha(complete)),
    )
    cache = {}
    first = registry._stock_signal_provider(cache, spec, **context)
    assert registry._stock_signal_provider(cache, copy.deepcopy(spec), **context) is first
    for field, value in (
        ("candidate_parameters", {"alpha": 10.0}),
        ("projection", "registered_wider"),
    ):
        changed = {**spec, field: value}
        assert registry._stock_signal_provider(cache, changed, **context) is not first
    for field, value in (
        ("refit_policy", "initial_only_mature"),
        ("history", {"source_sha256": file_sha(complete)}),
        ("release_sha", file_sha(PROJECT / "pyproject.toml")),
    ):
        changed = registry._stock_signal_provider(cache, spec, **{**context, field: value})
        assert changed is not first
    assert len(created) == 6


def test_actual_7195_labels_join_to_original_500_feature_cohort_before_maturity():
    from quantlab.stocks.development import (
        FEATURE_IDS,
        Candidate,
        DevelopmentData,
        Outcome,
        load_spec,
        training_batch,
    )

    root = Path(os.environ.get("QUANTLAB_PREPARED", PROJECT / "data/processed/stock-monthly-v2"))
    if not root.exists():
        pytest.skip("Supply the actual completed6273 preparation")
    manifest_path = root / "manifest.json"
    digest = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    assert file_sha(manifest_path) == digest
    manifest = json.loads(manifest_path.read_bytes())
    calendar = manifest["config"]["calendar"]
    schedule = XNYSSchedule(
        date.fromisoformat(calendar["start"]), date.fromisoformat(calendar["end"])
    )
    assert schedule.provenance == calendar
    days = (date(2010, 5, 28), date(2010, 6, 30))
    partitions = [
        {**item, "session": Path(item["path"]).stem}
        for item in manifest["outputs"]
        if item.get("kind") in ("features", "labels")
        and Path(item["path"]).stem in {str(d) for d in days}
    ]
    spec = load_spec(PROJECT / "configs/stock-development.toml")
    provider = StockPanelProvider(
        root,
        partitions,
        schedule,
        (),
        candidate=Candidate("control_momentum_F01", "control", (("F01", 1.0),)),
        feature_columns=tuple("rank_" + f for f in FEATURE_IDS),
        raw_columns=tuple("raw_" + f for f in FEATURE_IDS),
        target_policy=spec["target_transform"],
        threads=spec["threads"],
        target_spec=spec,
        minimum_rows=spec["minimum_training_rows"],
        minimum_dates=spec["minimum_training_signal_dates"],
    )
    day = date(2010, 8, 31)
    cutoff = schedule.session(day).decision_at
    view = HistoricalView((), day, cutoff, schedule.next_session(day).open_at)
    projections = []
    read = provider._read

    def observed_read(path, columns):
        projections.append((path.parent.name, tuple(columns)))
        return read(path, columns)

    provider._read = observed_read
    clocks = provider.clocks(view)
    assert len(clocks) == 1000
    assert all("return_value" not in columns for _, columns in projections)
    assert all(
        not any(c.startswith(("rank_", "raw_")) for c in columns)
        for kind, columns in projections
        if kind == "features"
    )
    selected = mature_training_keys(clocks, signal_session=day, decision_at=cutoff)
    provider._read = read
    features, outcomes = [], []
    for signal in days:
        features.extend(provider._features(signal))
        metadata = read(provider.partitions["labels", signal], ["asset_id", "planned_available_at"])
        assert len(metadata) == 7195 and all(r["planned_available_at"] <= cutoff for r in metadata)
        for row in read(
            provider.partitions["labels", signal],
            [
                "asset_id",
                "signal_session",
                "label_available_at",
                "return_value",
                "status",
                "input_rows_sha256",
            ],
        ):
            if row["status"] == "valid":
                outcomes.append(
                    Outcome(
                        row["asset_id"],
                        signal,
                        row["label_available_at"],
                        row["return_value"],
                        row["input_rows_sha256"],
                    )
                )
    data = DevelopmentData(
        digest, False, cutoff, tuple(features), tuple(outcomes), (), ((day, cutoff),), (), ()
    )
    original = training_batch(data, day, cutoff, spec)
    assert selected.keys == tuple((r.signal_session, r.asset_id) for r in original.feature_rows)
    assert selected.sample_weights == original.weights
    assert selected.maximum_available_at == original.max_label_available_at
    assert not original.ready  # Two real months are an engineering check, never a fitted study.


def test_actual_inception_stops_economics_but_not_full_independent_forecast_stream(tmp_path):
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", PROJECT / "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supply the actual unchanged archive")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        first = source.bars("etf", "BIL")[0].session
    schedule = XNYSSchedule(date(first.year - 1, 1, 1), date(first.year + 1, 1, 31))
    previous_year, previous_month = (
        (first.year - 1, 12) if first.month == 1 else (first.year, first.month - 1)
    )
    start = schedule.month_end(previous_year, previous_month).session
    end = schedule.next_session(schedule.month_end(first.year, first.month).session).session
    identity = SourceIdentity(
        "original_archive",
        schedule.provenance["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    loader = SharedHistory(
        ArchiveProvider(path, SUPPLIED_ARCHIVE_SHA256, (("etf", "BIL"),), identity)
    )

    class FixedInceptionControl:
        # An explicit constant target assumption tests actual missing-source refusal.
        # No price/bar/return or learned prediction is generated.
        def decide(self, view, state):
            assert state is None
            assert all(b.session <= view.session for _, bars in view.histories for b in bars)
            return Decision(
                "REBALANCE",
                (("etf:bil.us", 1.0),),
                view.cutoff,
                view.next_open_at,
                "fixed_source_inception_guard_engineering",
            )

    wrapper = RegisteredProvider(
        "actual_archive",
        loader,
        FixedInceptionControl(),
        arm_id="actual_inception_control",
        schedule=schedule,
        refit_policy="none",
    )
    wrapper.export_root = tmp_path / "streams"
    window = calendar_window(schedule, start, end)
    loaded = wrapper.load("actual_archive", window)
    wrapper.prepare_stream(loaded, window)
    assert len(window.rebalance_signals) == 2
    result = run_history(
        loaded.histories,
        schedule,
        start=start,
        end=end,
        initial_cash=100000.0,
        fee_bps=10.0,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=wrapper.decide,
        source_identity=identity,
    )
    assert result.failure is not None and len(result.daily) == 1
    assert set(wrapper.stream_exports) == set(window.rebalance_signals)
    assert all(wrapper.stream_status[d] == "forecast_exported" for d in window.rebalance_signals)
    assert all(file_sha(r["path"]) == r["sha256"] for r in wrapper.stream_exports.values())
    assert window.rebalance_signals[-1] > result.daily[-1].session
    from quantlab.evaluation.holdout_outcomes import covariance_outcomes
    from quantlab.evaluation.holdout_registry import ForecastSupportError

    class ActualInsufficientHistory:
        def decide(self, view, state):
            assert len(dict(view.histories)["etf:bil.us"]) < 253
            raise ForecastSupportError(
                "insufficient_past_eligible_panel", "Actual BIL inception has fewer than253 marks"
            )

    unsupported = RegisteredProvider(
        "actual_archive",
        loader,
        ActualInsufficientHistory(),
        arm_id="actual_thin_history",
        schedule=schedule,
        refit_policy="none",
    )
    unsupported.export_root = tmp_path / "unsupported"
    unsupported.load("actual_archive", window)
    unsupported.prepare_stream(loaded, window)
    assert set(unsupported.stream_status.values()) == {"unsupported_forecast"}
    outcome = covariance_outcomes(
        {},
        {},
        loaded.histories,
        schedule,
        expected_sessions=window.rebalance_signals,
        expected_estimators=("sample", "ewma"),
        registered_pairs=(("sample", "ewma"),),
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        archive_endpoint=end,
        scoring_cutoff=schedule.session(end).decision_at,
        evaluation_plan_id="engineering_thin_history",
        recipe_frozen_at=None,
        recipe_information_cutoff=None,
        unsupported_forecasts={
            d: "insufficient_past_eligible_panel" for d in window.rebalance_signals
        },
    )
    assert len(outcome["rows"]) == 2
    assert all(row["status"] == "unsupported_forecast" for row in outcome["rows"].values())
    assert all(row is None for row in outcome["paired_differences"].values())

    class SourceViolation:
        def decide(self, view, state):
            raise ValueError("Changed source authentication")

    invalid = RegisteredProvider(
        "actual_archive",
        loader,
        SourceViolation(),
        arm_id="actual_source_failure",
        schedule=schedule,
        refit_policy="none",
    )
    invalid.export_root = tmp_path / "invalid"
    invalid.load("actual_archive", window)
    with pytest.raises(ValueError, match="Changed source authentication"):
        invalid.prepare_stream(loaded, window)
    assert not invalid.stream_exports


def test_actual_development_policy_cannot_construct_final_factory(tmp_path, monkeypatch):
    policy = PROJECT / "configs/execution-projection.json"
    schedule = XNYSSchedule(date(2010, 1, 1), date(2010, 12, 31))
    with pytest.raises(ValueError, match="Development artifacts"):
        authenticate_factory_inputs(
            policy, file_sha(policy), project=PROJECT, inputs=PROJECT, schedule=schedule
        )
    from quantlab.evaluation.holdout_training_smoke import run_smoke

    with monkeypatch.context() as host:
        host.chdir(tmp_path)
        with pytest.raises(ValueError, match="Kaggle-only"):
            run_smoke(
                policy,
                file_sha(policy),
                project=PROJECT,
                inputs=PROJECT,
                output=tmp_path / "refused",
            )
    assert not (tmp_path / "refused").exists()


def test_actual_parent_future_partition_metadata_refused_without_column_reads():
    from quantlab.evaluation.holdout_preparation import validate_parent_partitions

    path = PROJECT / "data/processed/stock-monthly-v2/manifest.json"
    if not path.exists():
        pytest.skip("Supply actual completed6273 metadata")
    assert file_sha(path) == "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    outputs = json.loads(path.read_bytes())["outputs"]
    features = [
        {**r, "session": Path(r["path"]).stem} for r in outputs if r.get("kind") == "features"
    ]
    assert len(features) == 214
    pre2015 = [r for r in features if r["session"] <= "2014-12-31"]
    assert len(pre2015) == 180
    validate_parent_partitions({"partitions": pre2015})
    validate_parent_partitions(
        {"partitions": [r for r in features if r["session"] == "2010-06-30"]}
    )
    for kind in ("features", "labels"):
        future = next(
            {**r, "session": Path(r["path"]).stem}
            for r in outputs
            if r.get("kind") == kind and Path(r["path"]).stem == "2015-01-30"
        )
        with pytest.raises(ValueError, match="explicitly filtered"):
            validate_parent_partitions({"partitions": [future]})
        with pytest.raises(ValueError, match="explicitly filtered"):
            validate_parent_partitions({"wider_partitions": [future]})


def test_released_builder_formulas_match_genuine_2010_prepared_records(tmp_path):
    import hashlib

    import pyarrow.parquet as pq

    from quantlab.evaluation.holdout_preparation import feature_records, label_records
    from quantlab.stocks.features import FEATURE_IDS

    root = PROJECT / "data/processed/stock-monthly-v2"
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", PROJECT / "data/raw/archive.zip"))
    if not root.exists() or not path.exists():
        pytest.skip("Supply actual original archive and completed development panel")
    assert file_sha(root / "manifest.json") == (
        "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    )
    manifest = json.loads((root / "manifest.json").read_bytes())
    day = date(2010, 6, 30)
    entries = {
        r["kind"]: r
        for r in manifest["outputs"]
        if r.get("kind") in {"features", "labels"} and Path(r["path"]).stem == str(day)
    }
    for entry in entries.values():
        assert file_sha(root / entry["path"]) == entry["sha256"]
    original = pq.read_table(root / entries["features"]["path"]).to_pylist()[:2]
    ids = {r["asset_id"] for r in original}
    calendar = manifest["config"]["calendar"]
    schedule = XNYSSchedule(
        date.fromisoformat(calendar["start"]), date.fromisoformat(calendar["end"])
    )
    identity = SourceIdentity(
        "original_archive",
        schedule.provenance["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    histories, inventory = {}, {}
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        for asset in sorted(ids):
            symbol = asset.split(":", 1)[1]
            member, raw = source.source_bytes("stock", symbol)
            inventory[asset] = {
                "source_member": member,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            }
            histories[asset] = tuple(
                b for b in source.bars("stock", symbol) if b.session <= date(2010, 8, 3)
            )
    view = HistoricalView(
        tuple((a, tuple(b for b in rows if b.session <= day)) for a, rows in histories.items()),
        day,
        schedule.session(day).decision_at,
        schedule.next_session(day).open_at,
    )
    features, audit = feature_records(
        view, schedule, identity, inventory, "engineering_actual_source", top_n=500
    )
    assert len(audit) == len(ids) == len(features)
    by_id = {r["asset_id"]: r for r in features}
    for row in original:
        for key in (
            "signal_at",
            "intended_execution_at",
            "max_input_available_at",
            "input_rows_sha256",
            *("raw_" + f for f in FEATURE_IDS),
        ):
            assert by_id[row["asset_id"]][key] == row[key]
    labels = label_records(
        histories,
        schedule,
        identity,
        inventory,
        "engineering_actual_source",
        day=day,
        endpoint=date(2010, 8, 3),
    )
    original_labels = {
        r["asset_id"]: r
        for r in pq.read_table(root / entries["labels"]["path"]).to_pylist()
        if r["asset_id"] in ids
    }
    for row in labels:
        for key in (
            "return_value",
            "entry_mark",
            "exit_mark",
            "status",
            "label_available_at",
            "entry_source_row_sha256",
            "exit_source_row_sha256",
            "input_rows_sha256",
        ):
            assert row[key] == original_labels[row["asset_id"]][key]
    # Reproduce a genuinely earlier data vintage using the same unmodified
    # source marks and an actual calendar endpoint before the planned exit.
    import shutil
    from types import SimpleNamespace

    import pyarrow as pa

    from quantlab.evaluation.holdout import LoadedHistory
    from quantlab.evaluation.holdout_preparation import DeferredStockProvider
    from quantlab.stocks.development import Candidate, load_spec
    from quantlab.stocks.prepare import _schemas

    parent = tmp_path / "actual_2010_vintage"
    parent.mkdir()
    feature_path, label_path = parent / "features.parquet", parent / "labels.parquet"
    shutil.copyfile(root / entries["features"]["path"], feature_path)
    censored = label_records(
        histories,
        schedule,
        identity,
        inventory,
        manifest["source_snapshot_id"],
        day=day,
        endpoint=date(2010, 7, 30),
    )
    assert all(r["status"] == "right_censored" and r["return_value"] is None for r in censored)
    pq.write_table(pa.Table.from_pylist(censored, schema=_schemas(pa)["labels"]), label_path)
    parent_sha = file_sha(label_path)
    spec = load_spec(PROJECT / "configs/stock-development.toml")

    def vintage_provider():
        return StockPanelProvider(
            parent,
            [
                {
                    "kind": kind,
                    "session": str(day),
                    "path": name.name,
                    "bytes": name.stat().st_size,
                    "sha256": file_sha(name),
                }
                for kind, name in (("features", feature_path), ("labels", label_path))
            ],
            schedule,
            (day,),
            candidate=Candidate("control_momentum_F01", "control", (("F01", 1.0),)),
            feature_columns=tuple("rank_" + f for f in FEATURE_IDS),
            raw_columns=tuple("raw_" + f for f in FEATURE_IDS),
            target_policy=spec["target_transform"],
            threads=spec["threads"],
            target_spec=spec,
            minimum_rows=spec["minimum_training_rows"],
            minimum_dates=spec["minimum_training_signal_dates"],
        )

    bridge = DeferredStockProvider(
        {},
        inputs=parent,
        schedule=schedule,
        signals=(date(2010, 7, 30),),
        release={},
        release_sha256=file_sha(root / "manifest.json"),
    )
    bridge.actual = vintage_provider()
    bridge.loaded = LoadedHistory(histories, identity)
    bridge.source = SimpleNamespace(_assets=inventory, snapshot_id=manifest["source_snapshot_id"])
    bridge.window = calendar_window(schedule, date(2010, 7, 30), date(2010, 8, 3))
    bridge.root = tmp_path / "new_overrides"
    bridge.root.mkdir()
    with pytest.raises(ValueError, match="future label generation"):
        bridge._labels(
            day, cutoff=schedule.session(date(2010, 7, 30)).decision_at, purpose="mature_training"
        )
    bridge._labels(
        day, cutoff=schedule.session(date(2010, 8, 2)).decision_at, purpose="mature_training"
    )
    override = bridge.actual.partitions["labels", day]
    assert override != label_path and file_sha(label_path) == parent_sha
    assert all(r["status"] == "valid" for r in pq.read_table(override).to_pylist())
    assert file_sha(feature_path) == entries["features"]["sha256"]
    # For an initial final signal, even this existing censored partition may
    # only be refreshed after every frozen arm's genuine forecast bytes exist.
    bridge.window = calendar_window(schedule, day, date(2010, 8, 3))
    bridge.signals = bridge.window.rebalance_signals
    bridge.actual, bridge.generated = vintage_provider(), {}
    bridge.root = tmp_path / "initial_signal_overrides"
    bridge.root.mkdir()
    with pytest.raises(ValueError, match="prior forecast export"):
        bridge._labels(
            day, cutoff=schedule.session(date(2010, 8, 2)).decision_at, purpose="mature_training"
        )
    forecast = (
        PROJECT.parent / "quant-research-lab-private/stock-model-publication-v1/"
        "scientific-small-output-v1/stock-development/study/predictions/2010-06-30.parquet"
    )
    assert forecast.is_file()
    bridge.bind_forecast_exports(
        {
            "actual_saved_predictions": {
                day: {
                    "path": str(forecast),
                    "sha256": file_sha(forecast),
                    "bytes": forecast.stat().st_size,
                }
            }
        }
    )
    bridge._labels(
        day, cutoff=schedule.session(date(2010, 8, 2)).decision_at, purpose="mature_training"
    )
    assert bridge.actual.partitions["labels", day] != label_path
    assert file_sha(label_path) == parent_sha


def test_genuine_empty_source_only_exclusion_keeps_nonempty_inception_history():
    from dataclasses import replace

    from quantlab.evaluation.holdout import economic_source_receipt

    manifest = PROJECT / "data/processed/market-v1/manifest.json"
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", PROJECT / "data/raw/archive.zip"))
    if not manifest.exists() or not path.exists():
        pytest.skip("Supply authenticated original archive and normalized source metadata")
    assert file_sha(manifest) == (
        "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
    )
    inventory = {r["asset_id"]: r for r in json.loads(manifest.read_bytes())["source_inventory"]}
    schedule = XNYSSchedule(date(2010, 1, 1), date(2010, 12, 31))
    identity = SourceIdentity(
        "original_archive",
        schedule.provenance["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    loaded = ArchiveProvider(
        path, SUPPLIED_ARCHIVE_SHA256, (("stock", "ACCP"), ("etf", "BIL")), identity
    ).load()
    receipt = economic_source_receipt(loaded)
    assert receipt["forecast_population_count"] == 2
    assert receipt["economic_history_count"] == 1
    assert receipt["omitted_globally_empty_assets"] == ("stock:accp.us",)
    empty = loaded.source_empty_evidence[0]
    actual = inventory[empty.asset_id]
    assert actual["expected_rows"] == actual["rows"] == empty.source_rows == 0
    assert actual["source_sha256"] == empty.source_sha256
    assert actual["source_member"] == empty.source_member
    assert loaded.histories["etf:bil.us"]
    # Its actual inception remains later than many genuine exchange sessions;
    # this nonempty history is retained, never coverage-filtered away.
    assert loaded.histories["etf:bil.us"][0].session.year == 2007
    with pytest.raises(ValueError, match="Exact authenticated"):
        economic_source_receipt(replace(loaded, source_empty_assets=()))
    with pytest.raises(ValueError, match="Exact authenticated"):
        economic_source_receipt(
            replace(loaded, source_empty_evidence=(replace(empty, source_rows=1),))
        )
    from quantlab.evaluation.holdout import Arm, _execute

    window = calendar_window(schedule, date(2010, 1, 29), date(2010, 2, 3))
    arm = Arm(
        "empty_target_refusal",
        "stocks",
        "0" * 64,
        None,
        lambda view, state: Decision(
            "REBALANCE",
            ((empty.asset_id, 1.0),),
            view.cutoff,
            view.next_open_at,
            "explicit_invalid_empty_member_engineering_target",
        ),
    )
    with pytest.raises(ValueError, match="Target mentions"):
        _execute(loaded, schedule, window, arm, "none", 100000.0, 10.0)
