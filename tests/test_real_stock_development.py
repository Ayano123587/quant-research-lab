"""Actual prepared rows exercise clocks/support only: no host model fitting."""

import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pytest

from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks import development
from quantlab.stocks.prepare import SOURCE_MANIFEST_SHA256, prepare_stock_panel
from quantlab.stocks.training_targets import RANK, RAW, training_targets

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    source_path = Path(os.environ.get("QUANTLAB_SNAPSHOT", "data/processed/market-v1"))
    if not (source_path / "COMPLETE").exists():
        pytest.skip("Reviewed complete real snapshot absent; set QUANTLAB_SNAPSHOT")
    source = MarketSnapshot(source_path, SOURCE_MANIFEST_SHA256)
    schedule = XNYSSchedule(date(2008, 1, 1), date(2018, 12, 31))
    output = tmp_path_factory.mktemp("actual-stock-development-clocks") / "scoped-software-inputs"
    # Four unchanged real histories, sixty feature dates, zero fitted models.
    # TSLA's actual inception produces naturally different row counts per date.
    receipt = prepare_stock_panel(
        source,
        schedule,
        output,
        start=date(2010, 1, 1),
        end=date(2014, 12, 31),
        scoped_test_assets=("stock:ibm.us", "stock:ge.us", "stock:aapl.us", "stock:tsla.us"),
    )
    spec = development.load_spec("configs/stock-development.toml")
    return output, receipt["manifest_sha256"], schedule, spec


@pytest.fixture(scope="module")
def data(inputs):
    output, digest, schedule, spec = inputs
    return development.read_development_inputs(
        output, digest, spec, schedule, allow_scoped_test=True
    )


def test_actual_mature_training_rank_targets(data, inputs):
    import math

    from scipy.stats import rankdata

    spec = inputs[3]
    day, cutoff = data.decision_times[-1]
    batch = development.training_batch(data, day, cutoff, spec)
    original = tuple(row.value for row in batch.outcomes)
    assert training_targets(batch, RAW) == original
    actual = training_targets(batch, RANK)
    groups = defaultdict(list)
    for index, row in enumerate(batch.outcomes):
        groups[row.signal_session].append(index)
    assert len(groups) > 36
    for indices in groups.values():
        expected = (rankdata([original[i] for i in indices], method="average") - 0.5) / len(
            indices
        ) - 0.5
        assert [actual[i] for i in indices] == pytest.approx(expected)
        assert abs(math.fsum(actual[i] for i in indices)) < 1e-14
    assert tuple(row.value for row in batch.outcomes) == original
    assert all(-0.5 < value < 0.5 for value in actual)


def test_rank_target_spec_is_explicit_and_same_grid(inputs):
    ranked = development.load_spec("configs/stock-rank-target-development.toml")
    raw = inputs[3]
    assert ranked["target_transform"] == RANK
    assert development.registered_candidates(ranked) == development.registered_candidates(raw)
    assert {
        key: value for key, value in ranked.items() if key not in ("status", "target_transform")
    } == {key: value for key, value in raw.items() if key not in ("status", "target_transform")}


def test_real_prepared_panel_refuses_different_preparation_source(inputs, monkeypatch):
    output, digest, schedule, spec = inputs
    actual = development.prepared_source_closure()
    # Inject only a software-version mismatch; all data files stay unchanged.
    changed = {**actual, "features": "0" * 64}
    assert changed != actual
    monkeypatch.setattr(development, "prepared_source_closure", lambda: changed)
    with pytest.raises(ValueError, match="Preparation source differs"):
        development.read_development_inputs(output, digest, spec, schedule, allow_scoped_test=True)


def test_registered_sixteen_candidates_and_unsupported_cutoff_rejected(
    inputs, tmp_path, monkeypatch
):
    output, digest, schedule, spec = inputs
    candidates = development.registered_candidates(spec)
    assert len(candidates) == 16
    assert Counter(c.kind for c in candidates) == {"control": 4, "ridge": 4, "lightgbm": 8}
    assert len({c.candidate_id for c in candidates}) == 16
    with pytest.raises(ValueError, match="2015 outcomes"):
        development.read_development_inputs(
            output,
            digest,
            {**spec, "selection_information_cutoff_session": "2017-12-31"},
            schedule,
            allow_scoped_test=True,
        )
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
        with pytest.raises(ValueError, match="Kaggle runtime"):
            development.run_stock_development(
                output,
                digest,
                "configs/stock-development.toml",
                schedule,
                tmp_path / "forbidden-local-fit",
            )
    assert not (tmp_path / "forbidden-local-fit").exists()


def test_spilled_outcome_value_columns_never_read_even_before_row_filter(inputs, monkeypatch):
    output, digest, schedule, spec = inputs
    actual_reader = development._read_columns
    targets_read = []

    def checked_projection(path, columns):
        if "return_value" in columns:
            day = date.fromisoformat(Path(path).stem)
            assert day < date(2014, 11, 1)
            targets_read.append(day)
        assert "entry_mark" not in columns and "exit_mark" not in columns
        return actual_reader(path, columns)

    monkeypatch.setattr(development, "_read_columns", checked_projection)
    observed = development.read_development_inputs(
        output, digest, spec, schedule, allow_scoped_test=True
    )
    assert targets_read and max(targets_read) == date(2014, 10, 31)
    assert "labels/2014-11-28.parquet" in observed.outcome_value_partitions_refused
    assert "labels/2014-12-31.parquet" in observed.outcome_value_partitions_refused
    assert all(r.signal_session < date(2014, 11, 1) for r in observed.outcomes)
    assert all(r.available_at <= observed.selection_cutoff for r in observed.outcomes)


def test_mature_prequential_batches_and_mean_one_date_weights_on_actual_rows(inputs, data):
    _, _, schedule, spec = inputs
    cutoff = schedule.session(date(2014, 12, 31)).decision_at
    batch = development.training_batch(data, date(2014, 12, 31), cutoff, spec)
    assert len(batch.date_counts) >= 36
    assert len(batch.feature_rows) < 10000 and not batch.ready
    assert batch.reasons == ("INSUFFICIENT_TRAINING_ROWS",)
    assert all(r.signal_session < date(2014, 12, 31) for r in batch.feature_rows)
    assert all(r.available_at <= cutoff for r in batch.outcomes)
    assert max(r.signal_session for r in batch.outcomes) == date(2014, 10, 31)
    assert len({count for _, count in batch.date_counts}) > 1
    totals = defaultdict(float)
    for row, weight in zip(batch.feature_rows, batch.weights, strict=True):
        totals[row.signal_session] += weight
    expected = len(batch.feature_rows) / len(batch.date_counts)
    assert sum(batch.weights) == pytest.approx(len(batch.weights))
    assert all(value == pytest.approx(expected) for value in totals.values())
    previous_cutoff = schedule.session(date(2014, 10, 31)).decision_at
    earlier = development.training_batch(data, date(2014, 10, 31), previous_cutoff, spec)
    assert max(r.signal_session for r in earlier.outcomes) == date(2014, 8, 29)
    assert earlier.input_sha256 != batch.input_sha256
    assert earlier == development.training_batch(data, date(2014, 10, 31), previous_cutoff, spec)


def test_actual_short_cross_section_holds_and_ic_stays_undefined(inputs, data):
    _, _, _, spec = inputs
    day = date(2014, 10, 31)
    current = tuple(r for r in data.features if r.signal_session == day)
    outcomes = tuple(r for r in data.outcomes if r.signal_session == day)
    assert 0 < len(current) < 30
    controls = [c for c in development.registered_candidates(spec) if c.kind == "control"]
    metrics = []
    for candidate in controls:
        scores = development.control_scores(candidate, current)
        intent = development.target_from_scores(
            scores, current[0].signal_at, current[0].intended_execution_at, spec
        )
        assert intent.action == "HOLD" and intent.target is None
        measured = development.date_ic(scores, outcomes)
        assert (
            measured["ic"] is None and measured["status"] == "insufficient_resolved_cross_section"
        )
        assert measured["n_resolved"] == len(outcomes)
        metrics.append({"candidate_id": candidate.candidate_id, "signal_session": day, **measured})
    summary = development.common_paired_summary(metrics, development.registered_candidates(spec))
    assert summary["selected_candidate_id"] is None and summary["common_dates"] == ()


def test_checkpoint_copies_actual_artifact_and_rejects_pin_changes(inputs, tmp_path):
    output, digest, _, spec = inputs
    day = "2014-10-31"
    source = output / "features" / (day + ".parquet")
    origin = tmp_path / "interrupted"
    origin.mkdir()
    import shutil

    shutil.copyfile(source, origin / "actual-features.parquet")
    pins = {"prepared_sha256": digest, "spec_sha256": development._digest(spec)}
    cell = {
        "signal_session": day,
        "candidate_id": "control_reversal_F07",
        "artifacts": [
            {"path": "actual-features.parquet", "sha256": development.file_sha256(source)}
        ],
    }


    checkpoint = origin / "checkpoints" / "000001.json"
    development._write(checkpoint, {"pins": pins, "complete": False, "cells": [cell]})
    checkpoint_sha = development.file_sha256(checkpoint)
    destination = tmp_path / "continued"
    destination.mkdir()
    assert development.restore_checkpoint(checkpoint, checkpoint_sha, destination, pins) == [cell]
    assert development.file_sha256(
        destination / "actual-features.parquet"
    ) == development.file_sha256(source)
    assert not (destination / "COMPLETE").exists()
    candidates = development.registered_candidates(spec)
    with pytest.raises(ValueError, match="Incomplete registered"):
        development.require_completed_grid([cell], [date.fromisoformat(day)], candidates)
    # Cell identifiers are checkpoint metadata; no market observations are invented.
    complete_ids = [
        {"signal_session": day, "candidate_id": candidate.candidate_id} for candidate in candidates
    ]
    development.require_completed_grid(complete_ids, [date.fromisoformat(day)], candidates)
    with pytest.raises(ValueError, match="pins mismatch"):
        development.restore_checkpoint(
            checkpoint, checkpoint_sha, tmp_path / "refused", {**pins, "spec_sha256": "changed"}
        )
    with pytest.raises(ValueError, match="SHA mismatch"):
        development.restore_checkpoint(checkpoint, "0" * 64, tmp_path / "refused", pins)
    # Verify a hash-linked metadata continuation against the same real bytes.
    linked = origin / "checkpoints" / "000002.json"
    development._write(
        linked,
        {
            "pins": pins,
            "complete": False,
            "cells": [],
            "parent_checkpoint": {"path": "checkpoints/000001.json", "sha256": checkpoint_sha},
        },
    )
    linked_output = tmp_path / "linked"
    linked_output.mkdir()
    assert development.restore_checkpoint(
        linked, development.file_sha256(linked), linked_output, pins
    ) == [cell]
    assert (linked_output / "checkpoints" / "000001.json").exists()
    assert (linked_output / "checkpoints" / "000002.json").exists()


def test_score_units_follow_registered_training_target():
    raw = development.load_spec("configs/stock-development.toml")
    ranked = development.load_spec("configs/stock-rank-target-development.toml")
    for candidate in development.registered_candidates(raw):
        if candidate.kind == "control":
            assert development.prediction_score_kind(candidate, raw) == "ordinal_score"
            assert development.prediction_score_kind(candidate, ranked) == "ordinal_score"
        else:
            assert development.prediction_score_kind(candidate, raw) == "decimal_return_forecast"
            assert (
                development.prediction_score_kind(candidate, ranked)
                == "centered_cross_section_rank_forecast"
            )


def test_actual_2015_partition_branch_is_never_projected(inputs, tmp_path, monkeypatch):
    _, _, schedule, spec = inputs
    source = MarketSnapshot(
        Path(os.environ.get("QUANTLAB_SNAPSHOT", "data/processed/market-v1")),
        SOURCE_MANIFEST_SHA256,
    )
    output = tmp_path / "actual-2015-branch-software-scope"
    receipt = prepare_stock_panel(
        source,
        schedule,
        output,
        start=date(2014, 11, 1),
        end=date(2015, 1, 31),
        scoped_test_assets=("stock:ibm.us",),
    )
    original = development._read_columns
    calls = []

    def observed(path, columns):
        calls.append((Path(path).relative_to(output).as_posix(), columns))
        return original(path, columns)

    monkeypatch.setattr(development, "_read_columns", observed)
    data = development.read_development_inputs(
        output, receipt["manifest_sha256"], spec, schedule, allow_scoped_test=True
    )
    assert (output / "features" / "2015-01-30.parquet").is_file()
    assert "labels/2015-01-30.parquet" in data.outcome_value_partitions_refused
    assert all("2015-" not in path for path, _ in calls)
    assert not data.outcome_value_partitions_read
    assert all("return_value" not in columns for _, columns in calls)
