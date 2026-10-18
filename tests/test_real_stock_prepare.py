"""Explicit scoped software checks on real 2010 source histories, never a study."""

import json
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.features import FEATURE_IDS, build_candidates, select_cross_section
from quantlab.stocks.labels import label_from_history
from quantlab.stocks.prepare import (
    SOURCE_MANIFEST_SHA256,
    _ActualCalendarCache,
    prepare_stock_panel,
    verify_prepared_panel,
)

pytestmark = pytest.mark.real_data
TEST_ASSETS = ("stock:ibm.us", "stock:ge.us", "stock:aapl.us", "stock:tsla.us", "stock:accp.us")


@pytest.fixture(scope="module")
def inputs():
    root = Path(os.environ.get("QUANTLAB_SNAPSHOT", "data/processed/market-v1"))
    if not (root / "COMPLETE").exists():
        pytest.skip("Reviewed complete real snapshot absent; set QUANTLAB_SNAPSHOT")
    source = MarketSnapshot(root, SOURCE_MANIFEST_SHA256)
    calendar = XNYSSchedule(date(2008, 1, 1), date(2018, 12, 31))
    return source, calendar


@pytest.fixture(scope="module")
def prepared(inputs, tmp_path_factory):
    source, calendar = inputs
    output = tmp_path_factory.mktemp("real-stock-prepare") / "explicit-software-check"
    manifest = prepare_stock_panel(
        source,
        calendar,
        output,
        start=date(2010, 1, 1),
        end=date(2010, 2, 28),
        top_n=2,
        scoped_test_assets=TEST_ASSETS,
    )
    return output, manifest


def test_real_preparation_is_explicitly_scoped_and_cannot_be_consumed_as_full_research(prepared):
    output, manifest = prepared
    assert manifest["research_scope_complete"] is False
    assert manifest["scope"] == "scoped_software_check_not_research_panel"
    assert manifest["counts"]["source_histories_read"] == 5
    assert manifest["counts"]["empty_source_histories"] == 1
    assert manifest["counts"]["candidates_rows"] == 10
    assert manifest["counts"]["labels_rows"] == 10
    assert manifest["counts"]["features_rows"] == 4
    with pytest.raises(ValueError, match="not a full research panel"):
        verify_prepared_panel(output, manifest["manifest_sha256"])
    verified = verify_prepared_panel(output, manifest["manifest_sha256"], allow_scoped_test=True)
    assert verified["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    journal = [
        json.loads(line) for line in (output / "source-inputs.jsonl").read_text().splitlines()
    ]
    assert {r["asset_id"] for r in journal} == set(TEST_ASSETS)
    assert next(r for r in journal if r["asset_id"] == "stock:accp.us")["rows"] == 0


def test_real_streamed_features_match_primitives_and_labels_never_screen_candidates(
    inputs, prepared
):
    import pyarrow.parquet as pq

    source, calendar = inputs
    output, _ = prepared
    histories = {asset: source.bars("stock", asset.split(":")[1]) for asset in TEST_ASSETS}
    for signal in (date(2010, 1, 29), date(2010, 2, 26)):
        candidates = build_candidates(
            histories, calendar, as_of=signal, archive_sha256=source.archive_sha256
        )
        expected = select_cross_section(candidates, top_n=2)
        rows = pq.ParquetFile(output / f"features/{signal}.parquet").read().to_pylist()
        assert tuple(r["asset_id"] for r in rows) == expected.selected_asset_ids
        for row, actual in zip(rows, expected.rows, strict=True):
            assert row["activity_proxy"] == actual.activity_proxy
            for feature in FEATURE_IDS:
                assert row["raw_" + feature] == dict(actual.raw)[feature]
                assert row["rank_" + feature] == dict(actual.normalized)[feature]
        audit = pq.ParquetFile(output / f"candidates/{signal}.parquet").read().to_pylist()
        assert {r["asset_id"] for r in audit} == set(TEST_ASSETS)
        for asset in ("stock:accp.us", "stock:tsla.us"):
            rejected = next(r for r in audit if r["asset_id"] == asset)
            assert not rejected["eligible"] and rejected["observed_scheduled_marks"] == 0
            assert "MISSING_LATEST_MARK" in rejected["reasons"]
        labels = pq.ParquetFile(output / f"labels/{signal}.parquet").read().to_pylist()
        assert {r["asset_id"] for r in labels} == set(TEST_ASSETS)
        ibm = next(r for r in labels if r["asset_id"] == "stock:ibm.us")
        direct = label_from_history(
            "stock:ibm.us",
            histories["stock:ibm.us"],
            calendar,
            signal_session=signal,
            archive_endpoint=date(2017, 11, 10),
            archive_sha256=source.archive_sha256,
        )
        assert ibm["return_value"] == direct.return_value
        assert ibm["label_available_at"] == direct.label_available_at > ibm["signal_at"]
        assert "return_value" not in rows[0] and "label_available_at" not in rows[0]


def test_real_preparation_refuses_overwrite_or_source_destination(inputs, prepared):
    source, calendar = inputs
    output, manifest = prepared
    with pytest.raises(FileExistsError):
        prepare_stock_panel(
            source,
            calendar,
            output,
            start=date(2010, 1, 1),
            end=date(2010, 2, 28),
            scoped_test_assets=TEST_ASSETS,
        )
    assert verify_prepared_panel(output, manifest["manifest_sha256"], allow_scoped_test=True)
    with pytest.raises(ValueError, match="outside immutable source"):
        prepare_stock_panel(
            source,
            calendar,
            source.root / "forbidden-preparation",
            start=date(2010, 1, 1),
            end=date(2010, 2, 28),
            scoped_test_assets=TEST_ASSETS,
        )


def test_cache_preserves_actual_2010_calendar_and_tuple_index_semantics(inputs):
    _, calendar = inputs
    cached = _ActualCalendarCache(calendar)
    assert isinstance(cached, XNYSSchedule)
    assert isinstance(cached.sessions, tuple) and cached.sessions == calendar.sessions
    assert cached.provenance is calendar.provenance
    for month in range(1, 13):
        actual = calendar.month_end(2010, month)
        assert cached.month_end(2010, month) is actual
        assert cached.month_end(2010, month) is actual
    for day in calendar.sessions:
        if day.year == 2010:
            assert cached.session(day) is calendar.session(day)
            assert cached.next_session(day) is calendar.next_session(day)
    cases = (
        (date(2010, 1, 4), 0, len(calendar.sessions)),
        (date(2010, 1, 1), 0, len(calendar.sessions)),
        (calendar.sessions[0], 0, 1),
        (calendar.sessions[0], 1, len(calendar.sessions)),
        (calendar.sessions[-1], -1, len(calendar.sessions)),
        (calendar.sessions[-1], -2, -1),
        (calendar.sessions[0], 10, 2),
        (calendar.sessions[-1], 0, 10**30),
        ("not_a_session", -(10**30), 10**30),
    )
    for value, start, stop in cases:
        try:
            expected = calendar.sessions.index(value, start, stop)
        except ValueError:
            with pytest.raises(ValueError):
                cached.sessions.index(value, start, stop)
        else:
            assert cached.sessions.index(value, start, stop) == expected


def test_failure_keeps_real_partial_artifacts_without_consumable_completion(
    inputs, tmp_path, monkeypatch
):
    from quantlab.stocks import prepare as preparation_module

    source, calendar = inputs
    output = tmp_path / "failed-scoped-software-check"

    def software_failure(rows, top_n):
        assert rows and rows[0]["asset_id"] == "stock:ibm.us"
        raise RuntimeError("software failure after actual source preparation")

    # Inject a software exception only; the actual market records are unchanged.
    monkeypatch.setattr(preparation_module, "_rank_rows", software_failure)
    with pytest.raises(RuntimeError, match="software failure"):
        prepare_stock_panel(
            source,
            calendar,
            output,
            start=date(2010, 1, 1),
            end=date(2010, 1, 31),
            scoped_test_assets=("stock:ibm.us",),
        )
    assert (output / "FAILURE.json").is_file()
    assert (output / "candidates/2010-01-29.parquet").is_file()
    assert not (output / "COMPLETE").exists()
    with pytest.raises(ValueError, match="Completed error-free"):
        verify_prepared_panel(output, "0" * 64, allow_scoped_test=True)


def test_actual_read_session_postflight_failure_prevents_monthly_finalization(
    inputs, tmp_path, monkeypatch
):
    source, calendar = inputs
    actual_session = source.verified_read_session
    output = tmp_path / "postflight-failed-scoped-check"
    actual_postflight_completed = []

    @contextmanager
    def software_postflight_failure():
        # Delegate all real input and postflight checks; inject only a software
        # failure afterward. No source observation or snapshot file is changed.
        with actual_session() as verified:
            assert verified is source
            yield verified
        actual_postflight_completed.append(True)
        raise RuntimeError("software failure at verified input postflight")

    monkeypatch.setattr(source, "verified_read_session", software_postflight_failure)
    with pytest.raises(RuntimeError, match="verified input postflight"):
        prepare_stock_panel(
            source,
            calendar,
            output,
            start=date(2010, 1, 1),
            end=date(2010, 1, 31),
            scoped_test_assets=("stock:ibm.us",),
        )
    assert actual_postflight_completed == [True]
    failure = json.loads((output / "FAILURE.json").read_bytes())
    assert failure["stage"] == "source_read_session_postflight"
    assert failure["counts"]["source_histories_read"] == 1
    assert (output / "candidates/2010-01-29.parquet").is_file()
    assert not (output / "features").exists()
    assert not (output / "COMPLETE").exists()
    with pytest.raises(ValueError, match="Completed error-free"):
        verify_prepared_panel(output, "0" * 64, allow_scoped_test=True)
