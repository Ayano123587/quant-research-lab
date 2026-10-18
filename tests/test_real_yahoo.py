"""Whole authentic provider capture; explicit approved deterministic engineering."""

import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, file_sha256
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.data.yahoo import (
    POLICY,
    SYMBOLS,
    YahooETFRelease,
    YahooSourceError,
    build_yahoo_etf_release,
)
from quantlab.risk.covariance import ReturnWindow, estimate_covariance

pytestmark = pytest.mark.real_data
RAW_MANIFEST_SHA256 = "c7cc178eb60b3674388deac4b84315d6776ed66800ba5f7e5418cbf6f736c240"


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    raw = Path(os.environ.get("QUANTLAB_YAHOO_RAW_BUNDLE", "data/raw/yahoo-etf-v1"))
    if not (raw / "raw-manifest.json").is_file():
        pytest.skip("Actual externally pinned eight-ETF Yahoo capture absent")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    output = tmp_path_factory.mktemp("actual-yahoo-release") / "whole-eight-etfs"
    receipt = build_yahoo_etf_release(raw, RAW_MANIFEST_SHA256, schedule, output)
    release = YahooETFRelease(output, receipt["manifest_sha256"], schedule)
    rows = {asset: release.rows(asset) for asset in release.asset_ids}
    return raw, output, receipt, release, schedule, rows


def test_distinct_source_type_and_full_exact_warmup_coverage(actual):
    _, _, _, release, schedule, records = actual
    assert not isinstance(release, MarketSnapshot)
    assert not hasattr(release, "archive_sha256")
    assert release.source_bundle_sha256 == RAW_MANIFEST_SHA256 != SUPPLIED_ARCHIVE_SHA256
    expected = tuple(
        d.isoformat() for d in schedule.sessions if date(2006, 2, 27) <= d <= date(2017, 11, 10)
    )
    for asset, rows in records.items():
        dates = {row.session for row in rows}
        assert set(expected) <= dates
        assert "2011-02-17" in dates
        assert len(dates) == len(rows)
        assert all(row.asset_id == asset for row in rows)
    assert records["etf:dbc.us"][0].session == "2006-02-06"


def test_every_derived_cell_uses_observed_same_row_ratio_without_rounding_or_fill(actual):
    raw_root, _, _, _, _, records = actual
    nonzero_roundoff = 0
    for symbol in SYMBOLS:
        data = json.loads((raw_root / (symbol + "-chart-raw.json")).read_bytes())["chart"][
            "result"
        ][0]
        quote = data["indicators"]["quote"][0]
        adjusted = data["indicators"]["adjclose"][0]["adjclose"]
        for row in records["etf:" + symbol.lower() + ".us"]:
            i = row.raw_array_index
            factor = adjusted[i] / quote["close"][i]
            assert row.timestamp == data["timestamp"][i]
            assert row.published_adjclose == adjusted[i]
            for field in ("open", "high", "low", "close"):
                assert getattr(row, "observed_" + field) == quote[field][i]
                assert getattr(row, field) == quote[field][i] * factor
                assert math.isfinite(getattr(row, field)) and getattr(row, field) > 0
            assert row.volume == row.observed_volume == quote["volume"][i]
            assert abs(row.close - row.published_adjclose) <= 8 * math.ulp(
                max(row.close, row.published_adjclose)
            )
            nonzero_roundoff += row.close_roundoff != 0
    assert (
        nonzero_roundoff > 0
    )  # actual float residuals are recorded, never hidden by replacing close


def test_raw_bytes_events_and_docs_preserved_exactly_with_derived_identities(actual):
    raw_root, output, _, release, _, records = actual
    capture = json.loads((raw_root / "raw-manifest.json").read_bytes())
    for symbol in SYMBOLS:
        source = raw_root / (symbol + "-chart-raw.json")
        assert file_sha256(source) == file_sha256(output / "raw" / (symbol + ".json"))
        data = json.loads(source.read_bytes())["chart"]["result"][0]
        assert json.loads((output / "actions" / (symbol + ".json")).read_bytes()) == data.get(
            "events", {}
        )
        row = next(r for r in records["etf:" + symbol.lower() + ".us"] if r.session == "2011-02-17")
        assert row.raw_response_sha256 == capture["files"][symbol]["sha256"]
        assert row.raw_row_sha256 != row.derived_row_sha256
        bar = next(b for b in release.bars("etf", symbol) if b.session == date(2011, 2, 17))
        assert (
            bar.row_sha256 == row.derived_row_sha256 and bar.row_number == row.raw_array_index + 1
        )
        assert math.isnan(bar.open_interest)  # Yahoo did not publish this field; never invent zero
    for name, item in capture["primary_documents"].items():
        assert file_sha256(output / "primary-documents" / name) == item["sha256"]
    assert json.loads((output / "policy.json").read_bytes()) == POLICY


def test_real_price_index_covariance_keeps_daily_units_order_and_signal_availability(actual):
    _, _, _, release, schedule, records = actual
    signal = date(2010, 6, 30)
    position = schedule.sessions.index(signal)
    dates = schedule.sessions[position - 63 : position + 1]
    assets = release.asset_ids
    prices = np.asarray(
        [
            [next(r.close for r in records[asset] if r.session == str(day)) for asset in assets]
            for day in dates
        ]
    )
    values = prices[1:] / prices[:-1] - 1
    window = ReturnWindow(
        values,
        assets,
        tuple(schedule.session(d).close_at for d in dates[:-1]),
        tuple(schedule.session(d).close_at for d in dates[1:]),
        tuple(schedule.session(d).decision_at for d in dates[1:]),
        schedule.session(signal).decision_at,
        release.manifest_sha256,
        "published_adjclose_ratio_simple_daily_price_index_return",
        "actual_complete_63_intervals",
        release.manifest_sha256,
        release.provenance,
    )
    estimate = estimate_covariance(window)
    assert estimate.asset_ids == assets
    assert estimate.max_available_at == schedule.session(signal).decision_at
    assert estimate.units == "decimal_return_squared_per_session"
    assert estimate.source_provenance["raw_bundle_manifest_sha256"] == RAW_MANIFEST_SHA256
    assert SUPPLIED_ARCHIVE_SHA256 not in estimate.source_provenance.values()
    np.testing.assert_allclose(
        estimate.matrix, np.cov(values, rowvar=False, ddof=1), rtol=1e-12, atol=1e-18
    )


def test_wrong_external_pins_and_actual_different_calendar_fail_before_consumption(
    actual, tmp_path
):
    raw_root, output, receipt, _, _, _ = actual
    unrelated_actual_sha = file_sha256(Path("src/quantlab/data/yahoo.py"))
    with pytest.raises(YahooSourceError, match="SHA mismatch"):
        build_yahoo_etf_release(
            raw_root,
            unrelated_actual_sha,
            XNYSSchedule(date(2005, 1, 1), date(2018, 12, 31)),
            tmp_path / "no-release",
        )
    assert not (tmp_path / "no-release").exists()
    with pytest.raises(YahooSourceError, match="Completion marker"):
        YahooETFRelease(
            output, unrelated_actual_sha, XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
        )
    with pytest.raises(YahooSourceError, match="calendar mismatch"):
        YahooETFRelease(
            output, receipt["manifest_sha256"], XNYSSchedule(date(2005, 1, 1), date(2018, 12, 31))
        )
