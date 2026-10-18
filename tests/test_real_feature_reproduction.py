"""Portability proposal on authenticated saved cloud features; no fabricated prices."""

import hashlib
import json
import math
import os
import struct
from datetime import date
from pathlib import Path

import pytest

from quantlab.stocks.feature_reproduction import (
    DIAGNOSTIC_SHA256,
    INITIAL_DIAGNOSTIC_SHA256,
    PARENT_MANIFEST_SHA256,
    POLICY,
    SIX,
    _whitelist,
    reproduce_raw_six,
)
from quantlab.stocks.wider_panel import _same


@pytest.fixture(scope="module")
def full_actual():
    default = (
        Path(__file__).resolve().parents[2]
        / "quant-research-lab-private"
        / "stock-wider-A-diagnostic-publication-v1/observations/20261018T000454704697Z"
        / "a-diagnostic/report.json"
    )
    p = Path(os.environ.get("QUANTLAB_A_DIAGNOSTIC", str(default)))
    if not p.is_file():
        pytest.skip("Authenticated actual cloud diagnostic absent; no fake fallback")
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == INITIAL_DIAGNOSTIC_SHA256
    report = json.loads(raw)
    assert report["selected_dates_checked"] == 150 and report["fits"] == 0
    assert report["parent_prepared_manifest_sha256"] == PARENT_MANIFEST_SHA256
    assert report["first_failure"]["date"] == "2010-11-30"
    return report


@pytest.fixture(scope="module")
def actual():
    p = Path(__file__).parent / "fixtures/actual_A_feature_reproduction_2010-11-30.json"
    raw = p.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "52f4a90d4a64fc0216776653a191fa96579248860e6d03f17d757abe0cb68e20"
    )
    record = json.loads(raw)
    assert record["source_report_sha256"] == INITIAL_DIAGNOSTIC_SHA256
    assert record["parent_prepared_manifest_sha256"] == PARENT_MANIFEST_SHA256
    return {"first_failure": record["row"]}


def compare(row, *, policy=None, computed_platform="Linux"):
    return reproduce_raw_six(
        tuple((f, float(row["checks"][f]["parent"])) for f in SIX),
        tuple((f, float(row["checks"][f]["actual"])) for f in SIX),
        asset_id="stock:a.us",
        signal_session=date.fromisoformat(row["date"]),
        parent_prepared_manifest_sha256=PARENT_MANIFEST_SHA256,
        source_context=_whitelist().get(("stock:a.us", row["date"])),
        policy=policy,
        parent_platform="Windows",
        computed_platform=computed_platform,
        parent_python="3.12.15",
        computed_python="3.12.15",
    )


def test_actual_2010_A_exact_default_still_refuses_one_ulp(actual):
    row = actual["first_failure"]
    parent = row["checks"]["F09"]["parent"]
    computed = row["checks"]["F09"]["actual"]
    assert parent > 0 and computed > 0 and math.nextafter(parent, computed) == computed
    assert not _same(parent, computed)
    with pytest.raises(ValueError, match="F09 failed exact audited reproduction"):
        compare(row)


def test_actual_all150_parent_bytes_retained_and_one_drift_recorded(full_actual):
    actual = full_actual
    drifts = []
    for row in actual["all_date_results"]:
        result = compare(row, policy=POLICY)
        for f, value in result.retained_parent_features:
            assert struct.pack("<d", value).hex() == row["checks"][f]["parent_float64_hex"]
        drifts.extend(result.accepted_differences)
    assert len(drifts) == 1
    drift = drifts[0]
    assert drift.signal_session == date(2010, 11, 30) and drift.feature_id == "F09"
    assert drift.parent_float64_hex == "873bb6eebce3d03f"
    assert drift.computed_float64_hex == "883bb6eebce3d03f"
    assert drift.adjacent_float64_steps == 1 and drift.diagnostic_sha256 == DIAGNOSTIC_SHA256


def test_policy_is_explicit_immutable_and_runtime_scoped(actual):
    with pytest.raises(TypeError):
        POLICY["feature"] = "F19"
    with pytest.raises(ValueError, match="runtime provenance"):
        compare(actual["first_failure"], policy=POLICY, computed_platform="Windows")
    with pytest.raises(ValueError, match="explicitly frozen"):
        compare(actual["first_failure"], policy={})


def test_portable_actual_fixture_retains_parent_bytes_and_feature_identity(actual):
    row = actual["first_failure"]
    result = compare(row, policy=POLICY)
    assert len(result.accepted_differences) == 1
    assert result.accepted_differences[0].feature_id == "F09"
    assert result.accepted_differences[0].diagnostic_sha256 == DIAGNOSTIC_SHA256
    for f, value in result.retained_parent_features:
        assert struct.pack("<d", value).hex() == row["checks"][f]["parent_float64_hex"]
    # Bad feature identity/order is rejected using unchanged observed values.
    with pytest.raises(ValueError, match="ordered original six"):
        reproduce_raw_six(
            result.retained_parent_features,
            tuple(reversed(result.retained_parent_features)),
            asset_id="stock:a.us",
            signal_session=date(2010, 11, 30),
            parent_prepared_manifest_sha256=PARENT_MANIFEST_SHA256,
            policy=POLICY,
            parent_platform="Windows",
            computed_platform="Linux",
            parent_python="3.12.15",
            computed_python="3.12.15",
        )
