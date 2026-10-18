"""Wider formulas on actual253-row2010 stock windows; no invented market fixtures."""

import math
import struct
from datetime import date
from statistics import correlation, fmean, median, stdev

import pytest
import test_real_stock_features as real_six

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256
from quantlab.stocks.features import FEATURE_IDS as SIX
from quantlab.stocks.features import build_candidates, select_cross_section
from quantlab.stocks.prepare_wider import ARMS, extend_cross_section
from quantlab.stocks.wider_features import BASIS_GATED, FEATURE_IDS, wider_candidate_from_history

inputs = real_six.inputs
pytestmark = pytest.mark.real_data
DAY = date(2010, 1, 29)


def actual_candidate(inputs, asset="stock:ibm.us"):
    _, histories, schedule = inputs
    return wider_candidate_from_history(
        asset, histories[asset], schedule, as_of=DAY, archive_sha256=SUPPLIED_ARCHIVE_SHA256
    )


def test_actual21_formulas_and_four_explicit_source_basis_gates(inputs):
    result = actual_candidate(inputs)
    assert result.original.eligible and len(result.original.source_rows) == 253
    rows = result.original.source_rows
    c = [r.close for r in rows]
    logs = [math.log(c[i] / c[i - 1]) for i in range(190, 253)]
    v = [r.volume for r in rows]
    centered = [x - fmean(logs) for x in logs]
    m2 = fmean(x * x for x in centered)
    log_prices = [math.log(x) for x in c[-63:]]
    sign = math.copysign(1, correlation(range(63), log_prices))
    expected = {
        "F02": c[231] / c[126] - 1,
        "F03": c[247] / c[189] - 1,
        "F04": c[252] / c[189] - 1,
        "F05": (c[252] / c[189] - 1) - (c[189] / c[126] - 1),
        "F06": -(c[252] / c[247] - 1),
        "F08": math.sqrt(252) * stdev(logs[-21:]),
        "F10": math.sqrt(4 * sum(min(x, 0) ** 2 for x in logs)),
        "F11": stdev(logs[-21:]) / stdev(logs) - 1,
        "F12": fmean(x**3 for x in centered) / (m2**1.5),
        "F14": c[252] / fmean(c[1:253]) - 1,
        "F16": c[252] / max(c[190:253]) - 1,
        "F17": sign * correlation(range(63), log_prices) ** 2,
        "F18": median(v[248:253]) / median(v[190:253]) - 1,
        "F20": v[252] / median(v[189:252]) - 1,
        "F21": sum(x == 0 for x in v[190:253]) / 63,
    }
    observed = {x.feature_id: x for x in result.values}
    assert tuple(observed) == FEATURE_IDS
    for feature, value in expected.items():
        assert observed[feature].value == pytest.approx(value, rel=1e-11, abs=1e-13)
        assert observed[feature].missing_reason is None
    for feature, value in result.original.features:
        assert struct.pack("<d", observed[feature].value) == struct.pack("<d", value)
    for feature in BASIS_GATED:
        assert observed[feature].value is None
        assert observed[feature].missing_reason == "VERIFIED_SOURCE_BASIS_PROOF_REQUIRED"
    assert "not_capacity" in observed["F18"].source_convention
    assert result.zero_denominator_tolerance == 0


def test_actual_future_prefix_equality_and_observed_inception_exclusion(inputs):
    _, histories, schedule = inputs
    for asset, history in histories.items():
        full = wider_candidate_from_history(
            asset, history, schedule, as_of=DAY, archive_sha256=SUPPLIED_ARCHIVE_SHA256
        )
        prefix = wider_candidate_from_history(
            asset,
            tuple(b for b in history if b.session <= DAY),
            schedule,
            as_of=DAY,
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        )
        assert full == prefix
        assert all(b.session <= DAY for b in full.original.source_rows)
    early = actual_candidate(inputs, "stock:tsla.us")
    assert not early.original.eligible
    assert all(
        x.value is None and x.missing_reason == "ORIGINAL_REQUIRED_READINESS_FAILED"
        for x in early.values
    )


def test_actual_scoped_cohort_and_six_bitwise_projection_are_unchanged(inputs):
    _, histories, schedule = inputs
    original = select_cross_section(
        build_candidates(histories, schedule, as_of=DAY, archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    )
    result = extend_cross_section(original, software_test_scope=True)
    assert result.original == original and result.software_test_scope
    assert result.declared_parent_prepared_manifest_sha256 is None
    assert {r.asset_id for r in result.rows} == set(original.selected_asset_ids)
    assert len(result.rows) == 3  # Explicit real software scope, not a published subset.
    projection = result.project("fresh_six_reference")
    assert projection.column_ids == SIX
    source = {r.asset_id: dict(r.normalized) for r in original.rows}
    for asset, values in zip(projection.asset_ids, projection.rows, strict=True):
        assert tuple(struct.pack("<d", x) for x in values) == tuple(
            struct.pack("<d", source[asset][f]) for f in SIX
        )
    for feature in FEATURE_IDS[:21]:
        if feature in SIX:
            continue
        available = {
            r.asset_id: next(v.value for v in r.candidate.values if v.feature_id == feature)
            for r in result.rows
        }
        valid = [v for v in available.values() if v is not None]
        for row in result.rows:
            value = available[row.asset_id]
            if value is None:
                assert dict(row.normalized)[feature] is None
                continue
            average_rank = sum(v < value for v in valid) + (sum(v == value for v in valid) + 1) / 2
            expected = (average_rank - 0.5) / len(valid) - 0.5 if len(valid) > 1 else 0.0
            assert dict(row.normalized)[feature] == expected
    assert len(result.project("add_five_union_eleven").column_ids) == 16
    assert len(result.project("documented_compact_ten").column_ids) == 15
    full = result.project("all_supported_twenty_one")
    assert len(full.feature_ids) == 21 and len(full.column_ids) == 36
    assert not set(BASIS_GATED) & set(full.feature_ids)
    assert not result.training_authorized
    assert not result.fresh_reference_score_reproduction_verified
    assert len(result.original_five_factor_control) == 5
    assert len(result.separate_blueprint_three_factor_control) == 3
    assert tuple(a for a, _ in ARMS) == (
        "fresh_six_reference",
        "add_five_union_eleven",
        "documented_compact_ten",
        "all_supported_twenty_one",
    )
    with pytest.raises(ValueError, match="four prospectively"):
        result.project("unregistered_feature_search")
    with pytest.raises(ValueError, match="upstream prepared manifest"):
        extend_cross_section(original)
