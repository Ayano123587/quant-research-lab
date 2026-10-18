"""Signal-only sensitivity arithmetic using the supplied real 2010 ETF prices."""

import math

import numpy as np
import pytest
import test_real_allocation as real_allocation

from quantlab.allocation.sensitivity import SCENARIOS, sensitivity_target
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256
from quantlab.data.provenance import resolve_source_identity

pytestmark = pytest.mark.real_data
inputs = real_allocation.inputs


def targets(inputs):
    _, schedule, view = inputs
    identity = resolve_source_identity(
        schedule.provenance["schedule_sha256"], archive_sha256=SUPPLIED_ARCHIVE_SHA256
    )
    return [sensitivity_target(view, schedule, identity, s) for s in SCENARIOS]


def test_real_target_response_preserves_inputs_and_respects_cap(inputs):
    reference, lower, higher, *_ = targets(inputs)
    assert lower.active == reference.active == higher.active
    assert lower.forecast_before == reference.forecast_before == higher.forecast_before
    assert (
        lower.source_target.covariance.window_hash == reference.source_target.covariance.window_hash
    )
    assert lower.scale <= reference.scale <= higher.scale <= 1
    for target in (lower, reference, higher):
        np.testing.assert_array_equal(
            target.weights,
            [0.125 * target.scale if enabled else 0 for enabled in target.active],
        )
        assert target.cash_weight == 1 - math.fsum(target.weights)
        assert target.forecast_after <= target.scenario.annual_target + 1e-14
    assert reference.weights == reference.source_target.weights


def test_real_covariance_changes_preserve_clock_trend_and_budget(inputs):
    result = targets(inputs)
    reference = result[0]
    for alternative in result[3:5]:
        assert alternative.active == reference.active
        assert alternative.strategic_weights == reference.strategic_weights
        assert alternative.decision.signal_at == reference.decision.signal_at
        assert (
            alternative.decision.intended_execution_at == reference.decision.intended_execution_at
        )
        assert alternative.scenario.annual_target == reference.scenario.annual_target
        covariance = alternative.source_target.covariance
        base = np.array([0.125 if enabled else 0 for enabled in alternative.active])
        expected = np.sqrt(252 * base @ covariance.matrix @ base)
        assert alternative.forecast_before == pytest.approx(expected, rel=1e-14)
        assert covariance.diagnostics["n_observations"] == 63
        assert covariance.max_available_at <= alternative.decision.signal_at


def test_weight_arms_apply_trend_after_strategic_budgets(inputs):
    result = targets(inputs)
    for target in result[5:]:
        assert target.active == result[0].active
        assert (
            target.source_target.covariance.window_hash
            == result[0].source_target.covariance.window_hash
        )
        assert target.strategic_certificate is not None
        assert target.strategic_certificate.cap == 0.25
        assert math.fsum(target.strategic_weights) <= 1
        assert target.weights == tuple(
            weight * target.scale if active else 0
            for weight, active in zip(target.strategic_weights, target.active, strict=True)
        )
        assert target.cash_weight == 1 - math.fsum(target.weights)
        assert target.cash_weight >= 0
        assert target.forecast_after <= 0.1 + 1e-14
