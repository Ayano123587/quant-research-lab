"""Eight-asset weighting checks on covariance from supplied observed ETF marks."""

import math

import numpy as np
import pytest
import test_real_allocation as real_allocation

from quantlab.allocation.policies import allocate
from quantlab.allocation.weighting import strategic_weights
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256

pytestmark = pytest.mark.real_data
inputs = real_allocation.inputs


def test_observed_strategic_weights_caps_risk_shares_and_optimality(inputs):
    _, schedule, view = inputs
    target = allocate(
        view,
        schedule,
        policy="trend_252_volatility_63",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    estimate = target.covariance
    methods = ("equal_capital", "capped_inverse_volatility", "capped_risk_budget")
    for method in methods:
        result = strategic_weights(estimate, method)
        weights = np.asarray(result.weights)
        assert result.asset_ids == estimate.asset_ids
        assert result.covariance_window_sha256 == estimate.window_hash
        assert abs(math.fsum(weights) - 1) <= 1e-12
        assert weights.min() > 0 and weights.max() <= 0.25 + 1e-10
        risk = weights * (estimate.matrix @ weights)
        np.testing.assert_allclose(result.risk_shares, risk / risk.sum(), atol=1e-13)
        if method == "equal_capital":
            assert result.weights == (0.125,) * 8
        elif method == "capped_inverse_volatility":
            free = weights < 0.25 - 1e-10
            scaled = weights[free] * np.sqrt(np.diag(estimate.matrix))[free]
            np.testing.assert_allclose(scaled, scaled[0], rtol=1e-12)
        else:
            assert result.solver["stationarity_residual"] <= 1e-6
            assert result.solver["complementarity_residual"] <= 1e-7
        # Disabled strategic sleeves become cash, never redistributed to winners.
        masked = weights * np.asarray(target.active)
        assert np.all(masked <= weights)
        assert 1 - math.fsum(masked) >= -1e-12
