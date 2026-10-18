"""Short actual-source fee-funding checks; full study runs only in cloud."""

import os
import time
from datetime import date
from pathlib import Path

import pytest

from quantlab.allocation.policies import POLICIES
from quantlab.allocation.robustness import COSTS_BPS, compare_costs, run_cost_study
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import identity_from_source
from quantlab.data.yahoo import YahooETFRelease

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def actual_inputs():
    path = Path(os.environ.get("QUANTLAB_YAHOO_RELEASE", "data/processed/yahoo-etf-price-index-v1"))
    if not (path / "COMPLETE").is_file():
        pytest.skip("Approved complete actual Yahoo ETF release required")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    release = YahooETFRelease(
        path, "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3", schedule
    )
    histories = {asset: release.bars("etf", asset.split(":")[1]) for asset in release.asset_ids}
    return histories, schedule, identity_from_source(release, schedule)


@pytest.fixture(scope="module")
def actual_costs(actual_inputs):
    histories, schedule, identity = actual_inputs
    return compare_costs(
        histories,
        schedule,
        source_identity=identity,
        start_signal=date(2010, 6, 30),
        end=date(2010, 7, 9),
        service_deadline_monotonic=time.monotonic() + 120,
    )


def test_actual_costs_share_full_targets_and_ownership_support(actual_costs):
    assert tuple(actual_costs) == COSTS_BPS
    for policy in POLICIES:
        reference = actual_costs[0.0][policy]
        for cost in COSTS_BPS:
            result = actual_costs[cost][policy]
            assert result["targets"] == reference["targets"]
            assert len(result["measured"]) == 6
            assert result["attribution"]["net_pnl"] == pytest.approx(
                result["engine"].endpoint.nav - 100_000
            )
            assert result["attribution"]["fee_pnl"] == -result["metrics"]["total_fee_money"]
            assert result["engine"].failure is None


def test_fees_fund_actual_initial_purchase_not_cagr_subtraction(actual_costs):
    navs = []
    for cost in COSTS_BPS:
        result = actual_costs[cost]["strategic_equal"]
        purchase = result["measured"][0].execution
        expected_fee = 100_000 * (cost / 10_000) / (1 + cost / 10_000)
        assert purchase.fee == pytest.approx(expected_fee, abs=1e-9)
        assert purchase.cash_after == 0
        navs.append(result["engine"].endpoint.nav)
    assert all(a > b for a, b in zip(navs, navs[1:], strict=False))


def test_expired_service_budget_stops_before_any_full_study(actual_inputs, tmp_path):
    histories, schedule, identity = actual_inputs
    with pytest.raises(ValueError, match="deadline"):
        run_cost_study(
            histories,
            schedule,
            source_identity=identity,
            output=tmp_path / "refused",
            service_deadline_monotonic=time.monotonic() - 1,
        )
    assert not (tmp_path / "refused").exists()
