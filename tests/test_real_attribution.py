"""Small real-source ownership checks, without fitting a strategy or full study."""

import math
import os
from datetime import date
from pathlib import Path

import pytest

from quantlab.backtest.engine import Decision, run_history
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import identity_from_source
from quantlab.data.yahoo import YahooETFRelease
from quantlab.evaluation.attribution import monetary_contributions

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def actual_ownership():
    path = Path(os.environ.get("QUANTLAB_YAHOO_RELEASE", "data/processed/yahoo-etf-price-index-v1"))
    if not (path / "COMPLETE").is_file():
        pytest.skip("The approved whole actual Yahoo release is required")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    release = YahooETFRelease(
        path, "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3", schedule
    )
    bars = {
        asset: release.bars("etf", asset.split(":")[1]) for asset in ("etf:spy.us", "etf:tlt.us")
    }

    def decision(view, _):
        target = (
            {"etf:spy.us": 0.5, "etf:tlt.us": 0.25}
            if view.session == date(2010, 6, 30)
            else ({"etf:tlt.us": 0.75} if view.session == date(2010, 7, 6) else None)
        )
        return Decision(
            "REBALANCE" if target is not None else "HOLD",
            target,
            view.cutoff,
            view.next_open_at,
            "short_real_ownership_check",
        )

    result = run_history(
        bars,
        schedule,
        start=date(2010, 6, 30),
        end=date(2010, 7, 9),
        initial_cash=100_000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=decision,
        source_identity=identity_from_source(release, schedule),
    )
    assert result.failure is None and len(result.daily) == 7
    return result


def test_real_asset_cash_fee_contributions_reconcile(actual_ownership):
    result = actual_ownership
    report = monetary_contributions(result.daily, initial_nav=100_000)
    assert report["net_pnl"] == pytest.approx(result.endpoint.nav - 100_000)
    assert report["fee_pnl"] == -math.fsum(row.execution.fee for row in result.daily)
    assert report["cash_pnl"] == 0
    assert math.fsum(report["initial_nav_contributions"].values()) == pytest.approx(
        result.endpoint.nav / 100_000 - 1, abs=1e-14
    )
    sold_day = next(row for row in report["daily"] if row["session"] == "2010-07-07")
    assert sold_day["asset_pnl"]["etf:spy.us"] != 0  # Overnight ownership before sale remains.


def test_missing_held_interval_or_wrong_start_nav_is_refused(actual_ownership):
    rows = actual_ownership.daily
    with pytest.raises(ValueError, match="ownership chain"):
        monetary_contributions(rows[:3] + rows[4:], initial_nav=100_000)
    with pytest.raises(ValueError, match="ownership chain"):
        monetary_contributions(rows, initial_nav=100_001)
