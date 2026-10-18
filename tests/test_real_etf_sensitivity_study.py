"""Actual small 2010 ledgers check economics/provenance, never performance ranking."""

import os
from datetime import date
from pathlib import Path

import pytest
import test_real_allocation as real_allocation

from quantlab.allocation.sensitivity import SCENARIOS
from quantlab.allocation.sensitivity_study import (
    authenticated_histories,
    encoded,
    execute_scenario,
    source_closure,
    target_record,
)
from quantlab.allocation.study import compare_controls
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import resolve_source_identity
from quantlab.data.yahoo import YahooETFRelease

pytestmark = pytest.mark.real_data
inputs = real_allocation.inputs


def test_actual_yahoo_release_interface_is_authenticated_before_and_after(monkeypatch):
    root = Path(
        os.environ.get(
            "QUANTLAB_YAHOO_RELEASE", "data/processed/yahoo-etf-price-index-v1"
        )
    )
    if not (root / "COMPLETE").is_file():
        pytest.skip("Approved actual Yahoo release required")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    release = YahooETFRelease(
        root,
        "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3",
        schedule,
    )
    assert not hasattr(release, "verified_read_session")
    original, calls = release.rows, []

    def observed(asset):
        rows = original(asset)
        calls.append(asset)
        return rows

    monkeypatch.setattr(release, "rows", observed)
    with authenticated_histories(release) as histories:
        assert set(histories) == set(release.asset_ids)
        assert all(histories.values())
        assert calls == list(histories)
    assert calls == list(histories) * 2


def test_actual_seven_paths_share_clock_fee_ledger_and_reference(inputs):
    pairs, schedule, view = inputs
    identity = resolve_source_identity(
        schedule.provenance["schedule_sha256"], archive_sha256=SUPPLIED_ARCHIVE_SHA256
    )
    histories = dict(pairs)
    reference, _ = compare_controls(
        histories,
        schedule,
        start_signal=view.session,
        end=date(2010, 2, 3),
        source_identity=identity,
        fee_bps=10,
    )
    assert len(SCENARIOS) == 7
    for scenario in SCENARIOS:
        result, targets = execute_scenario(
            histories,
            schedule,
            identity,
            scenario,
            start=view.session,
            end=date(2010, 2, 3),
            initial_cash=100000.0,
            fee_bps=10,
        )
        assert result.failure is None and result.incomplete_day is None
        assert len(result.daily) == 4 and len(targets) == 1
        assert result.daily[0].execution.fee == 0
        assert all(r.execution.fee == 0 for r in result.daily[2:])
        assert max(abs(r.reconciliation_residual) for r in result.daily) < 1e-7
        assert targets[0].source_target.max_input_available_at <= view.cutoff
        row = target_record(targets[0])
        assert encoded(row)
        assert row["covariance_window_hash"]
        if scenario.scenario_id.startswith("weight_"):
            assert row["strategic_certificate"]["cap"] == 0.25
            assert row["strategic_certificate"]["solver"]
        if scenario == SCENARIOS[0]:
            original = reference["trend_252_volatility_63"]["engine"]
            assert [r.state for r in result.daily] == [r.state for r in original.daily]
            assert [r.execution for r in result.daily] == [
                r.execution for r in original.daily
            ]
    assert "quantlab/allocation/weighting.py" in source_closure()
