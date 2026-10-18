"""Actual calendar, frozen parent and prospective diagnostic metadata only."""

import hashlib
import json
from pathlib import Path

import pytest
import test_real_source_integration as actual_source

from cloud import etf_refined_target_diagnostic_runner as runner
from cloud import stage_etf_refined_target_diagnostic as stage

source = actual_source.source
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


def test_actual95_calendar_dates_without_target_loop(source):
    days = runner.diagnostic_signal_dates(source[1])
    assert len(days) == 95 and str(days[0]) == "2007-02-28" and str(days[-1]) == "2014-12-31"
    assert all(day == source[1].month_end(day.year, day.month).session for day in days)


def test_actual_parent_has_exactly_one_reviewed_source_amendment():
    historical_root = PRIVATE / 'etf-sensitivity-cloud-stage-v2/payload'
    if not historical_root.is_dir():
        pytest.skip("Actual frozen historical source package is unavailable")
    historical_plan = json.loads((historical_root / 'etf-sensitivity-plan.json').read_bytes())
    historical_sources = {
        item["path"]: item
        for item in historical_plan["package_files"].values()
        if item["path"].startswith("src/")
    }
    for flat, item in historical_plan["package_files"].items():
        assert hashlib.sha256((historical_root / flat).read_bytes()).hexdigest() == item["sha256"]
    refined_root = PRIVATE / 'etf-refined-target-diagnostic-cloud-stage-v1/payload'
    if not refined_root.is_dir():
        pytest.skip("Actual refined diagnostic source package is unavailable")
    refined_path = refined_root / 'etf-refined-target-diagnostic-plan.json'
    refined_plan = json.loads(refined_path.read_bytes())
    refined_sources = {
        item["path"]: item
        for item in refined_plan["package_files"].values()
        if item["path"].startswith("src/")
    }
    for flat, item in refined_plan["package_files"].items():
        assert hashlib.sha256((refined_root / flat).read_bytes()).hexdigest() == item["sha256"]
    assert set(historical_sources) == set(refined_sources)
    changed = {
        name
        for name in historical_sources
        if historical_sources[name]["sha256"] != refined_sources[name]["sha256"]
    }
    assert changed == {"src/quantlab/allocation/weighting.py"}
    assert refined_sources["src/quantlab/allocation/weighting.py"]["sha256"] == (
        "6407da185bd1c89798e75ad438bff78ef4fd95a16a9ca3e07ec2c5328e0b2971"
    )
    current = {
        name: item["sha256"]
        for name, item in stage.production_inventory(ROOT).items()
        if name.startswith("src/")
    }
    assert set(current) == set(refined_sources)
    observed = {
        name: {"historical": item["sha256"], "current": current[name]}
        for name, item in refined_sources.items()
        if item["sha256"] != current[name]
    }
    expected_changes = {
        "src/quantlab/backtest/engine.py": {
            "current": "b10eb5e4a8e66472a3d2376510b280623fb6f3daa6a723cb3f5be27d6c69409e",
            "historical": "86caf2fe87d2a570a72f6d14af1e029f118e200955ac01ba30385b57da3ccbfd"
        }
    }
    assert observed == expected_changes



def test_original_report_authenticated_and_old_action_rejected():
    root = PRIVATE / "etf-target-diagnostic-publication-v1/scientific-small-output-v1"
    report = root / "etf-target-diagnostic/target-diagnostic.json"
    if not report.is_file():
        pytest.skip("Actual authenticated diagnostic report is unavailable")
    assert hashlib.sha256(report.read_bytes()).hexdigest() == (
        "4b04679506bdaa8b27f02531f11f67a526cc061806aaa197975634094eb7d14f"
    )
    plan = PRIVATE / "etf-target-diagnostic-cloud-stage-v1/payload/etf-target-diagnostic-plan.json"
    if not plan.is_file():
        pytest.skip("Actual original diagnostic plan is unavailable")
    with pytest.raises(ValueError, match="EXACT_TARGET_DIAGNOSTIC_ACTION_REQUIRED"):
        runner.validate_plan(json.loads(plan.read_bytes()))
