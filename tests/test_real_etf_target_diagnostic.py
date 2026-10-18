"""One real target window and validation traceback only; no host 95-date scan."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
import test_real_source_integration as actual_source

from cloud import etf_target_diagnostic_runner as runner
from cloud import stage_etf_target_diagnostic as stage
from quantlab.allocation.policies import allocate
from quantlab.allocation.weighting import strategic_weights
from quantlab.backtest.engine import HistoricalView

source = actual_source.source
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


def test_real_calendar_produces_exact95_date_values(source):
    schedule = source[1]
    days = runner.diagnostic_signal_dates(schedule)
    assert len(days) == len(set(days)) == 95
    assert days[0] == date(2007, 2, 28)
    assert days[-1] == date(2014, 12, 31)
    assert all(type(day) is date for day in days)
    assert days == tuple(schedule.month_end(day.year, day.month).session for day in days)
    assert all(schedule.session(day).session == day for day in days)


def test_allowlisted_traceback_on_actual_covariance_validation_refusal(source):
    _, schedule, identity, histories = source
    day = date(2007, 2, 28)
    view = HistoricalView(
        tuple((a, tuple(b for b in rows if b.session <= day)) for a, rows in histories.items()),
        day,
        schedule.session(day).decision_at,
        schedule.next_session(day).open_at,
    )
    estimate = allocate(
        view, schedule, policy="trend_252_volatility_63", source_identity=identity
    ).covariance
    path = ROOT / "src/quantlab/allocation/weighting.py"
    # Actual API validation refusal, explicitly not the cloud optimality failure.
    try:
        strategic_weights(estimate, "unregistered_diagnostic_validation_request")
    except ValueError as error:
        captured = runner.capture_solver_failure(error, path, runner.file_sha(path))
    else:
        pytest.fail("Unregistered method must refuse")
    assert captured["function"] == "strategic_weights"
    observed = captured["observed_allowlisted_locals"]
    assert observed["estimate"]["window_hash"] == estimate.window_hash
    assert observed["matrix"] == estimate.matrix.tolist()
    assert "grad" in captured["not_present_at_exception"]
    assert not {"view", "histories", "schedule", "source", "error"} & set(observed)
    assert json.loads(runner.encoded(captured)) == captured


def test_actual_parent_production_source_is_unchanged():
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
    diagnostic_root = PRIVATE / 'etf-target-diagnostic-cloud-stage-v1/payload'
    if not diagnostic_root.is_dir():
        pytest.skip("Actual original diagnostic source package is unavailable")
    diagnostic_plan = json.loads((diagnostic_root / 'etf-target-diagnostic-plan.json').read_bytes())
    diagnostic_sources = {
        item["path"]: item
        for item in diagnostic_plan["package_files"].values()
        if item["path"].startswith("src/")
    }
    for flat, item in diagnostic_plan["package_files"].items():
        assert hashlib.sha256((diagnostic_root / flat).read_bytes()).hexdigest() == item["sha256"]
    original = {name: item["sha256"] for name, item in historical_sources.items()}
    diagnostic = {name: item["sha256"] for name, item in diagnostic_sources.items()}
    assert original == diagnostic
    current = {
        name: item["sha256"]
        for name, item in stage.production_inventory(ROOT).items()
        if name.startswith("src/")
    }
    assert set(current) == set(historical_sources)
    observed = {
        name: {"historical": item["sha256"], "current": current[name]}
        for name, item in historical_sources.items()
        if item["sha256"] != current[name]
    }
    expected_changes = {
        "src/quantlab/allocation/weighting.py": {
            "current": "6407da185bd1c89798e75ad438bff78ef4fd95a16a9ca3e07ec2c5328e0b2971",
            "historical": "d090c64bccc1a96b8257f2e5b4573d46314635f19c8a69fef97675da60e8457a"
        },
        "src/quantlab/backtest/engine.py": {
            "current": "b10eb5e4a8e66472a3d2376510b280623fb6f3daa6a723cb3f5be27d6c69409e",
            "historical": "86caf2fe87d2a570a72f6d14af1e029f118e200955ac01ba30385b57da3ccbfd"
        }
    }
    assert observed == expected_changes
    assert "src/quantlab/allocation/weighting.py" in stage.import_closure(ROOT)
    runner_source = (ROOT / "cloud/etf_target_diagnostic_runner.py").read_bytes()
    compile(runner_source, "actual_diagnostic", "exec")



def test_host_scan_and_missing_actual_source_approval_refused(tmp_path, monkeypatch):
    output = tmp_path / "no-host-scan"
    with monkeypatch.context() as host:
        host.chdir(tmp_path)
        actual_is_dir = Path.is_dir
        host.setattr(
            Path,
            "is_dir",
            lambda path: False
            if path in (Path("/kaggle/working"), Path("/kaggle/input"))
            else actual_is_dir(path),
        )
        with pytest.raises(ValueError, match="KAGGLE_ONLY"):
            runner.run(tmp_path, output)
    assert not output.exists()
    freeze = PRIVATE / "yahoo-etf-transport-v1/root-data-freeze-v1.json"
    if not freeze.exists():
        pytest.skip("Actual externally pinned Yahoo release freeze required")
    with pytest.raises(ValueError, match="REGULAR_BOUND_FILE_REQUIRED"):
        stage.build_stage(
            ROOT,
            output,
            data_freeze_path=freeze,
            expected_data_freeze_sha256=(
                "6dbb5162e047077b962dbe60f815f8a623d21f70043a61e9f2e5da210449fb0f"
            ),
            source_freeze_path=tmp_path / "no-root-approval.json",
            expected_source_freeze_sha256=runner.PARENT_RUNNER_SHA,
            parent_plan_path=PRIVATE
            / "etf-sensitivity-cloud-stage-v2/payload/etf-sensitivity-plan.json",
            source_dataset_id="aikyatansinha/portfolio-quant-etf-target-diag-source-v1",
            kernel_id="aikyatansinha/portfolio-quant-etf-target-diag-v1",
        )
    assert not output.exists()
