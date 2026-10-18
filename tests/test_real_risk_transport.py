"""Actual source/config/approval boundary checks; never stage or launch a job."""

import json
import sys
import time
from pathlib import Path

import pytest

from cloud import risk_forecast_runner as runner
from cloud import stage_risk_forecast as stager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


def test_actual_ast_source_closure_and_locked_core_runtime():
    assert set(stager.local_import_closure(ROOT)) == set(runner.SOURCE_FILES)
    inventory = stager.production_inventory(ROOT)
    assert set(inventory) == set(
        runner.SOURCE_FILES + runner.CONFIG_FILES + runner.PROJECT_FILES + runner.TRANSPORT_FILES
    )
    assert len(inventory) == 19
    assert runner.CONFIG_FILES == ("configs/risk-development.toml",)
    runtime = stager.runtime_from_lock((ROOT / "uv.lock").read_bytes())
    assert runtime["threadpoolctl"] == "3.7.0"
    assert runtime["exchange-calendars"] == "4.13.2"
    assert "lightgbm" not in runtime
    assert len(set(stager.flat_name(name) for name in inventory)) == len(inventory)


def test_actual_runner_template_binding_only_no_stage():
    template = (ROOT / "cloud/risk_forecast_runner.py").read_bytes()
    # Actual existing config hash tests canonical rendering, not a plan approval.
    digest = runner.sha((ROOT / "configs/risk-development.toml").read_bytes())
    rendered = stager.render_runner(template, digest)
    assert runner.canonical_runner(template) == runner.canonical_runner(rendered)
    assert runner.canonical_runner(rendered.replace(b"\n", b"\r\n")) == runner.canonical_runner(
        template
    )
    compile(rendered, "actual-risk-template-render-check.py", "exec")


def test_actual_ready_data_cannot_replace_missing_root_risk_source_freeze(tmp_path):
    freeze = PRIVATE / "kaggle-market-data-freeze-v1.json"
    if not freeze.is_file():
        pytest.skip("Actual root remote-data approval absent")
    raw = freeze.read_bytes()
    value = json.loads(raw)
    runner.validate_data_freeze(value)
    assert value["dataset"]["id"] == 12654575 and value["dataset"]["version"] == 1
    output = tmp_path / "must-not-stage"
    with pytest.raises(FileNotFoundError):
        stager.build_stage(
            ROOT,
            output,
            data_freeze_path=freeze,
            expected_data_freeze_sha256=runner.sha(raw),
            source_freeze_path=tmp_path / "absent-root-risk-freeze.json",
            expected_source_freeze_sha256=runner.sha(
                (ROOT / "configs/risk-development.toml").read_bytes()
            ),
            source_dataset_id=None,
            kernel_id=None,
            service_budget_seconds=600,
        )
    assert not output.exists()
    with pytest.raises(runner.TransportError, match="ROOT_SOURCE_FREEZE_REQUIRED"):
        stager.validate_source_freeze(ROOT, value)


def test_expired_real_clock_budget_retains_failure_without_install(tmp_path, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("No installation allowed by software budget check")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        runner.bootstrap_with_receipt(
            tmp_path / "private", ROOT, tmp_path, deadline=time.monotonic() - 1
        )
    assert not calls
    receipt = json.loads((tmp_path / "bootstrap-status.json").read_bytes())
    assert receipt["phase"] == "bootstrap_failed" and receipt["complete"] is False
    assert receipt["error_type"] == "TimeoutError"
    assert not (tmp_path / "COMPLETE.json").exists()


def test_unbound_runner_refuses_before_project_or_native_imports(tmp_path):
    before = set(sys.modules)
    with pytest.raises(runner.TransportError, match="ROOT_BINDINGS_REQUIRED"):
        runner.run(tmp_path / "no-inputs", tmp_path / "refused-run")
    after = set(sys.modules) - before
    assert not any(name.startswith(("quantlab", "numpy", "pyarrow", "sklearn")) for name in after)
    output = tmp_path / "refused-run"
    assert (output / "FAILURE.json").is_file()
    assert not (output / "COMPLETE.json").exists()


def test_private_environment_keeps_caches_and_fixed_threads_outside_outputs(tmp_path):
    private, project = tmp_path / "private", tmp_path / "private/project"
    environment = runner.private_environment(private, project)
    assert environment["UV_CACHE_DIR"] == str(private / "uv-cache")
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(private / "runtime/.venv")
    assert environment["PYTHONPATH"] == str(project / "src")
    assert environment["OPENBLAS_NUM_THREADS"] == "4"
    assert environment["OMP_NUM_THREADS"] == "4"
