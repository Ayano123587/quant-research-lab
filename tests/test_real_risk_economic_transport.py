"""Actual forecast/container byte guards only: no fits, backtests or invented READY records."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from cloud import risk_economic_runner as runner
from cloud.package_risk_economic_inputs import UPSTREAM_SHA, profile_inventory
from cloud.stage_risk_economic import build_stage, import_closure, runtime_from_lock

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


def test_actual_full_fixed_forecast_profile_hashes():
    root = Path(
        os.environ.get("QUANTLAB_RISK_FORECASTS", ROOT / "data/processed/risk-integration-v1")
    )
    if not (root / "COMPLETE").exists():
        pytest.skip("Actual completed fixed forecast profile required")
    inventory = profile_inventory(root, UPSTREAM_SHA)
    assert len(inventory) == 1526
    assert sum(name.startswith("cells/") for name in inventory) == 1520
    assert sum(item["bytes"] for item in inventory.values()) == 655768487


def test_actual_binary_container_matches_root_roundtrip_pins():
    root = Path(
        os.environ.get("QUANTLAB_RISK_CONTAINER", ROOT / "data/raw/risk-economic-forecasts")
    )
    binary = root / "risk-integration-forecast-v1.bin"
    if not binary.exists():
        pytest.skip("Actual root-created full 1526-member container required")
    inventory = runner.inspect_forecast_container(
        binary, (root / "risk-integration-transport.json").read_bytes()
    )
    assert len(inventory) == 1526
    assert inventory["manifest.json"]["sha256"] == UPSTREAM_SHA


def test_actual_static_closure_locked_optimization_and_canonical_runner():
    closure = import_closure(ROOT)
    assert len(closure) == len(set(closure)) == 25
    assert all(
        "src/quantlab/" + name in closure
        for name in (
            "risk/economic_study.py",
            "risk/baseline_execution.py",
            "risk/signal_weights.py",
            "evaluation/attribution.py",
            "backtest/engine.py",
            "backtest/accounting.py",
            "data/provenance.py",
            "data/yahoo.py",
        )
    )
    runtime = runtime_from_lock((ROOT / "uv.lock").read_bytes())
    assert runtime["cvxpy"] == "1.9.3" and runtime["clarabel"] == "0.11.1"
    assert runtime["osqp"] == "1.1.3"
    raw = (ROOT / "cloud/risk_economic_runner.py").read_bytes()
    assert runner.canonical_runner(raw) == runner.canonical_runner(raw.replace(b"\n", b"\r\n"))


def test_actual_interpreter_child_diagnostic_and_host_refusal(tmp_path, monkeypatch):
    private, output = tmp_path / "private", tmp_path / "output"
    private.mkdir()
    output.mkdir()
    runner.run_child_with_diagnostics(
        [sys.executable, "-c", "import sys; print(sys.version)"],
        os.environ.copy(),
        private,
        output,
        timeout=20,
    )
    receipt = json.loads((output / "child-diagnostics.json").read_bytes())
    assert "3.12.15" in receipt["tail"] and not receipt["truncated"]
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
            runner.run(tmp_path / "inputs", tmp_path / "forbidden_full_study")
    assert not (tmp_path / "forbidden_full_study").exists()


def test_absent_actual_forecast_publication_refuses_before_mkdir(tmp_path):
    data = PRIVATE / "kaggle-market-data-freeze-v1.json"
    etf = PRIVATE / "yahoo-etf-transport-v1/root-data-freeze-v1.json"
    if not data.exists() or not etf.exists():
        pytest.skip("Actual original/Yahoo root publication receipts required")
    output = tmp_path / "no_stage"
    with pytest.raises(ValueError, match="REGULAR_BOUND_FILE_REQUIRED"):
        build_stage(
            ROOT,
            output,
            data_freeze_path=data,
            expected_data_freeze_sha256=runner.file_sha(data),
            etf_data_freeze_path=etf,
            expected_etf_data_freeze_sha256=runner.file_sha(etf),
            forecast_data_freeze_path=tmp_path / "not_published.json",
            expected_forecast_data_freeze_sha256="0" * 64,
            source_freeze_path=tmp_path / "not_approved.json",
            expected_source_freeze_sha256="0" * 64,
            source_dataset_id="aikyatansinha/quant-risk-economic-source",
            kernel_id="aikyatansinha/quant-risk-economic-development",
        )
    assert not output.exists()


def test_actual_nested_plan_and_real_binary_discovery_refuse_ambiguity(tmp_path):
    stage = PRIVATE / "risk-economic-cloud-stage-v1/payload/risk-economic-plan.json"
    container_root = Path(
        os.environ.get("QUANTLAB_RISK_CONTAINER", ROOT / "data/raw/risk-economic-forecasts")
    )
    binary = container_root / "risk-integration-forecast-v1.bin"
    if not stage.exists() or not binary.exists():
        pytest.skip("Actual approved plan and complete real forecast container required")
    nested = tmp_path / "input/datasets/aikyatansinha/actual-source/version1"
    nested.mkdir(parents=True)
    copied = nested / stage.name
    shutil.copyfile(stage, copied)
    plan_sha = runner.file_sha(stage)
    assert runner.discover_bound(tmp_path / "input", stage.name, plan_sha) == copied
    with pytest.raises(ValueError, match="ONE_EXACT_RECURSIVE_BOUND_INPUT_REQUIRED"):
        runner.discover_bound(tmp_path / "input", stage.name, "0" * 64)
    duplicate = nested.parent / stage.name
    shutil.copyfile(stage, duplicate)
    with pytest.raises(ValueError, match="ONE_EXACT_RECURSIVE_BOUND_INPUT_REQUIRED"):
        runner.discover_bound(tmp_path / "input", stage.name, plan_sha)
    duplicate.unlink()
    # Copy the actual immutable bytes; source and pytest temp can be different drives.
    linked = nested / binary.name
    shutil.copyfile(binary, linked)
    assert (
        runner.discover_bound(
            tmp_path / "input",
            binary.name,
            runner.FORECAST_BINARY_SHA,
            runner.FORECAST_BINARY_BYTES,
        )
        == linked
    )
    with pytest.raises(ValueError, match="ONE_EXACT_RECURSIVE_BOUND_INPUT_REQUIRED"):
        runner.discover_bound(
            tmp_path / "input",
            binary.name,
            runner.FORECAST_BINARY_SHA,
            runner.FORECAST_BINARY_BYTES - 1,
        )
