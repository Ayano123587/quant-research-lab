"""Actual saved-prediction bytes and source-only transport checks; no economics run."""

import json
import os
from pathlib import Path

import pytest

from cloud.stage_stock_economic import (
    import_closure,
    production_inventory,
    runtime_from_lock,
)
from cloud.stock_economic_runner import (
    canonical_runner,
    discover_bound,
    extract_predictions,
    file_sha,
    validate_prediction_freeze,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


@pytest.fixture(scope="module")
def actual():
    folder = Path(
        os.environ.get(
            "QUANTLAB_PREDICTION_TRANSPORT",
            PRIVATE / "stock-economic-prediction-transport-v1",
        )
    )
    freeze_path = (
        PRIVATE / "stock-economic-prediction-publication-v1/root-data-freeze-v1.json"
    )
    if not folder.is_dir() or not freeze_path.is_file():
        pytest.skip(
            "Actual complete99-member prediction transport and root freeze required"
        )
    assert (
        file_sha(freeze_path)
        == "cee55829a9df7aea3966feaf40fe62ed845931c8efc379567562779630ac813b"
    )
    return folder, json.loads(freeze_path.read_bytes())


def test_actual_99_raw_members_before_native_parsing(actual, tmp_path):
    folder, freeze = actual
    validate_prediction_freeze(freeze)
    reached = []
    target = tmp_path / "actual-predictions"
    extract_predictions(
        folder / "stock-economic-predictions-v1.bin",
        (folder / "stock-prediction-transport.json").read_bytes(),
        target,
        freeze,
        lambda: reached.append(True),
    )
    assert len(list(target.rglob("*.parquet"))) == 96
    assert len(list(target.rglob("*.*"))) == 98  # COMPLETE has no suffix.
    assert reached
    assert file_sha(target / "manifest.json") == freeze["upstream_manifest_sha256"]
    assert not list(target.rglob("*.pkl")) and not list(target.rglob("*.joblib"))


def test_actual_profile_refuses_native_model_flag_before_extraction(actual, tmp_path):
    folder, freeze = actual
    changed = dict(freeze, native_models_packaged=True)
    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="PROFILE"):
        extract_predictions(
            folder / "stock-economic-predictions-v1.bin",
            (folder / "stock-prediction-transport.json").read_bytes(),
            target,
            changed,
            lambda: pytest.fail("Extraction reached before invalidfreeze refusal"),
        )
    assert not target.exists()


def test_actual_source_and_lock_closure_no_core_changes():
    inventory = production_inventory(ROOT)
    assert len(import_closure(ROOT)) == 19
    assert set(import_closure(ROOT)) <= set(inventory)
    assert runtime_from_lock((ROOT / "uv.lock").read_bytes())
    raw = (ROOT / "cloud/stock_economic_runner.py").read_bytes()
    assert canonical_runner(raw) == raw.replace(b"\r\n", b"\n")
    assert (
        inventory["src/quantlab/stocks/economic_study.py"]["sha256"]
        == "1cc718af86e5ee10d97f97e618b64b36decf606a42d29ccf541ff4bc2dc884e7"
    )


def test_actual_nested_plan_and_whole_prediction_blob_identity(actual, tmp_path):
    import shutil

    folder, freeze = actual
    nested = tmp_path / "input/datasets/aikyatansinha/approved/versions/1"
    nested.mkdir(parents=True)
    plan = PRIVATE / "stock-economic-cloud-stage-v1/payload/stock-economic-plan.json"
    target = nested / plan.name
    shutil.copyfile(plan, target)
    assert (
        discover_bound(
            tmp_path / "input", plan.name, file_sha(plan), plan.stat().st_size
        )
        == target
    )
    binary = folder / freeze["container"]["name"]
    copied = nested / binary.name
    shutil.copyfile(binary, copied)
    assert (
        discover_bound(
            tmp_path / "input",
            binary.name,
            freeze["container"]["sha256"],
            freeze["container"]["bytes"],
        )
        == copied
    )
    with pytest.raises(ValueError, match="ONE_EXACT"):
        discover_bound(
            tmp_path / "input",
            binary.name,
            freeze["container"]["sha256"],
            freeze["container"]["bytes"] + 1,
        )
    duplicate = tmp_path / "input/another"
    duplicate.mkdir()
    shutil.copyfile(plan, duplicate / plan.name)
    with pytest.raises(ValueError, match="ONE_EXACT"):
        discover_bound(tmp_path / "input", plan.name, file_sha(plan))


def test_startup_failure_receipt_before_any_runtime_bootstrap(tmp_path, monkeypatch):
    from cloud import stock_economic_runner as runner

    original = Path.is_dir
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        Path, "is_dir", lambda p: p == Path("/kaggle/working") or original(p)
    )
    monkeypatch.setattr(
        runner, "load_module", lambda *_: pytest.fail("Bootstrap reached")
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="ROOT_FREEZE_REQUIRED"):
        runner.run(tmp_path, output)
    assert (
        json.loads((output / "status.json").read_bytes())["phase"]
        == "startup_plan_discovery"
    )
    assert (
        json.loads((output / "FAILURE.json").read_bytes())["phase"]
        == "startup_plan_discovery"
    )
