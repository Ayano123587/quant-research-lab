"""Actual immutable inputs and source guards only; no fits or cloud mutations."""

import ast
import json
import os
import shutil
from pathlib import Path

import pytest

from cloud import stage_stock_ablation as stage
from cloud import stock_ablation_runner as runner

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


@pytest.fixture(scope="module")
def actual():
    folder = Path(
        os.environ.get(
            "QUANTLAB_PREDICTION_TRANSPORT", PRIVATE / "stock-economic-prediction-transport-v1"
        )
    )
    reference = Path(
        os.environ.get(
            "QUANTLAB_STOCK_REFERENCE",
            PRIVATE
            / "stock-model-publication-v1/scientific-small-output-v1/stock-development/study",
        )
    )
    freeze = PRIVATE / "stock-economic-prediction-publication-v1/root-data-freeze-v1.json"
    if not folder.is_dir() or not reference.is_dir() or not freeze.is_file():
        pytest.skip("Actual99-member input, original reference and root freeze required")
    assert (
        runner.file_sha(freeze)
        == "cee55829a9df7aea3966feaf40fe62ed845931c8efc379567562779630ac813b"
    )
    return folder, reference, json.loads(freeze.read_bytes())


def test_actual_99_inputs_then_explicit_original_scalar_metrics(actual, tmp_path):
    folder, reference, freeze = actual
    target = tmp_path / "reference"
    runner.extract_predictions(
        folder / "stock-economic-predictions-v1.bin",
        (folder / "stock-prediction-transport.json").read_bytes(),
        target,
        freeze,
        lambda: None,
    )
    assert len([p for p in target.rglob("*") if p.is_file()]) == 99
    assert not (target / "metrics.json").exists()
    raw, pin = stage.reference_metrics(reference)
    runner.restore_reference_metrics(target, raw, pin)
    assert (target / "metrics.json").read_bytes() == (reference / "metrics.json").read_bytes()
    assert len([p for p in target.rglob("*") if p.is_file()]) == 100


def test_actual_nested_container_sha_size_and_uniqueness(actual, tmp_path):
    folder, _, _ = actual
    source = folder / "stock-economic-predictions-v1.bin"
    parent = tmp_path / "input/datasets/actual-owner/actual-release/1"
    parent.mkdir(parents=True)
    target = parent / source.name
    shutil.copyfile(source, target)
    assert (
        runner.discover_bound(
            tmp_path / "input", source.name, runner.PREDICTION_BINARY, source.stat().st_size
        )
        == target
    )
    with pytest.raises(ValueError, match="ONE_EXACT"):
        runner.discover_bound(
            tmp_path / "input", source.name, runner.PREDICTION_BINARY, source.stat().st_size + 1
        )
    shutil.copyfile(source, parent.parent / source.name)
    with pytest.raises(ValueError, match="ONE_EXACT"):
        runner.discover_bound(tmp_path / "input", source.name, runner.PREDICTION_BINARY)


def test_exact_reviewed_recursive_helper_and_dynamic_production_closure():
    def helper(path):
        tree = ast.parse(path.read_bytes())
        return ast.dump(
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "discover_bound"
            )
        )

    assert helper(ROOT / "cloud/stock_ablation_runner.py") == helper(
        ROOT / "cloud/risk_economic_runner.py"
    )
    closure = stage.import_closure(ROOT)
    assert "src/quantlab/stocks/training_targets.py" in closure
    assert "src/quantlab/stocks/development.py" in closure
    assert "src/quantlab/stocks/ablation.py" in closure
    inventory = stage.production_inventory(ROOT)
    assert set(closure) <= set(inventory)
    assert stage.runtime_from_lock((ROOT / "uv.lock").read_bytes())["lightgbm"] == "4.7.0"
    assert b"\r" not in (ROOT / "cloud/stock_ablation_runner.py").read_bytes()
    compile((ROOT / "cloud/stock_ablation_runner.py").read_bytes(), "actual_runner", "exec")


def test_actual_freezes_missing_source_approval_refused_before_mkdir(actual, tmp_path):
    _, reference, _ = actual
    output = tmp_path / "not-staged"
    with pytest.raises(ValueError, match="REGULAR_BOUND_FILE_REQUIRED"):
        stage.build_stage(
            ROOT,
            output,
            prepared_data_freeze_path=PRIVATE
            / "stock-panel-publication-v1/root-data-freeze-v1.json",
            expected_prepared_data_freeze_sha256="b24b7e20a90c5e74d7921aa643a8caeb1d9ca38d237c3fe14732407d50063d04",
            prediction_data_freeze_path=PRIVATE
            / "stock-economic-prediction-publication-v1/root-data-freeze-v1.json",
            expected_prediction_data_freeze_sha256="cee55829a9df7aea3966feaf40fe62ed845931c8efc379567562779630ac813b",
            source_freeze_path=tmp_path / "no-approval.json",
            expected_source_freeze_sha256=runner.PREDICTION_MANIFEST,
            reference_root=reference,
            source_dataset_id="aikyatansinha/portfolio-quant-stock-ablation-source-v1",
            kernel_id="aikyatansinha/portfolio-quant-stock-feature-ablation-v1",
        )
    assert not output.exists()
