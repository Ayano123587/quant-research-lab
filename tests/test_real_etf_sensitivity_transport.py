"""Actual Yahoo bytes and prospective guard mechanics; no full study or invented prices."""

import ast
import json
import shutil
from pathlib import Path

import pytest

from cloud import etf_development_runner as legacy
from cloud import etf_sensitivity_runner as runner
from cloud.stage_etf_sensitivity import (
    build_stage,
    import_closure,
    production_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


@pytest.fixture
def actual():
    folder = PRIVATE / "yahoo-etf-transport-v1"
    freeze = folder / "root-data-freeze-v1.json"
    if not freeze.is_file() or not (folder / "payload").is_dir():
        pytest.skip("Reviewed actual whole Yahoo transport absent")
    assert (
        runner.file_sha(freeze)
        == "6dbb5162e047077b962dbe60f815f8a623d21f70043a61e9f2e5da210449fb0f"
    )
    return folder, freeze


def test_actual_nested_whole_yahoo_identity_and_duplicate_refusal(actual, tmp_path):
    folder, freeze = actual
    nested = tmp_path / "input/datasets/aikyatansinha/whole-etf/versions/1"
    nested.mkdir(parents=True)
    copied = nested / freeze.name
    shutil.copyfile(freeze, copied)
    assert (
        runner.discover_bound(tmp_path / "input", freeze.name, runner.file_sha(freeze))
        == copied
    )
    binary = folder / "payload" / legacy.CONTAINER_NAME
    target = nested / binary.name
    shutil.copyfile(binary, target)
    frozen = json.loads(freeze.read_bytes())
    assert (
        runner.discover_bound(
            tmp_path / "input",
            binary.name,
            frozen["container"]["sha256"],
            frozen["container"]["bytes"],
        )
        == target
    )
    duplicate = tmp_path / "input/duplicate"
    duplicate.mkdir()
    shutil.copyfile(freeze, duplicate / freeze.name)
    with pytest.raises(ValueError, match="ONE_EXACT"):
        runner.discover_bound(tmp_path / "input", freeze.name, runner.file_sha(freeze))


def test_missing_actual_source_approval_cannot_stage(actual, tmp_path):
    _, freeze = actual
    output = tmp_path / "never-staged"
    with pytest.raises(ValueError, match="REGULAR_BOUND_FILE"):
        build_stage(
            ROOT,
            output,
            data_freeze_path=freeze,
            expected_data_freeze_sha256=runner.file_sha(freeze),
            source_freeze_path=tmp_path / "absent-root-approval.json",
            expected_source_freeze_sha256=None,
            source_dataset_id="aikyatansinha/portfolio-quant-etf-sensitivity-source-v1",
            kernel_id="aikyatansinha/portfolio-quant-etf-sensitivity-development-v1",
        )
    assert not output.exists()


def test_startup_receipt_before_native_bootstrap(tmp_path, monkeypatch):
    original = Path.is_dir
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        Path, "is_dir", lambda p: p == Path("/kaggle/working") or original(p)
    )
    monkeypatch.setattr(
        runner, "load_module", lambda *_: pytest.fail("Native bootstrap reached")
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="ROOT_FREEZE_REQUIRED"):
        runner.run(tmp_path, output)
    assert (
        json.loads((output / "FAILURE.json").read_bytes())["phase"]
        == "startup_plan_discovery"
    )


def test_actual_complete_weighting_closure_and_reviewed_discovery_ast():
    closure = import_closure(ROOT)
    assert "src/quantlab/allocation/weighting.py" in closure
    assert len(closure) == 19 and len(production_inventory(ROOT)) == 28

    def function(path):
        tree = ast.parse(path.read_bytes())
        return ast.dump(
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "discover_bound"
            ),
            include_attributes=False,
        )

    assert function(ROOT / "cloud/risk_economic_runner.py") == function(
        ROOT / "cloud/etf_sensitivity_runner.py"
    )
