"""Real source/approval metadata checks only; no invented provider or market fixtures."""

import ast
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest

from cloud.stage_stock_wider_notebook import (
    _identity,
    build_stage,
    production_inventory,
    runtime_from_lock,
)

REPO = Path(__file__).resolve().parents[1]


def test_actual_ast_inventory_and_locked_dependency_versions():
    inventory = production_inventory(REPO)
    for name, item in inventory.items():
        data = (REPO / name).read_bytes()
        assert item == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert "src/quantlab/stocks/wider_features.py" in inventory
    assert "src/quantlab/stocks/prepare_wider.py" in inventory
    assert "src/quantlab/data/provenance.py" in inventory
    assert "src/quantlab/data/yahoo.py" in inventory
    assert "src/quantlab/stocks/training_targets.py" in inventory
    assert runtime_from_lock(REPO / "uv.lock")["lightgbm"] == "4.7.0"
    assert (REPO / ".python-version").read_text().strip() == "3.12.15"


def test_absent_actual_root_approval_refuses_before_stage_directory(tmp_path):
    output = tmp_path / "refused-stage"
    with pytest.raises(FileNotFoundError):
        build_stage(
            REPO,
            output,
            source_freeze_path=tmp_path / "absent.json",
            expected_source_freeze_sha256="UNBOUND",
            data_inputs={},
            reference_root=tmp_path,
        )
    assert not output.exists()


def test_actual_published_freeze_schema_normalization():
    private = REPO.parent / "quant-research-lab-private"
    names = {
        "original_market": private / "risk-economic-cloud-stage-v2/payload/data-freeze.json",
        "original_prepared_panel": private / "stock-panel-publication-v1/root-data-freeze-v1.json",
        "original_raw_reference": private
        / "stock-economic-prediction-publication-v1/root-data-freeze-v1.json",
    }
    if not all(path.is_file() for path in names.values()):
        pytest.skip("Actual private published freeze receipts are not portable public inputs")
    actual = {role: _identity(json.loads(path.read_bytes()), role) for role, path in names.items()}
    assert {role: item["dataset_id"] for role, item in actual.items()} == {
        "original_market": 12654575,
        "original_prepared_panel": 12657513,
        "original_raw_reference": 12660883,
    }
    assert all(item["private"] is True and item["version"] == 1 for item in actual.values())


def test_actual_sdk_reads_all_four_generated_attachments():
    """Evaluate the real metadata literal with accepted actual bindings, no stage/fit."""
    private = REPO.parent / "quant-research-lab-private"
    stage = private / "stock-wider-cloud-stage-v2"
    proof = private / "stock-wider-publication-v1/remote-byte-proof.json"
    sdk_python = os.environ.get("QUANTLAB_KAGGLE_PYTHON")
    if not sdk_python or not proof.is_file() or not (stage / "stage-receipt.json").is_file():
        pytest.skip("Actual accepted publication receipts and Kaggle SDK interpreter required")
    plan_raw = (stage / "payload/stock-wider-plan.json").read_bytes()
    assert hashlib.sha256(plan_raw).hexdigest() == (
        "548f88e827c32933bd9db1cd301babd842056726ca309d58477d36f88fee544d"
    )
    plan = json.loads(plan_raw)
    accepted = json.loads(proof.read_bytes())
    assert accepted["dataset_id"] == 12666737 and accepted["all35_remote_raw_bytes_exact"]
    metadata = json.loads((stage / "kernel/kernel-metadata.json").read_bytes())
    tree = ast.parse(inspect.getsource(build_stage))
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets)
    )
    generated = eval(
        compile(ast.Expression(assignment.value), "actual-metadata-literal", "eval"),
        build_stage.__globals__,
        {
            "kernel_id": metadata["id"],
            "bindings": plan["data_bindings"],
            "source_dataset": accepted["dataset"],
        },
    )
    assert generated == metadata and "data_sources" not in generated
    result = subprocess.run(
        [
            sdk_python,
            "-c",
            "import contextlib,io,json,sys; "
            "sdk_stdout,sdk_stderr=io.StringIO(),io.StringIO(); "
            "\nwith contextlib.redirect_stdout(sdk_stdout), contextlib.redirect_stderr(sdk_stderr):"
            "\n    from kaggle.api.kaggle_api_extended import KaggleApi"
            "\nprint(json.dumps(KaggleApi().get_or_default(json.load(sys.stdin),'dataset_sources',[])))",
        ],
        input=json.dumps(generated),
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    effective = json.loads(result.stdout)
    assert effective == metadata["dataset_sources"] and len(effective) == 4
