"""No native fits or market-record fabrication: source/actual frozen-input guards."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def runner():
    path = REPO / "cloud/stock_wider_a_diagnostic_runner.py"
    spec = importlib.util.spec_from_file_location("a_diagnostic_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unbound_refuses_before_output(tmp_path):
    module = runner()
    with pytest.raises(ValueError, match="Unbound"):
        module.run(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_actual_parent_plan_and_frozen_bootstrap_discovery():
    module = runner()
    stage = REPO.parent / "quant-research-lab-private/stock-wider-cloud-stage-v2/payload"
    if not stage.is_dir():
        pytest.skip("Actual authenticated stage absent; never invent input fixtures")
    import time

    plan_path = module.discover(
        stage, module.PARENT_PLAN_SHA256, time.monotonic() + 60, name="*.json"
    )
    plan = json.loads(plan_path.read_bytes())
    item = next(
        i for i in plan["source_files"] if i["path"] == "cloud/stock_wider_notebook_bootstrap.py"
    )
    actual = module.discover(
        stage, item["sha256"], time.monotonic() + 60, name=item["package_name"], size=item["bytes"]
    )
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == item["sha256"]
    assert set(plan["data_bindings"]) == {
        "original_market",
        "original_prepared_panel",
        "original_raw_reference",
    }


def test_both_actual_sources_compile_and_child_has_no_fit_calls():
    for name in ("runner", "child"):
        path = REPO / f"cloud/stock_wider_a_diagnostic_{name}.py"
        source = path.read_text()
        tree = ast.parse(source)
        compile(tree, str(path), "exec")
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"fit", "predict", "fit_predict"}
            for n in ast.walk(tree)
        )
    child = (REPO / "cloud/stock_wider_a_diagnostic_child.py").read_text()
    assert "verify_prepared_panel" in child
    assert "len(selected_rows) == 150" in child
    assert "DIAGNOSTIC_COMPLETE" in child
