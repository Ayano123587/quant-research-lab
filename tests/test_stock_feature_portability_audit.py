"""Actual observed float comparisons and frozen source metadata, no host audit."""

import ast
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO.parent / "quant-research-lab-private"


def module():
    path = REPO / "cloud/stock_feature_portability_audit.py"
    spec = importlib.util.spec_from_file_location("full_portability_source", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def arithmetic():
    """Load only two stdlib arithmetic helpers, never the cloud child's main."""
    source = module().CHILD
    tree = ast.parse(source)
    compile(tree, "actual-plain-child", "exec")
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"ordered_float_bits", "feature_comparison"}
    ]
    namespace = {"struct": struct, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "arithmetic-only", "exec"), namespace)
    return namespace


def test_unbound_audit_refuses_before_directory(tmp_path):
    with pytest.raises(ValueError, match="Unbound"):
        module().run(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_actual_portable_A_float_bits_and_distance():
    path = REPO / "tests/fixtures/actual_A_feature_reproduction_2010-11-30.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "52f4a90d4a64fc0216776653a191fa96579248860e6d03f17d757abe0cb68e20"
    )
    actual = json.loads(raw)["row"]["checks"]
    compare = arithmetic()["feature_comparison"]
    for feature in ("F01", "F07", "F09", "F13", "F15", "F19"):
        observed = actual[feature]
        result = compare(observed["parent"], observed["actual"])
        assert result["parent_float64_hex"] == observed["parent_float64_hex"]
        assert result["computed_float64_hex"] == observed["actual_float64_hex"]
        assert result["exact"] is observed["equal"]
        assert result["ulp_distance"] == (1 if feature == "F09" else 0)


def test_actual_AA_F19_failed_record_distance():
    path = (
        PRIVATE
        / "stock-wider-publication-v2/terminal-small-output-v2"
        / "wider-run/wider-panel/FAILURE.json"
    )
    if not path.is_file():
        pytest.skip("Actual private version2 failure receipt absent; no fabricated substitute")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "227b070f4803178ae67b9236a07e31d610689460bf6fc97e39c2c1251395b1f1"
    )
    failure = json.loads(raw)["original_gate_diagnostic"]
    assert failure["asset_id"] == "stock:aa.us"
    observed = failure["features"]["F19"]
    result = arithmetic()["feature_comparison"](observed["parent"], observed["computed"])
    assert result["ulp_distance"] == 1 and result["exact"] is False and result["finite"] is True
    assert result["parent_float64_hex"] == observed["parent_float64_hex"]
    assert result["computed_float64_hex"] == observed["computed_float64_hex"]


def test_actual_whole_parent_scope_and_frozen_source_unchanged():
    manifest = REPO / "data/processed/stock-monthly-v2/manifest.json"
    freeze = PRIVATE / "root-stock-wider-source-freeze-v3.json"
    if not manifest.is_file() or not freeze.is_file():
        pytest.skip("Actual frozen parent/source receipts absent; no invented metadata")
    raw = manifest.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    )
    records = [
        r
        for r in json.loads(raw)["outputs"]
        if r.get("kind") == "features" and Path(r["path"]).stem < "2015-01-01"
    ]
    tree = ast.parse(module().CHILD)
    constants = {
        n.targets[0].id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
    }
    assert constants["EXPECTED_ROWS"] == sum(r["rows"] for r in records) == 71309
    assert constants["EXPECTED_DATES"] == len(records) == 180
    inventory = json.loads(freeze.read_bytes())["production_inventory"]
    assert len(inventory) == 34
    frozen_root = PRIVATE / "stock-wider-cloud-stage-v3/payload"
    if not frozen_root.is_dir():
        pytest.skip("Actual frozen wider source package is unavailable")
    plan = json.loads((frozen_root / "stock-wider-plan.json").read_bytes())
    package = {item["path"]: item for item in plan["source_files"]}
    for name, expected in inventory.items():
        assert package[name]["sha256"] == expected["sha256"]
        frozen_source = frozen_root / package[name]["package_name"]
        assert hashlib.sha256(frozen_source.read_bytes()).hexdigest() == expected["sha256"]
    # Authenticate historical documentation above; current documentation may evolve.
    current_inventory = {name: item for name, item in inventory.items() if name != "README.md"}
    observed = {}
    for name, item in current_inventory.items():
        current_sha = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        if item["sha256"] != current_sha:
            observed[name] = {"historical": item["sha256"], "current": current_sha}
    expected_changes = {
        "cloud/build_stock_wider_notebook.py": {
            "current": "b10a1a8d47277ca7e3d014738feeb80ffefec06bdb4300e54eab1253a94e37d2",
            "historical": "bb156f5468f47b5da8d1a4ca6eb16d3581c0864ef3fe92642ec7e276e6f979d2"
        },
        "cloud/stage_stock_wider_notebook.py": {
            "current": "69275de3070845844ed347297741cdb912cf60f353abf06ee8b2e86549082b75",
            "historical": "1a0a7c89dfa0eeb3fb74e7c56ed0c9b503587f3225a4f3e418d62c676c538a32"
        },
        "configs/stock-wider-development.toml": {
            "current": "2bc7f19f9b05fea877db06b8a1ce6ed3e0181f4a776a38091d156aeeb16604bd",
            "historical": "2c2a72bbe001924dcdb8746b82f9b96350811a3765a9620375390be056d9e686"
        },
        "src/quantlab/stocks/feature_reproduction.py": {
            "current": "2d4d2c691f26416c1d726e3f4325fbce214f873403d222d62930bf37bd3b02ec",
            "historical": "ce16a13d5573b39f6b3e61f9295bc42e05cc52078ee333b2479185c4ca62b657"
        },
        "src/quantlab/stocks/wider_panel.py": {
            "current": "c2e38b2f77a8126a2874fc7b0e73be85d3cd78237a974991ec9b1ad727c79519",
            "historical": "2f0dcfa635f5c3215c8fa6bee423a3c9905b4bebc24e40f2d31d6a52c3e76390"
        },
        "src/quantlab/stocks/wider_study.py": {
            "current": "a114802fd83a5224056acf5c688d3e6d1bcb9f68c6c7f32b0a3fb6209e8abd13",
            "historical": "c67c11a4fd05b7a855e12398a630e0e6ae516761559e5fe5c3f0d58dfba56c66"
        }
    }
    assert observed == expected_changes


def test_plain_child_and_outer_compile_without_estimator_calls():
    source = (REPO / "cloud/stock_feature_portability_audit.py").read_text()
    for text in (source, module().CHILD):
        tree = ast.parse(text)
        compile(tree, "reviewed-source", "exec")
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"fit", "predict", "fit_predict"}
            for n in ast.walk(tree)
        )
