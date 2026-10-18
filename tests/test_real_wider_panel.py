"""Scoped real-row ranking parity; never construct a released subset panel."""

from datetime import date
from pathlib import Path

import pytest
import test_real_stock_features as real_six

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, file_sha256
from quantlab.stocks import wider_panel
from quantlab.stocks.features import build_candidates, select_cross_section
from quantlab.stocks.prepare_wider import extend_cross_section
from quantlab.stocks.wider_panel import code_pins, prepare_wider_panel, rank_added
from quantlab.stocks.wider_study import (
    fit_predict_visible,
    load_spec,
    project_rows,
    validate_visible_fit,
)

inputs = real_six.inputs
pytestmark = pytest.mark.real_data


def test_actual_three_stock_ranker_matches_independent_prototype(inputs):
    _, histories, schedule = inputs
    original = select_cross_section(
        build_candidates(
            {a: histories[a] for a in ("stock:ibm.us", "stock:ge.us", "stock:aapl.us")},
            schedule,
            as_of=date(2010, 1, 29),
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        ),
        top_n=500,
    )
    enriched = extend_cross_section(original, software_test_scope=True)
    rows = []
    for row in enriched.rows:
        values = {v.feature_id: v.value for v in row.candidate.values}
        rows.append(
            {
                "asset_id": row.asset_id,
                **{"raw_" + f: v for f, v in values.items()},
                **{"rank_" + f: v for f, v in original.rows[0].normalized},
            }
        )
        rows[-1].update(
            {
                "rank_" + f: v
                for f, v in dict((r.asset_id, r) for r in original.rows)[row.asset_id].normalized
            }
        )
    rank_added(rows)
    expected = {r.asset_id: dict(r.normalized) for r in enriched.rows}
    for row in rows:
        for feature, value in expected[row["asset_id"]].items():
            assert row["rank_" + feature] == value
    assert len(rows) == 3
    for row, enriched_row in zip(rows, enriched.rows, strict=True):
        row.update({"missing_" + f: v for f, v in enriched_row.missing_indicators})
    for arm in (
        "fresh_six_reference",
        "add_five_union_eleven",
        "documented_compact_ten",
        "all_supported_twenty_one",
    ):
        columns, matrix = project_rows(rows, arm)
        assert matrix == enriched.project(arm).rows
        assert columns == enriched.project(arm).column_ids


def test_duplicate_actual_cohort_keys_refused(inputs):
    _, histories, _ = inputs
    asset = next(iter(histories))
    with pytest.raises(ValueError, match="Duplicate"):
        rank_added([{"asset_id": asset}, {"asset_id": asset}])


def test_actual_preparation_source_boundary_hashes_and_refuses_before_output(inputs, tmp_path):
    root = Path(wider_panel.__file__).resolve().parents[2]
    pins = code_pins()
    assert len(pins) == 5
    assert all(
        file_sha256((root.parent if name.startswith("configs/") else root) / name) == digest
        for name, digest in pins.items()
    )
    _, _, schedule = inputs
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="Authenticated original"):
        prepare_wider_panel(None, tmp_path, schedule, output)
    assert not output.exists()


def test_actual_protocol_and_visible_estimator_source_without_native_fit():
    project = Path(__file__).resolve().parents[1]
    spec = load_spec(project / "configs/stock-wider-development.toml")
    assert spec["model_columns"] == [6, 16, 15, 36]
    assert len(validate_visible_fit(fit_predict_visible)) == 64
    with pytest.raises(ValueError, match="differs"):
        validate_visible_fit(load_spec)


def test_private_main_notebook_contains_real_unexecuted_workflow(tmp_path):
    import ast
    import importlib.util
    import json

    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "wider_notebook_builder", project / "cloud/build_stock_wider_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "private-draft.ipynb"
    proof = module.build_notebook(output, project_root=project)
    saved = json.loads(output.read_bytes())
    assert proof["source_only"] and saved["metadata"]["quantlab"]["executed"] is False
    assert saved["metadata"]["quantlab"]["bootstrap_bound"] is False
    code = [c for c in saved["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is None and c["outputs"] == [] for c in code)
    child_blocks = []
    for cell in code:
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_locked"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                child_blocks.append(node.args[0].value)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "training_source"
                for target in node.targets
            ):
                child_blocks.append(node.value.value)
    assert len(child_blocks) == 6
    for child in child_blocks:
        ast.parse(child)
        compile(child, "actual-emitted-wider-child", "exec")
    source = "\n".join("".join(c["source"]) for c in code)
    assert "model.fit(" in source and "model.predict(" in source
    assert "prepare_wider_panel(" in source and "training_batch(" in source
    assert "run_wider_study(" in source and "SOURCE ONLY" in source


def test_notebook_transport_unbound_guard_and_actual_nested_source_bytes(tmp_path):
    import importlib.util
    import time

    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "wider_bootstrap", project / "cloud/stock_wider_notebook_bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = project / "src/quantlab/stocks/wider_features.py"
    nested = tmp_path / "actual-source-mount" / "owner" / "dataset" / source.name
    nested.parent.mkdir(parents=True)
    nested.write_bytes(source.read_bytes())
    kwargs = {
        "sha256": file_sha256(source),
        "size": source.stat().st_size,
        "filename": source.name,
        "deadline": time.monotonic() + 60,
    }
    assert module.discover_bound(tmp_path / "actual-source-mount", **kwargs) == nested
    duplicate = nested.parent / "duplicate" / source.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Exactly one"):
        module.discover_bound(tmp_path / "actual-source-mount", **kwargs)
    output = tmp_path / "unbound-worker"
    with pytest.raises(ValueError, match="Externally bound"):
        module.run(tmp_path / "actual-source-mount", output)
    assert (output / "STARTED.json").is_file() and (output / "FAILURE.json").is_file()
    assert not (output / "COMPLETE").exists()
