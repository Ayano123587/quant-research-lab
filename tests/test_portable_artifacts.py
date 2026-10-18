"""Validate checked-in real-study artifacts without raw data or cell execution."""

import ast
import hashlib
import json
from pathlib import Path

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_schema_python_syntax_and_saved_outputs(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    transformer = TransformerManager()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.source.strip():
            assert isinstance(cell.execution_count, int) and cell.execution_count > 0
        # Parse notebook magics as IPython would; never execute cells, download
        # datasets, import notebook models or reconstruct research outcomes.
        python = transformer.transform_cell(cell.source)
        compile(python, f"{path.name}:cell-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        assert not any(output.output_type == "error" for output in cell.get("outputs", []))
    assert any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")


@pytest.mark.parametrize(
    ("directory", "manifest_name"),
    [
        ("etf_controls_v3", "export_manifest.json"),
        ("stock_development_v1", "export-manifest.json"),
        ("stock_economic_support_v1", "export-manifest.json"),
        ("etf_cost_sensitivity_v2", "export_manifest.json"),
        ("risk_forecast_v1", "export-manifest.json"),
        ("etf_target_calibration_v1", "export-manifest.json"),
        ("risk_forecast_calibration_v1", "export-manifest.json"),
        ("stock_feature_ablation_v1", "export-manifest.json"),
        ("stock_target_comparison_v1", "export-manifest.json"),
        ("risk_economic_development_v1", "export-manifest.json"),
        ("risk_economic_uncertainty_v1", "export-manifest.json"),
        ("etf_sensitivity_v3", "export-manifest.json"),
        ("etf_covariance_integration_v1", "export-manifest.json"),
    ],
)
def test_public_real_report_files_match_completed_export(directory, manifest_name):
    report = ROOT / "reports" / directory
    raw = (report / manifest_name).read_bytes()
    manifest = json.loads(raw)
    assert manifest["complete"] is True
    assert (report / "COMPLETE").read_text().strip() == hashlib.sha256(raw).hexdigest()
    assert manifest["outputs_sha256"]
    for name, expected in manifest["outputs_sha256"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert "\\" not in name and ":" not in name
        target = report / relative
        assert target.is_file() and not target.is_symlink()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected




