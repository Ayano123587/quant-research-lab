"""Verify complete actual cloud ledger export and executed portable notebook."""

import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.reporting import etf

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"


@pytest.fixture(scope="module")
def source():
    study = Path(
        os.environ.get(
            "QUANTLAB_ETF_STUDY",
            PRIVATE / "etf-yahoo-publication-v1/scientific-output-v3/etf-development/study",
        )
    )
    if not (study / "COMPLETE").is_file():
        pytest.skip("Actual complete cloud ETF study required")
    manifest = etf.authenticate_study(study, etf.STUDY_SHA)
    return study, manifest


@pytest.mark.real_data
def test_complete_public_csv_matches_all_actual_cloud_ledger_values(source):
    study, manifest = source
    report = ROOT / "reports/etf_controls_v3"
    exported = json.loads((report / "export_manifest.json").read_bytes())
    assert (report / "COMPLETE").read_text().strip() == etf.digest(report / "export_manifest.json")
    assert exported["study_manifest_sha256"] == etf.STUDY_SHA
    for name, digest in exported["outputs_sha256"].items():
        assert etf.digest(report / name) == digest
    daily = pd.read_csv(report / "portfolio_daily.csv", float_precision="round_trip")
    metrics = json.loads((report / "metrics.json").read_bytes())
    assert len(daily) == 7900 and set(daily.policy) == set(etf.POLICIES)
    assert not any(name in daily for name in ("open", "high", "low", "close", "volume"))
    for policy in etf.POLICIES:
        original = pd.read_parquet(study / (policy + ".parquet"))
        public = daily.loc[daily.policy == policy]
        assert len(public) == 1975
        assert public.session.tolist() == original.session.astype(str).tolist()
        for name in original:
            if name == "session":
                continue
            target = (
                "value_" + name.split(":")[1].removesuffix(".us").upper() if ":" in name else name
            )
            np.testing.assert_array_equal(original[name].to_numpy(), public[target].to_numpy())
        etf.verify_metrics(etf.metrics_from_ledger(public), metrics[policy])
    assert len(manifest["outputs_sha256"]) == 14


def test_actual_wrong_artifact_pin_refuses_before_parquet_or_output(source, tmp_path, monkeypatch):
    study, _ = source
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Refused source must never reach Parquet parsing")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    output = tmp_path / "refused-report"
    with pytest.raises(ValueError, match="reviewed v3 study pin"):
        etf.export_report(study, etf.digest(study / "metrics.json"), output)
    assert not calls and not output.exists()


def test_notebook_was_executed_and_contains_no_error_or_private_input_path():
    notebook = json.loads((ROOT / "notebooks/04_etf_controls.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) >= 5
    assert all(isinstance(cell["execution_count"], int) for cell in cells)
    assert not any(output["output_type"] == "error" for cell in cells for output in cell["outputs"])
    indexed = json.loads((ROOT / "reports/notebook-execution.json").read_bytes())["notebooks"]
    declared = [row for row in indexed if row["notebook"] == "notebooks/04_etf_controls.ipynb"]
    assert len(declared) == 1 and declared[0]["saved_outputs_valid"] is True
    raw = (ROOT / declared[0]["notebook"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == declared[0]["sha256"]
    images = [
        base64.b64decode("".join(output["data"]["image/png"]), validate=True)
        for cell in cells for output in cell["outputs"]
        if "image/png" in output.get("data", {})
    ]
    assert len(images) == declared[0]["inline_plots"] > 0
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in images)
    assert len({hashlib.sha256(image).hexdigest() for image in images}) == len(images)
    assert "quant-research-lab-private" not in json.dumps(notebook)
    assert "Verified portable report artifact hashes" in json.dumps(notebook)
