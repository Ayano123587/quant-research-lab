"""Prepare a reviewed plain-source cost study; no upload or study execution."""

from __future__ import annotations

import json
from pathlib import Path

from cloud import etf_cost_runner as runner
from cloud import etf_development_runner as helper

FILES = (
    helper.SOURCE_FILES
    + (
        "src/quantlab/allocation/robustness.py",
        "src/quantlab/evaluation/__init__.py",
        "src/quantlab/evaluation/attribution.py",
    )
    + helper.PROJECT_FILES
    + (runner.HELPER, runner.SELF, "cloud/stage_etf_costs.py")
)


def inventory(repo):
    return {
        name: {
            "sha256": runner.sha((Path(repo) / name).read_bytes()),
            "bytes": (Path(repo) / name).stat().st_size,
        }
        for name in FILES
    }


def stage(
    repo,
    output,
    *,
    source_freeze,
    source_freeze_sha256,
    data_freeze,
    data_freeze_sha256,
    dataset_id,
    kernel_id,
    runtime,
):
    repo, output = Path(repo), Path(output)
    if output.exists():
        raise ValueError("Preserve the previous immutable stage")
    source = json.loads(runner.read(source_freeze, source_freeze_sha256))
    data_raw = runner.read(data_freeze, data_freeze_sha256)
    helper.validate_data_freeze(json.loads(data_raw))
    actual = inventory(repo)
    if source != {
        "schema": "quant_etf_cost_source_freeze_v1",
        "production": actual,
        "data_freeze_sha256": data_freeze_sha256,
        "costs_bps": [0.0, 5.0, 10.0, 25.0],
        "full_measured_sessions": 1975,
        "holdout_evaluated": False,
    }:
        raise ValueError("Actual root cost-source freeze required")
    if dataset_id == kernel_id or dataset_id.split("/")[-1] == kernel_id.split("/")[-1]:
        raise ValueError("Distinct source and kernel identifiers required")
    if set(runtime) != {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "exchange-calendars",
    }:
        raise ValueError("Explicit complete locked numerical runtime required")
    payload = {name: (repo / name).read_bytes() for name in FILES}
    payload.update(
        {
            "provenance/data-freeze.json": data_raw,
            "provenance/source-freeze.json": Path(source_freeze).read_bytes(),
        }
    )
    entries = {
        name: {
            "upload_name": f"source-{index:03d}" + Path(name).suffix,
            "sha256": runner.sha(raw),
            "bytes": len(raw),
        }
        for index, (name, raw) in enumerate(payload.items())
    }
    plan = {
        "schema": "quant_etf_cost_transport_v1",
        "source_dataset": dataset_id,
        "kernel_id": kernel_id,
        "files": entries,
        "runtime": runtime,
        "costs_bps": [0.0, 5.0, 10.0, 25.0],
        "max_seconds": 36000,
        "full_measured_sessions": 1975,
        "holdout_evaluated": False,
        "source_freeze_sha256": source_freeze_sha256,
        "data_freeze_sha256": data_freeze_sha256,
    }
    raw_plan = helper.canonical_json(plan)
    plan_sha = runner.sha(raw_plan)
    template = payload[runner.SELF].replace(b"\r\n", b"\n")
    token = b'ROOT_PLAN_SHA256 = "UNBOUND"'
    if template.count(token) != 1:
        raise ValueError("One source binding required")
    rendered = template.replace(token, ("ROOT_PLAN_SHA256 = " + json.dumps(plan_sha)).encode())
    compile(rendered, "etf_cost_runner.py", "exec")
    output.mkdir(parents=True)
    package, kernel = output / "payload", output / "kernel"
    package.mkdir()
    kernel.mkdir()
    for name, raw in payload.items():
        (package / entries[name]["upload_name"]).write_bytes(raw)
    (package / runner.PLAN_NAME).write_bytes(raw_plan)
    (kernel / "etf_cost_runner.py").write_bytes(rendered)
    metadata = {
        "id": kernel_id,
        "title": "ETF whole-period cost and ownership study",
        "code_file": "etf_cost_runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_id, helper.DATA_ID],
        "kernel_sources": [],
    }
    helper.write_json(kernel / "kernel-metadata.json", metadata)
    (output / "upload-metadata").mkdir()
    helper.write_json(
        output / "upload-metadata/dataset-metadata.json",
        {
            "id": dataset_id,
            "title": "ETF cost and ownership plain study source",
            "licenses": [{"name": "other"}],
        },
    )
    receipt = {
        "source_dataset": dataset_id,
        "kernel_id": kernel_id,
        "plan_sha256": plan_sha,
        "runner_raw_sha256": runner.sha(rendered),
        "payload_sha256": {p.name: runner.sha(p.read_bytes()) for p in package.iterdir()},
    }
    helper.write_json(output / "stage-receipt.json", receipt)
    if inventory(repo) != actual:
        raise ValueError("Source changed during staging")
    return receipt
