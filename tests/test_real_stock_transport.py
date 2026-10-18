"""Actual full prepared-panel transport checks; no source staging or model fits."""

import ast
import json
import os
import sys
import time
from pathlib import Path

import pytest

from cloud import stage_stock_development as stager
from cloud import stock_development_runner as runner

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"
DATA_FREEZE_SHA = "b24b7e20a90c5e74d7921aa643a8caeb1d9ca38d237c3fe14732407d50063d04"


def test_actual_rank_stage_authentication_and_explicit_spec():
    mount = PRIVATE / "stock-rank-target-cloud-stage-v2/payload"
    path = mount / "stock-development-plan.json"
    if not path.is_file():
        pytest.skip("Actual reviewed rank-target stage required")
    raw = path.read_bytes()
    assert runner.sha(raw) == "d3e0e2c53859cd30dd2792a79db5bac92af8f3d62e16138a0420bc1ecbea84c7"
    plan = json.loads(raw)
    before = set(sys.modules)
    authenticated = runner.authenticate_package(plan, mount)
    assert len(authenticated) == 28
    assert plan["stock_spec_path"] == "configs/stock-rank-target-development.toml"
    assert not any(
        name.startswith(("quantlab", "numpy", "pyarrow", "sklearn", "lightgbm"))
        for name in set(sys.modules) - before
    )
    for invalid in (None, "../outside.toml", "configs/unregistered.toml"):
        with pytest.raises(runner.TransportError, match="EXPLICIT_REGISTERED_STOCK_SPEC"):
            runner.validate_plan({**plan, "stock_spec_path": invalid})


def test_actual_exact_ast_closure_and_locked_ml_runtime():
    assert set(stager.local_import_closure(ROOT)) == set(runner.SOURCE_FILES)
    inventory = stager.production_inventory(ROOT)
    assert len(inventory) == 26
    assert set(inventory) == set(
        runner.SOURCE_FILES + runner.CONFIG_FILES + runner.PROJECT_FILES + runner.TRANSPORT_FILES
    )
    assert len(set(stager.flat_name(name) for name in inventory)) == len(inventory)
    for name, digest in runner.PREPARATION_SOURCES.items():
        section = "stocks" if name in {"features", "labels", "prepare"} else "data"
        assert inventory[f"src/quantlab/{section}/{name}.py"]["sha256"] == digest
    runtime = stager.runtime_from_lock((ROOT / "uv.lock").read_bytes())
    assert set(runtime) == set(runner.RUNTIME_NAMES)
    assert runtime["lightgbm"] == "4.7.0"
    assert runtime["threadpoolctl"] == "3.7.0"
    assert runtime["exchange-calendars"] == "4.13.2"


def test_development_initial_and_final_source_checks_use_same_complete_closure():
    source = (ROOT / "src/quantlab/stocks/development.py").read_bytes()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assignments = {
        target.id: node.value
        for node in ast.walk(functions["run_stock_development"])
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in ("closure", "final_closure")
    }
    assert set(assignments) == {"closure", "final_closure"}
    assert all(
        ast.unparse(value) == "development_source_closure()" for value in assignments.values()
    )


def test_actual_template_only_binding_without_staging():
    raw = (ROOT / "cloud/stock_development_runner.py").read_bytes()
    pin = runner.sha((ROOT / "configs/stock-development.toml").read_bytes())
    rendered = stager.render_runner(raw, pin)
    assert runner.canonical_runner(raw) == runner.canonical_runner(rendered)
    assert runner.canonical_runner(rendered.replace(b"\n", b"\r\n")) == runner.canonical_runner(raw)
    compile(rendered, "actual-stock-render-check.py", "exec")


@pytest.fixture(scope="module")
def actual_transport():
    directory = Path(
        os.environ.get("QUANTLAB_STOCK_TRANSPORT", PRIVATE / "stock-panel-transport-v1")
    )
    manifest, container = directory / "transport-manifest.json", directory / runner.CONTAINER_NAME
    if not manifest.is_file() or not container.is_file():
        pytest.skip("Actual completed full-stock panel transport required")
    raw = runner.read_bound(manifest, runner.TRANSPORT_SHA)
    return container, raw


@pytest.mark.real_data
def test_actual_whole_prepared_container_and_648_members_before_any_native_parse(
    actual_transport, tmp_path
):
    container, raw = actual_transport
    output = tmp_path / "authenticated-prepared-input"
    before = set(sys.modules)
    proof = runner.extract_data(container, raw, output)
    assert proof["source_type"] == "completed_full_prepared_stock_panel"
    assert proof["binary_bytes"] == 507964547
    assert proof["authenticated_members"] == 648
    assert proof["prepared_manifest_sha256"] == runner.MANIFEST_SHA
    assert proof["original_archive_sha256"] == runner.ARCHIVE_SHA
    assert proof["outcome_value_parsing"] is False
    assert not any(
        name.startswith(("quantlab", "numpy", "pyarrow", "sklearn", "lightgbm"))
        for name in set(sys.modules) - before
    )
    expected = json.loads(raw)["members"]
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == set(
        expected
    )
    for name, item in expected.items():
        path = output / name
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as stream:
            assert runner.hash_stream(stream) == item["sha256"]


def test_actual_local_prepared_transport_cannot_authorize_remote_run(actual_transport, tmp_path):
    container, raw = actual_transport
    metadata = container.parent / "transport-manifest.json"
    actual = json.loads(raw)
    with pytest.raises(runner.TransportError, match="ACTUAL_READY_PRIVATE_PREPARED"):
        runner.validate_data_freeze(actual)
    with pytest.raises(runner.TransportError, match="ACTUAL_READY_PRIVATE_RESUME"):
        runner.validate_resume_freeze(actual, None)
    output = tmp_path / "must-not-stage"
    with pytest.raises(runner.TransportError, match="ACTUAL_READY_PRIVATE_PREPARED"):
        stager.build_stage(
            ROOT,
            output,
            data_freeze_path=metadata,
            expected_data_freeze_sha256=runner.sha(raw),
            source_freeze_path=metadata,
            expected_source_freeze_sha256=runner.sha(raw),
            source_dataset_id=None,
            kernel_id=None,
        )
    assert not output.exists()


def test_actual_ready_prepared_data_cannot_replace_missing_source_freeze(tmp_path):
    path = Path(
        os.environ.get(
            "QUANTLAB_STOCK_DATA_FREEZE",
            PRIVATE / "stock-panel-publication-v1/root-data-freeze-v1.json",
        )
    )
    if not path.is_file():
        pytest.skip("Actual root remote prepared-panel freeze not yet available")
    raw = runner.read_bound(path, DATA_FREEZE_SHA)
    value = json.loads(raw)
    runner.validate_data_freeze(value)
    assert value["dataset_id"] == 12657513 and value["version"] == 1
    output = tmp_path / "must-not-stage-without-root-source"
    with pytest.raises(FileNotFoundError):
        stager.build_stage(
            ROOT,
            output,
            data_freeze_path=path,
            expected_data_freeze_sha256=runner.sha(raw),
            source_freeze_path=tmp_path / "absent-root-stock-source-freeze.json",
            expected_source_freeze_sha256=runner.sha((ROOT / runner.CONFIG_FILES[0]).read_bytes()),
            source_dataset_id=None,
            kernel_id=None,
        )
    assert not output.exists()


def test_expired_actual_clock_retains_failure_before_install(tmp_path, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("No install or fitting in software transport checks")

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        runner.bootstrap_with_receipt(
            tmp_path / "private", ROOT, tmp_path, deadline=time.monotonic() - 1
        )
    assert calls == []
    failure = json.loads((tmp_path / "bootstrap-status.json").read_bytes())
    assert failure["complete"] is False and failure["phase"] == "bootstrap_failed"
    assert not (tmp_path / "COMPLETE.json").exists()


def test_unbound_runner_fails_before_source_or_native_import(tmp_path):
    before = set(sys.modules)
    with pytest.raises(runner.TransportError, match="ROOT_BINDINGS_REQUIRED"):
        runner.run(tmp_path / "absent-inputs", tmp_path / "failed-output")
    added = set(sys.modules) - before
    assert not any(name.startswith(("quantlab", "numpy", "pyarrow", "lightgbm")) for name in added)
    assert (tmp_path / "failed-output/FAILURE.json").is_file()
    assert not (tmp_path / "failed-output/COMPLETE.json").exists()


def test_actual_source_orders_authentication_and_private_ml_settings(tmp_path):
    source = (ROOT / "cloud/stock_development_runner.py").read_bytes()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    extract = functions["extract_data"]
    calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(extract) if isinstance(n, ast.Call)]
    assert min(line for line, name in calls if name == "hash_stream") < min(
        line for line, name in calls if name == "zipfile.ZipFile"
    )
    child = functions["child"]
    first_project_import = min(
        n.lineno
        for n in ast.walk(child)
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("quantlab")
    )
    assert (
        min(
            n.lineno
            for n in ast.walk(child)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "check_project"
        )
        < first_project_import
    )
    settings = runner.private_environment(tmp_path / "private", tmp_path / "private/project")
    assert settings["OPENBLAS_NUM_THREADS"] == settings["OMP_NUM_THREADS"] == "4"
    assert settings["UV_CACHE_DIR"] == str(tmp_path / "private/uv-cache")
    assert settings["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "private/runtime/.venv")
    bootstrap = ast.unparse(functions["bootstrap"])
    assert "'--group', 'ml'" in bootstrap and "'--locked'" in bootstrap
