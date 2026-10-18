"""Plain transport source/metadata checks; no strategy or invented price runs."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cloud import etf_development_runner as runner
from cloud import risk_forecast_runner as risk_runner
from cloud import stage_etf_development as stager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "quant-research-lab-private"
YAHOO_DATA_FREEZE_SHA = "6dbb5162e047077b962dbe60f815f8a623d21f70043a61e9f2e5da210449fb0f"


def test_actual_remote_yahoo_freeze_cannot_replace_missing_source_approval(tmp_path):
    path = Path(
        os.environ.get(
            "QUANTLAB_YAHOO_DATA_FREEZE",
            PRIVATE / "yahoo-etf-transport-v1/root-data-freeze-v1.json",
        )
    )
    if not path.is_file():
        pytest.skip("Actual root remote Yahoo release approval required")
    raw = runner.read_bound(path, YAHOO_DATA_FREEZE_SHA)
    value = json.loads(raw)
    runner.validate_data_freeze(value)
    risk_runner.validate_etf_data_freeze(value)
    assert value["dataset_id"] == 12657227 and value["version"] == 1
    output = tmp_path / "must-not-stage-without-source-approval"
    with pytest.raises(FileNotFoundError):
        stager.build_stage(
            ROOT,
            output,
            data_freeze_path=path,
            expected_data_freeze_sha256=YAHOO_DATA_FREEZE_SHA,
            source_freeze_path=tmp_path / "absent-root-source-freeze.json",
            expected_source_freeze_sha256=runner.sha(
                (ROOT / "cloud/etf_development_runner.py").read_bytes()
            ),
            source_dataset_id=None,
            kernel_id=None,
        )
    assert not output.exists()


@pytest.mark.real_data
@pytest.mark.parametrize("kind", ["etf", "risk"])
def test_actual_whole_yahoo_container_and_all_members_authenticated_before_use(tmp_path, kind):
    directory = Path(
        os.environ.get("QUANTLAB_YAHOO_TRANSPORT", PRIVATE / "yahoo-etf-transport-v1/payload")
    )
    manifest = directory / "transport-manifest.json"
    binary = directory / runner.CONTAINER_NAME
    if not binary.is_file() or not manifest.is_file():
        pytest.skip("Actual complete Yahoo transport required")
    raw = runner.read_bound(manifest, runner.TRANSPORT_SHA, 5639)
    destination = tmp_path / "authenticated-real-source"
    extract = runner.extract_data if kind == "etf" else risk_runner.extract_etf_data
    before = set(sys.modules)
    proof = extract(binary, raw, destination)
    assert proof["authenticated_members"] == 33
    assert proof["release_manifest_sha256"] == runner.MANIFEST_SHA
    assert proof["raw_bundle_manifest_sha256"] == runner.RAW_BUNDLE_SHA
    assert proof["price_basis"] == runner.PRICE_BASIS
    assert proof["archive_sha256"] is None
    assert not any(
        name.startswith(("quantlab", "numpy", "pyarrow", "sklearn"))
        for name in set(sys.modules) - before
    )
    for name, item in json.loads(raw)["members"].items():
        extracted = destination / name
        assert extracted.stat().st_size == item["bytes"]
        assert runner.sha(extracted.read_bytes()) == item["sha256"]


def test_local_yahoo_transport_is_not_remote_ready_approval(tmp_path):
    path = PRIVATE / "yahoo-etf-transport-v1/payload/transport-manifest.json"
    if not path.is_file():
        pytest.skip("Actual Yahoo packaging metadata required")
    raw = path.read_bytes()
    for validate in (runner.validate_data_freeze, risk_runner.validate_etf_data_freeze):
        with pytest.raises(ValueError, match="ACTUAL_READY_PRIVATE"):
            validate(json.loads(raw))
    output = tmp_path / "must-not-stage-local-data"
    with pytest.raises(runner.TransportError, match="ACTUAL_READY_PRIVATE"):
        stager.build_stage(
            ROOT,
            output,
            data_freeze_path=path,
            expected_data_freeze_sha256=runner.sha(raw),
            source_freeze_path=path,
            expected_source_freeze_sha256=runner.sha(raw),
            source_dataset_id=None,
            kernel_id=None,
        )
    assert not output.exists()


def test_child_failure_records_actual_guard_context_and_bounded_software_error(tmp_path):
    try:
        runner.require(False, "ACTUAL_SOURCE_GUARD_DIAGNOSTIC")
    except runner.TransportError as error:
        detail = runner.child_failure_detail(error)
    assert detail["complete"] is False
    assert detail["error_type"] == "TransportError"
    assert detail["guard_reason"] == "ACTUAL_SOURCE_GUARD_DIAGNOSTIC"
    assert "test_child_failure_records" in detail["traceback"]
    try:
        raise ValueError((ROOT / "cloud/etf_development_runner.py").read_text() * 2)
    except ValueError as error:
        detail = runner.child_failure_detail(error)
    assert detail["error_type"] == "ValueError"
    assert len(detail["error_message"]) == 2000
    assert len(detail["traceback"]) == 12000
    runner.write_json(tmp_path / "child-failure.json", detail)
    assert json.loads((tmp_path / "child-failure.json").read_bytes()) == detail


@pytest.mark.real_data
def test_support_diagnostic_allowlist_uses_real_are_failure():
    from datetime import date

    from quantlab.backtest.engine import Decision, run_history
    from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
    from quantlab.data.calendar import XNYSSchedule

    archive = Path(os.environ.get("QUANTLAB_ARCHIVE", ROOT / "data/raw/archive.zip"))
    if not archive.exists():
        pytest.skip("Supplied real archive required")
    with MarketArchive(archive, SUPPLIED_ARCHIVE_SHA256) as source:
        bars = source.bars("stock", "ARE")
    asset = bars[0].asset_id
    schedule = XNYSSchedule(date(2006, 5, 30), date(2006, 6, 9))

    def decision(view, state):
        return Decision(
            "REBALANCE",
            {asset: 1},
            view.cutoff,
            view.next_open_at,
            "fixed real source-quality event check",
        )

    actual = run_history(
        {asset: bars},
        schedule,
        start=date(2006, 6, 5),
        end=date(2006, 6, 6),
        initial_cash=10000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=decision,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )

    def compare_controls(result):
        policy = "real_are_software_diagnostic"
        assert policy
        raise ValueError("Incomplete economic support; no complete ETF comparison permitted")

    try:
        compare_controls(actual)
    except ValueError as error:
        detail = runner.child_failure_detail(error)
    context = detail["support_failure"]
    assert context["failure"]["session"] == "2006-06-06"
    assert context["failure"]["event"] == "close"
    assert context["failure"]["unresolved_assets"] == [asset]
    assert context["last_marked_session"] == "2006-06-05"
    assert context["incomplete_open_session"] == "2006-06-06"
    assert context["marked_record_count"] == 1
    assert set(context) == {
        "policy",
        "marked_record_count",
        "last_marked_session",
        "incomplete_open_session",
        "failure",
    }
    assert "histories" not in json.dumps(context) and "nav" not in json.dumps(context)


def test_actual_source_closure_and_plain_payload_inventory():
    assert set(stager.local_import_closure(ROOT)) == set(runner.SOURCE_FILES)
    inventory = stager.production_inventory(ROOT)
    assert set(inventory) == set(
        runner.SOURCE_FILES + runner.PROJECT_FILES + runner.TRANSPORT_FILES
    )
    flats = [stager.flat_name(name) for name in inventory]
    assert len(set(flats)) == len(flats)
    assert all("/" not in flat and flat != "dataset-metadata.json" for flat in flats)
    runtime = stager.runtime_from_lock((ROOT / "uv.lock").read_bytes())
    assert runtime["pyarrow"] == "26.0.0"
    assert runtime["exchange-calendars"] == "4.13.2"


def test_actual_template_roundtrip_and_code_edit_rejection():
    raw = (ROOT / "cloud/etf_development_runner.py").read_bytes()
    # A real existing artifact hash exercises text rendering only; this is not
    # a staged plan or an approval to run anything.
    pin = runner.sha((ROOT / "configs/research.toml").read_bytes())
    rendered = stager.render_runner(raw, pin)
    assert runner.sha(runner.canonical_runner(rendered)) == runner.sha(runner.canonical_runner(raw))
    assert runner.canonical_runner(rendered.replace(b"\n", b"\r\n")) == runner.canonical_runner(raw)
    assert runner.canonical_runner(
        rendered + b"\n# software source edit probe\n"
    ) != runner.canonical_runner(raw)
    with pytest.raises(runner.TransportError, match="LONE_CR"):
        runner.canonical_runner(rendered + b"\r")
    compile(rendered, "actual-rendered-runner.py", "exec")


def test_actual_staged_data_is_not_a_root_remote_freeze(tmp_path):
    path = PRIVATE / "kaggle-market-v1-binary/transport-manifest.json"
    if not path.is_file():
        pytest.skip("Actual binary transport metadata required")
    raw = path.read_bytes()
    output = tmp_path / "forbidden-stage"
    with pytest.raises(runner.TransportError, match="ACTUAL_READY_PRIVATE"):
        stager.build_stage(
            ROOT,
            output,
            data_freeze_path=path,
            expected_data_freeze_sha256=runner.sha(raw),
            source_freeze_path=path,
            expected_source_freeze_sha256=runner.sha(raw),
            source_dataset_id="aikyatansinha/portfolio-quant-etf-source-v1",
            kernel_id="aikyatansinha/portfolio-quant-etf-development-v1",
        )
    assert not output.exists()


@pytest.mark.real_data
def test_actual_container_metadata_exact_without_rehashing_whole_binary():
    directory = Path(
        os.environ.get("QUANTLAB_YAHOO_TRANSPORT", PRIVATE / "yahoo-etf-transport-v1/payload")
    )
    manifest = directory / "transport-manifest.json"
    binary = directory / runner.CONTAINER_NAME
    if not binary.is_file() or not manifest.is_file():
        pytest.skip("Actual binary transport required")
    raw = runner.read_bound(manifest, runner.TRANSPORT_SHA, 5639)
    value = json.loads(raw)
    # This bounded test inspects the real central directory. The production
    # extract_data path separately requires whole-binary SHA before opening ZIP.
    with zipfile.ZipFile(binary) as archive:
        members = runner.inspect_container(archive, value)
    assert len(members) == 33
    normalized = (
        Path(os.environ.get("QUANTLAB_YAHOO_RELEASE", PRIVATE / "yahoo-etf-price-index-v1"))
        / "manifest.json"
    )
    assert value["release_manifest_sha256"] == hashlib.sha256(normalized.read_bytes()).hexdigest()
    assert value["raw_bundle_manifest_sha256"] == runner.RAW_BUNDLE_SHA


def test_bootstrap_fault_records_failure_before_any_subprocess_install(tmp_path, monkeypatch):
    calls = []

    def denied(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", denied)
    with pytest.raises(subprocess.CalledProcessError):
        runner.bootstrap_with_receipt(tmp_path / "private", ROOT, tmp_path)
    receipt = json.loads((tmp_path / "bootstrap-status.json").read_bytes())
    assert receipt["complete"] is False and receipt["phase"] == "bootstrap_failed"
    assert receipt["error_type"] == "CalledProcessError"
    assert len(calls) == 1 and calls[0][-1] == "uv==0.13.1"
    assert "--target" in calls[0]


def test_unbound_production_template_writes_failure_without_native_imports(tmp_path):
    path = ROOT / "cloud/etf_development_runner.py"
    spec = importlib.util.spec_from_file_location("actual_etf_template_probe", path)
    module = importlib.util.module_from_spec(spec)
    before = set(sys.modules)
    spec.loader.exec_module(module)
    output = tmp_path / "result"
    with pytest.raises(module.TransportError, match="ROOT_BINDINGS_REQUIRED"):
        module.run(tmp_path, output)
    failure = json.loads((output / "FAILURE.json").read_bytes())
    assert failure["complete"] is False
    assert not (output / "COMPLETE.json").exists()
    assert not any(
        name.startswith(("quantlab.", "pyarrow", "numpy")) for name in set(sys.modules) - before
    )


def test_source_orders_full_authentication_before_zip_and_child_native_imports():
    tree = ast.parse((ROOT / "cloud/etf_development_runner.py").read_bytes())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    extract = functions["extract_data"]
    calls = [
        (node.lineno, ast.unparse(node.func))
        for node in ast.walk(extract)
        if isinstance(node, ast.Call)
    ]
    hash_line = min(line for line, name in calls if name == "hash_stream")
    zip_line = min(line for line, name in calls if name == "zipfile.ZipFile")
    assert hash_line < zip_line
    child = functions["child"]
    native_line = min(
        node.lineno
        for node in ast.walk(child)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quantlab")
    )
    source_guard = min(
        node.lineno
        for node in ast.walk(child)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "check_project"
    )
    assert source_guard < native_line
    source = ast.unparse(child)
    assert "start_signal=date(2007, 2, 28)" in source
    assert "end=date(2014, 12, 31)" in source
