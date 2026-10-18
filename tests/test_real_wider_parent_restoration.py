"""Actual immutable wider metadata only; no generated or loaded Parquet values."""

import copy
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PRIVATE = PROJECT.parent / "quant-research-lab-private"
pytestmark = pytest.mark.real_data


@pytest.fixture
def actual_parent():
    root = PRIVATE / "stock-wider-publication-v3/terminal-small-output-v3"
    runtime_path = PRIVATE / (
        "holdout-adapter-smoke-publication-v1/terminal-small-v1/"
        "holdout-adapter-smoke/engineering/COMPLETE.json"
    )
    paths = {
        "panel": root / "wider-run/wider-panel/manifest.json",
        "study": root / "wider-run/stock-wider-study/manifest.json",
        "workflow": root / "supplemental/wider-run/workflow-manifest.json",
    }
    if not all(p.exists() for p in [*paths.values(), runtime_path]):
        pytest.skip("Supply actual verified immutable version3 JSON metadata")
    spec = importlib.util.spec_from_file_location(
        "actual_wider_transport", PROJECT / "cloud/shared_holdout_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module,
        {k: p.read_bytes() for k, p in paths.items()},
        json.loads(runtime_path.read_bytes())["runtime"],
        root,
    )


def test_actual180_no_kind_metadata_derives_exact_roles_without_numerical_imports(
    actual_parent, monkeypatch
):
    import builtins

    module, raw, runtime, _ = actual_parent
    original_import = builtins.__import__

    def no_native(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "pyarrow", "lightgbm", "sklearn"}:
            raise AssertionError("Opaque restoration must not import numerical/native readers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_native)
    normalized = module.authenticate_wider_metadata(raw, project=PROJECT, expected_runtime=runtime)
    panel = json.loads(raw["panel"])
    assert len(normalized) == 180 and all("kind" not in r for r in panel["outputs"])
    assert sum(r["bytes"] for r in normalized) == 32266950
    assert sum(r["rows"] for r in normalized) == 71309
    assert normalized[0]["session"] == "2000-01-31" and normalized[-1]["session"] == "2014-12-31"
    assert all(r["kind"] == "wider_features" for r in normalized)
    assert {r["path"]: r["sha256"] for r in normalized} == {
        r["path"]: r["sha256"] for r in panel["outputs"]
    }


def test_actual_metadata_missing_double_and_genuine_future_partition_refused(actual_parent):
    module, raw, _, _ = actual_parent
    panel = json.loads(raw["panel"])
    missing = copy.deepcopy(panel)
    missing["outputs"].pop()
    with pytest.raises(ValueError, match="EXACT_ORIGINAL_180"):
        module.normalize_wider_partitions(missing)
    duplicate = copy.deepcopy(panel)
    duplicate["outputs"].append(copy.deepcopy(panel["outputs"][0]))
    with pytest.raises(ValueError, match="DUPLICATE_OR_DOUBLE"):
        module.normalize_wider_partitions(duplicate)
    original = PROJECT / "data/processed/stock-monthly-v2/manifest.json"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    )
    future = next(
        r
        for r in json.loads(original.read_bytes())["outputs"]
        if r.get("kind") == "features" and Path(r["path"]).stem == "2015-01-30"
    )
    changed = copy.deepcopy(panel)
    changed["outputs"].append({k: future[k] for k in ("path", "bytes", "rows", "sha256")})
    with pytest.raises(ValueError, match="NO_FUTURE"):
        module.normalize_wider_partitions(changed)


def test_actual_parent_wrong_role_or_host_runtime_cannot_authenticate(actual_parent):
    module, raw, runtime, _ = actual_parent
    with pytest.raises(ValueError, match="IMMUTABLE_WIDER_METADATA_DIFFERS"):
        module.authenticate_wider_metadata(
            {**raw, "panel": raw["study"]}, project=PROJECT, expected_runtime=runtime
        )
    with pytest.raises(ValueError, match="UNCHANGED_PARENT_NUMERICAL_RUNTIME"):
        module.authenticate_wider_metadata(
            raw, project=PROJECT, expected_runtime={**runtime, "machine": platform.platform()}
        )


def test_actual_small_only_review_cannot_authorize_copy_or_native_access(
    actual_parent, monkeypatch, tmp_path
):
    module, raw, runtime, root = actual_parent
    review = (root / "root-independent-small-review-v3-bound.json").read_bytes()
    binding = {
        "parent": module.WIDER_PARENT,
        "destination": "wider",
        "full_replay_review_sha256": hashlib.sha256(review).hexdigest(),
    }

    def no_access(*args, **kwargs):
        raise AssertionError("Pending full replay must stop before any parent native file lookup")

    monkeypatch.setattr(module, "discover_bound", no_access)
    destination = tmp_path / "not-created-wider"
    with pytest.raises(ValueError, match="FULL_PARENT_REPLAY_APPROVAL_REQUIRED"):
        module.restore_wider_parent(
            root,
            destination,
            binding,
            raw,
            review,
            project=PROJECT,
            expected_runtime=runtime,
            budget=lambda: 1,
        )
    assert not destination.exists()
