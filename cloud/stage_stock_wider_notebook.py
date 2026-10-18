"""Source-only private notebook stager; actual approval is required before mkdir.

No provider API is called. A root-issued source freeze must bind the exact AST
closure, runtime, three already published data freezes and derived b46 metrics.
"""

import ast
import hashlib
import json
import re
import time
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

from cloud import stock_wider_notebook_bootstrap as worker
from cloud.build_stock_wider_notebook import build_notebook

SOURCE_SCHEMA = "quant_stock_wider_source_freeze_v1"
NORMALIZED_SHA = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
PREPARED_SHA = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
REFERENCE_SHA = "b46c22e5a6f46330668f6f3870c0c8cc2ddde15c283d3f9af8637f93557492d6"
ARCHIVE_SHA = "d9317c8fb2d63b9b00db5f933b6c9639d2bf7ea3b918169bb5cec5903dce85a1"
EXTRA = (
    "configs/stock-feature-portability-whitelist.json",
    "configs/stock-development.toml",
    "configs/stock-wider-development.toml",
    "cloud/build_stock_wider_notebook.py",
    "cloud/stock_wider_notebook_bootstrap.py",
    "cloud/stage_stock_wider_notebook.py",
    "cloud/stock_development_runner.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(path, expected):
    raw = Path(path).read_bytes()
    require(
        isinstance(expected, str)
        and re.fullmatch("[0-9a-f]{64}", expected)
        and sha(raw) == expected,
        "Actual externally bound input SHA required",
    )
    return raw


def import_closure(repo):
    src = Path(repo) / "src"
    pending = [src / "quantlab/stocks/wider_study.py", src / "quantlab/stocks/wider_panel.py"]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for parent in path.parents:
            if parent == src:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer != path:
                pending.append(initializer)
        package = ".".join(path.relative_to(src).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join((*parts[: len(parts) - node.level + 1], base)).strip(".")
                names = [base, *(base + "." + a.name for a in node.names)]
            for name in names:
                if name == "quantlab" or name.startswith("quantlab."):
                    target = src.joinpath(*name.split("."))
                    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                        if candidate.is_file():
                            pending.append(candidate)
    return tuple(sorted(p.relative_to(Path(repo)).as_posix() for p in found))


def production_inventory(repo):
    repo = Path(repo)
    names = import_closure(repo) + EXTRA
    require(len(names) == len(set(names)), "Unique exact production closure required")
    return {
        name: {"bytes": (repo / name).stat().st_size, "sha256": sha((repo / name).read_bytes())}
        for name in names
    }


def runtime_from_lock(path):
    lock = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for name in (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "threadpoolctl",
        "exchange-calendars",
        "matplotlib",
    ):
        found = [p["version"] for p in lock["package"] if p["name"] == name]
        require(len(found) == 1, "One exact locked dependency version required")
        result[name] = found[0]
    return result


def _identity(value, role):
    if role == "original_market":
        dataset = value["dataset"]
        identity = {
            "dataset": dataset["canonical"],
            "dataset_id": dataset["id"],
            "version": dataset["version"],
            "private": dataset["is_private"],
        }
        require(
            value["schema"] == "quant_market_data_freeze_v1"
            and value["snapshot_manifest_sha256"] == NORMALIZED_SHA
            and value["archive_sha256"] == ARCHIVE_SHA
            and dataset["status"] == "READY",
            "Original whole market source required",
        )
    else:
        identity = {k: value[k] for k in ("dataset", "dataset_id", "version", "private")}
        if role == "original_prepared_panel":
            require(
                value["schema"] == "quant_stock_prepared_data_freeze_v1"
                and value["prepared_manifest_sha256"] == PREPARED_SHA
                and value["original_archive_sha256"] == ARCHIVE_SHA
                and value["source_normalized_manifest_sha256"] == NORMALIZED_SHA,
                "Original full prepared source required",
            )
        else:
            require(
                value["schema"] == "quant_stock_economic_prediction_data_freeze_v1"
                and value["upstream_manifest_sha256"] == REFERENCE_SHA
                and value["prediction_months"] == 96
                and value["native_models_packaged"] is False,
                "All96 original scalar predictions required",
            )
    require(
        value["status"] == "READY"
        and identity["private"] is True
        and identity["version"] == 1
        and type(identity["dataset_id"]) is int
        and identity["dataset_id"] > 0,
        "Actual private READY numeric-v1 identity required",
    )
    return identity


def _transport(role, request, freeze):
    container = Path(request["container_path"])
    transport = Path(request["transport_path"])
    if role == "original_market":
        item = freeze["remote_files"]["market-v1.bin"]
        expected_transport = freeze["remote_files"]["transport-manifest.json"]["sha256"]
    else:
        item = freeze["container"]
        expected_transport = freeze["transport_manifest_sha256"]
    require(container.stat().st_size == item["bytes"], "Actual full container size required")
    with container.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"],
            "Actual whole input container SHA mismatch",
        )
    manifest = json.loads(read_bound(transport, expected_transport))
    source = manifest["source_files"] if role == "original_market" else manifest["members"]
    if role == "original_market":
        with zipfile.ZipFile(container) as archive:
            members = [
                {"path": info.filename, "sha256": source[info.filename], "bytes": info.file_size}
                for info in archive.infolist()
            ]
        require(
            manifest["source_manifest_sha256"] == NORMALIZED_SHA
            and manifest["archive_sha256"] == ARCHIVE_SHA,
            "Original normalized upstream source pins required",
        )
    elif isinstance(source, dict):
        members = [
            {"path": name, "sha256": value["sha256"], "bytes": value["bytes"]}
            for name, value in sorted(source.items())
        ]
    else:
        members = [{k: value[k] for k in ("path", "sha256", "bytes")} for value in source]
    require(
        len(members)
        == {"original_market": 8515, "original_prepared_panel": 648, "original_raw_reference": 99}[
            role
        ],
        "Exact whole bundle scope required",
    )
    # Byte authentication/whole archive roundtrip, no Arrow/native model parsing.
    worker._inspect(container, members, time.monotonic() + 3600)
    return {
        "name": container.name,
        "sha256": item["sha256"],
        "bytes": item["bytes"],
        "members": members,
        "transport_manifest_sha256": expected_transport,
    }


def build_stage(
    repo,
    output,
    *,
    source_freeze_path,
    expected_source_freeze_sha256,
    data_inputs,
    reference_root,
    source_dataset="aikyatansinha/portfolio-quant-stock-wider-source-v1",
    kernel_id="aikyatansinha/portfolio-quant-stock-wider-main-v1",
):
    """Create one local reviewable stage only after every real approval/input guard."""
    repo, output = Path(repo), Path(output)
    # Deliberately first: absent/UNBOUND approval never creates a stage directory.
    approval_raw = read_bound(source_freeze_path, expected_source_freeze_sha256)
    approval = json.loads(approval_raw)
    require(
        approval["schema"] == SOURCE_SCHEMA and approval["status"] == "READY",
        "Actual root READY wider source freeze required",
    )
    frozen_at = datetime.fromisoformat(approval["frozen_at"])
    require(frozen_at.utcoffset() is not None, "Actual aware source registration clock required")
    inventory, runtime = production_inventory(repo), runtime_from_lock(repo / "uv.lock")
    require(
        approval["production_inventory"] == inventory
        and approval["runtime"] == runtime
        and approval["python"] == "3.12.15"
        and approval["threads"] == 4
        and approval["service_seconds"] == 36000,
        "Exact approved closure/runtime/budget required",
    )
    require(
        set(data_inputs) == set(worker.INPUT_ROLES), "Exactly three actual data inputs required"
    )
    freezes, bindings, inputs, freeze_hashes = {}, {}, {}, {}
    for role, request in data_inputs.items():
        freeze_raw = read_bound(request["freeze_path"], request["freeze_sha256"])
        freeze = json.loads(freeze_raw)
        freezes[role], bindings[role] = freeze, _identity(freeze, role)
        freeze_hashes[role] = request["freeze_sha256"]
        inputs[role] = _transport(role, request, freeze)
    require(
        approval["data_freeze_sha256"] == freeze_hashes,
        "Actual source approval must bind all three data freeze bytes",
    )
    reference_root = Path(reference_root)
    ref = json.loads(read_bound(reference_root / "manifest.json", REFERENCE_SHA))
    metric = next(a for a in ref["artifacts"] if a["path"] == "metrics.json")
    metrics_raw = read_bound(reference_root / "metrics.json", metric["sha256"])
    require(len(metrics_raw) == metric["bytes"], "Original derived metrics size mismatch")
    require(
        approval["reference_metrics"] == {"sha256": metric["sha256"], "bytes": metric["bytes"]},
        "Derived reference metrics must be explicitly approved",
    )
    for canonical in (source_dataset, kernel_id):
        require(
            re.fullmatch("aikyatansinha/[a-z0-9-]{1,50}", canonical),
            "Owned bounded canonical required",
        )
    package = {}
    for name, item in inventory.items():
        package[name] = {**item, "path": name, "package_name": name.replace("/", "__")}
    package["reference/metrics.json"] = {
        "path": "reference/metrics.json",
        "package_name": "reference__metrics.json",
        "bytes": metric["bytes"],
        "sha256": metric["sha256"],
    }
    plan = {
        "schema": "quant_stock_wider_notebook_plan_v1",
        "private": True,
        "service_seconds": 36000,
        "holdout_evaluated": False,
        "source_approval": approval,
        "source_approval_sha256": expected_source_freeze_sha256,
        "source_files": list(package.values()),
        "data_inputs": inputs,
        "data_freezes": freezes,
        "data_bindings": bindings,
        "data_freeze_sha256": freeze_hashes,
        "runtime": runtime,
        "normalized_manifest_sha256": NORMALIZED_SHA,
        "reference_manifest_sha256": REFERENCE_SHA,
        "roots": approval["input_roots"],
        "derived_reference_metrics": {
            "source_path": "reference/metrics.json",
            "sha256": metric["sha256"],
            "bytes": metric["bytes"],
        },
    }
    plan["minimum_free_tmp_bytes"] = (
        sum(m["bytes"] for item in inputs.values() for m in item["members"]) + 8 * 1024**3
    )
    require(production_inventory(repo) == inventory, "Source changed during preflight")
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir()
    for name, item in package.items():
        (payload / item["package_name"]).write_bytes(
            metrics_raw if name == "reference/metrics.json" else (repo / name).read_bytes()
        )
    plan_raw = encoded(plan)
    (payload / "stock-wider-plan.json").write_bytes(plan_raw)
    bootstrap = (repo / "cloud/stock_wider_notebook_bootstrap.py").read_text(encoding="utf-8")
    bootstrap = bootstrap.replace(
        'ROOT_PLAN_SHA256 = "UNBOUND"', 'ROOT_PLAN_SHA256 = "' + sha(plan_raw) + '"'
    )
    bootstrap += (
        "\nLOCKED_PYTHON, CONTEXT_PATH, LOCKED_ENV = "
        "run('/kaggle/input', '/kaggle/working/wider-run')\n"
    )
    bootstrap_path = output / "rendered-bootstrap.py"
    bootstrap_path.write_text(bootstrap, encoding="utf-8", newline="\n")
    kernel = output / "kernel"
    kernel.mkdir()
    notebook = build_notebook(
        kernel / "stock_wider_main.ipynb",
        project_root=repo,
        bootstrap_source=bootstrap_path,
        expected_bootstrap_sha256=sha(bootstrap.encode()),
    )
    metadata = {
        "id": kernel_id,
        "title": "Quant stock wider main workflow",
        "code_file": "stock_wider_main.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [bindings[r]["dataset"] for r in worker.INPUT_ROLES] + [source_dataset],
        "kernel_sources": [],
    }
    (kernel / "kernel-metadata.json").write_bytes(encoded(metadata))
    receipt = {
        "source_only": True,
        "provider_mutations": False,
        "plan_sha256": sha(plan_raw),
        "notebook": notebook,
        "kernel_metadata": metadata,
        "payload_files": {
            p.name: {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
            for p in sorted(payload.iterdir())
        },
        "production_inventory": inventory,
    }
    require(production_inventory(repo) == inventory, "Source changed while staging")
    (output / "stage-receipt.json").write_bytes(encoded(receipt))
    return receipt
