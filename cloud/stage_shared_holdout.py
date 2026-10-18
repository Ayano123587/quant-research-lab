"""Prospective one-shot final transport. No actual freeze or dispatch is created on import."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from cloud.shared_holdout_runner import (
    WIDER_METADATA,
    WIDER_PARENT,
    authenticate_wider_metadata,
    authenticate_wider_review,
    canonical_runner,
    module,
    require,
    safe_name,
    validate_descriptor,
)
from quantlab.evaluation.holdout import source_closure


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def exclusive(path, value):
    raw = encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def consume_dispatch(
    ledger, *, release_sha256, plan_sha256, root_stage_review_path, root_stage_review_sha256
):
    """The publisher calls once immediately BEFORE one push; failure never unconsumes."""
    ledger = Path(ledger)
    entry_raw = (ledger / "reserved.json").read_bytes()
    bound = json.loads((ledger / "bound-stage.json").read_bytes())
    require(
        bound["release_sha256"] == release_sha256
        and bound["plan_sha256"] == plan_sha256
        and bound["reservation_sha256"] == sha(entry_raw),
        "EXACT_RESERVED_STAGE_REQUIRED",
    )
    review_raw = Path(root_stage_review_path).read_bytes()
    require(sha(review_raw) == root_stage_review_sha256, "ACTUAL_ROOT_STAGE_REVIEW_REQUIRED")
    review = json.loads(review_raw)
    require(
        review["status"] == "PASS"
        and review["release_sha256"] == release_sha256
        and review["plan_sha256"] == plan_sha256,
        "ROOT_REVIEW_MUST_BIND_EXACT_RELEASE_AND_STAGE",
    )
    return exclusive(
        ledger / "consumed-before-push.json",
        {
            **bound,
            "root_stage_review_sha256": root_stage_review_sha256,
            "consumed_at": datetime.now(UTC).isoformat(),
            "automatic_retry": False,
        },
    )


def prepare(
    repo,
    output,
    *,
    release_path,
    expected_release_sha256,
    approval_path,
    expected_approval_sha256,
    project_artifacts,
    auxiliary_inputs,
    data_descriptors,
    source_dataset,
    kernel,
    service_budget_seconds,
    ledger,
    wider_request=None,
):
    """Requires actual reviewed release/data bytes; never guesses a future recipe/hash.

    data_descriptors: exact market/prepared/etf entries with freeze_path (local),
    freeze_sha256, dataset/id/version, container name/bytes/SHA, transport SHA,
    and destination relative to materialized inputs. The existing bound extractors
    independently enforce their immutable actual dataset/version/container identities.
    project_artifacts and auxiliary_inputs map frozen relative paths to actual files.
    """
    repo, output, ledger = Path(repo).resolve(), Path(output).resolve(), Path(ledger).resolve()
    require(not output.exists() and not ledger.exists(), "NEW_STAGE_AND_RELEASE_LEDGER_REQUIRED")
    release_raw, approval_raw = Path(release_path).read_bytes(), Path(approval_path).read_bytes()
    require(sha(release_raw) == expected_release_sha256, "ACTUAL_RELEASE_SHA_REQUIRED")
    require(sha(approval_raw) == expected_approval_sha256, "ACTUAL_ROOT_APPROVAL_SHA_REQUIRED")
    release, approval = json.loads(release_raw), json.loads(approval_raw)
    require(
        release["schema"] == "shared_quant_holdout_release_v1"
        and release["phase"] == "all_components_frozen_before_outcomes"
        and release["initial_signal"] == "2014-12-31"
        and release["terminal_close"] == "2017-11-10",
        "ACTUAL_SHARED_FINAL_RELEASE_REQUIRED",
    )
    require(
        approval["release_sha256"] == expected_release_sha256
        and approval["authorized_one_shared_evaluation"] is True,
        "ACTUAL_RELEASE_AUTHORIZATION_REQUIRED",
    )
    require(set(data_descriptors) == {"market", "prepared", "etf"}, "ALL_THREE_INPUTS_REQUIRED")
    require(0 < service_budget_seconds <= 39000, "EXPLICIT_SERVICE_BUDGET_REQUIRED")
    own = ("cloud/shared_holdout_runner.py", "cloud/stage_shared_holdout.py")
    helpers = (
        "cloud/stock_development_runner.py",
        "cloud/risk_forecast_runner.py",
        "cloud/etf_development_runner.py",
    )
    closure = source_closure(repo, ["src/quantlab/evaluation/holdout_pipeline.py"])
    required = {**closure, **{n: sha((repo / n).read_bytes()) for n in (*own, *helpers)}}
    require(
        all(release["source_sha256"].get(n) == d for n, d in required.items()),
        "EXTERNALLY_FROZEN_COMPLETE_CURRENT_SOURCE_REQUIRED",
    )
    files = {}
    for name, digest in release["source_sha256"].items():
        safe_name(name)
        path = repo / name
        require(
            path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == digest,
            "RELEASE_SOURCE_BYTE_DRIFT",
        )
        files[name] = path.read_bytes()
    for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md"):
        require(name in files, "LOCKED_PROJECT_FILES_MUST_BE_EXTERNALLY_FROZEN")
    lock = tomllib.loads(files["uv.lock"].decode())
    versions = {p["name"]: p["version"] for p in lock["package"] if "version" in p}
    runtime = release["runtime"]
    needed = {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "exchange-calendars",
        "threadpoolctl",
        "cvxpy",
        "clarabel",
        "osqp",
        "scs",
    }
    require(
        runtime["python"] == "3.12.15"
        and runtime["machine"] == "x86_64"
        and needed <= set(runtime["packages"])
        and all(versions[p] == v for p, v in runtime["packages"].items()),
        "BOTH_ML_AND_OPTIMIZATION_ACTUAL_LOCK_PINS_REQUIRED",
    )
    for name, source in project_artifacts.items():
        safe_name(name)
        raw = Path(source).read_bytes()
        require(name not in files or files[name] == raw, "PROJECT_ARTIFACT_COLLISION")
        files[name] = raw
    selected_wider = False
    for arm in release["portfolio_arms"] + release.get("forecast_arms", []):
        require(
            sha(files[arm["recipe_path"]]) == arm["recipe_sha256"], "ACTUAL_FROZEN_RECIPE_REQUIRED"
        )
        decision = json.loads(files[arm["recipe_path"]])["decision"]
        stock = decision.get("stock", decision)
        selected_wider |= stock.get("projection", {}).get("kind") == "registered_wider"
    wider_binding = release.get("wider_parent")
    require(
        selected_wider == (wider_binding is not None)
        and selected_wider == (wider_request is not None),
        "EXPLICIT_SELECTED_WIDER_PARENT_REQUIRED",
    )
    if selected_wider:
        require(
            not any(name == "wider" or name.startswith("wider/") for name in auxiliary_inputs),
            "WIDER_NATIVE_INPUTS_MUST_COME_ONLY_FROM_AUTHENTICATED_PARENT",
        )
        require(
            all(
                row["destination"] not in {"wider", "."}
                and not row["destination"].startswith("wider/")
                for row in data_descriptors.values()
            ),
            "WIDER_RESTORE_ROOT_MUST_BE_DISJOINT",
        )
        require(
            wider_binding["parent"] == WIDER_PARENT and wider_binding["destination"] == "wider",
            "ONLY_ACTUAL_IMMUTABLE_WIDER_PARENT_ALLOWED",
        )
        metadata = {
            role: Path(wider_request["metadata_paths"][role]).read_bytes()
            for role in WIDER_METADATA
        }
        expected = authenticate_wider_metadata(metadata, project=repo, expected_runtime=runtime)
        require(wider_binding["partitions"] == expected, "ROOT_FROZEN_WIDER_PARTITIONS_DIFFER")
        review_raw = Path(wider_request["full_replay_review_path"]).read_bytes()
        authenticate_wider_review(review_raw, wider_binding["full_replay_review_sha256"])
        for role, raw in metadata.items():
            name = "release/wider-parent/" + role + ".json"
            require(
                wider_binding["metadata_package_paths"][role] == name and name not in files,
                "UNAMBIGUOUS_WIDER_METADATA_PACKAGE_REQUIRED",
            )
            files[name] = raw
        name = "release/wider-parent/root-full-replay-review.json"
        require(
            wider_binding["review_package_path"] == name and name not in files,
            "EXACT_PARENT_REVIEW_PACKAGE_REQUIRED",
        )
        files[name] = review_raw
        expected_inputs = {
            "wider/manifest.json": {
                "bytes": len(metadata["panel"]),
                "sha256": WIDER_METADATA["panel"][1],
            },
            **{
                "wider/" + row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for row in expected
            },
        }
        actual_inputs = {
            r["path"]: {"bytes": r["bytes"], "sha256": r["sha256"]}
            for r in release["input_files"].values()
            if r["path"].startswith("wider/")
        }
        require(actual_inputs == expected_inputs, "ALL_180_WIDER_INPUT_FILES_MUST_BE_FROZEN")
    files["release/final-release.json"] = release_raw
    files["release/root-approval.json"] = approval_raw
    data, datasets = {}, [source_dataset]
    helper_paths = dict(zip(("prepared", "market", "etf"), helpers, strict=True))
    for kind, descriptor in data_descriptors.items():
        descriptor = dict(descriptor)
        local_freeze = Path(descriptor.pop("freeze_path"))
        raw = local_freeze.read_bytes()
        require(sha(raw) == descriptor.pop("freeze_sha256"), "ACTUAL_DATA_FREEZE_DIFFERS")
        freeze = json.loads(raw)
        helper = module(repo / helper_paths[kind], "frozen_stage_helper_" + kind)
        validate_descriptor(descriptor, freeze, helper, kind)
        require(
            freeze["private"] is True
            and freeze["status"] == "READY"
            and descriptor["version"] == 1,
            "ACTUAL_PRIVATE_VERSIONED_DATA_REQUIRED",
        )
        destination = safe_name(descriptor["destination"]).as_posix()
        require(
            not any(destination == d["destination"] for d in data.values()),
            "DISTINCT_INPUT_DESTINATIONS_REQUIRED",
        )
        name = "release/" + kind + "-data-freeze.json"
        files[name] = raw
        descriptor["freeze_path"] = name
        data[kind] = descriptor
        datasets.append(descriptor["dataset"])
    aux = {}
    for relative, path in auxiliary_inputs.items():
        safe_name(relative)
        raw = Path(path).read_bytes()
        records = [r for r in release["input_files"].values() if r["path"] == relative]
        require(
            len(records) == 1
            and records[0]["bytes"] == len(raw)
            and records[0]["sha256"] == sha(raw),
            "FROZEN_AUXILIARY_INPUT_REQUIRED",
        )
        name = "release/auxiliary/" + relative
        files[name] = raw
        aux[relative] = {"package_path": name}
    for name, raw in files.items():
        if name.endswith(".py"):
            ast.parse(raw)
    runner = files[own[0]]
    canonical = canonical_runner(runner)
    require(canonical == runner.replace(b"\r\n", b"\n"), "UNBOUND_SOURCE_TEMPLATE_REQUIRED")
    # The durable reservation is only created after all actual source/metadata gates pass.
    ledger.mkdir(parents=True, exist_ok=False)
    reservation = {
        "release_id": release["release_id"],
        "release_sha256": expected_release_sha256,
        "reserved_at": datetime.now(UTC).isoformat(),
        "scope": "one_shared_final_evaluation_no_automatic_retry",
    }
    reservation_sha = exclusive(ledger / "reserved.json", reservation)
    ticket = {
        "schema": "shared_quant_single_dispatch_ticket_v1",
        **reservation,
        "host_ledger_entry_sha256": reservation_sha,
    }
    files["release/ticket.json"] = encoded(ticket)
    mapping = {
        "source__" + n.replace("/", "__"): {"path": n, "bytes": len(raw), "sha256": sha(raw)}
        for n, raw in files.items()
    }
    require(len(mapping) == len(files), "FLAT_SOURCE_COLLISION")
    plan = {
        "schema": "shared_quant_final_cloud_transport_v1",
        "package_files": mapping,
        "release_sha256": expected_release_sha256,
        "ticket_sha256": sha(files["release/ticket.json"]),
        "runtime": runtime,
        "calendar": release["calendar"],
        "data": data,
        "auxiliary_inputs": aux,
        "service_budget_seconds": service_budget_seconds,
        "runner_canonical_sha256": sha(canonical),
        "datasets": datasets,
        "host_reservation_sha256": reservation_sha,
        "wider_parent": wider_binding,
        "root_approval_sha256": expected_approval_sha256,
    }
    plan_raw = encoded(plan)
    rendered = canonical.replace(
        b'ROOT_PLAN_SHA256 = "UNBOUND"', ('ROOT_PLAN_SHA256 = "' + sha(plan_raw) + '"').encode(), 1
    )
    rendered = rendered.replace(
        b'RUNNER_CANONICAL_SHA256 = "UNBOUND"',
        ('RUNNER_CANONICAL_SHA256 = "' + sha(canonical) + '"').encode(),
        1,
    )
    require(
        canonical_runner(rendered) == canonical and b"\r\n" not in rendered,
        "ACTUAL_RENDER_ROUNDTRIP_REQUIRED",
    )
    ast.parse(rendered)
    payload = {flat: files[r["path"]] for flat, r in mapping.items()}
    payload["shared-holdout-plan.json"] = plan_raw
    upload, job = output / "upload", output / "kernel"
    upload.mkdir(parents=True, exist_ok=False)
    job.mkdir()
    with zipfile.ZipFile(upload / "payload.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payload.items()):
            archive.writestr(name, raw)
    with zipfile.ZipFile(upload / "payload.zip") as archive:
        require(
            len(archive.namelist()) == len(payload)
            and set(archive.namelist()) == set(payload)
            and all(archive.read(n) == raw for n, raw in payload.items()),
            "ZIP_RAW_ROUNDTRIP_REQUIRED",
        )
    (upload / "dataset-metadata.json").write_bytes(
        encoded(
            {
                "id": source_dataset,
                "title": "Quant Shared Frozen Evaluation Source",
                "isPrivate": True,
                "licenses": [{"name": "other"}],
            }
        )
    )
    (job / "run.py").write_bytes(rendered)
    (job / "kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel,
                "title": "Quant Shared Frozen Evaluation",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": True,
                "dataset_sources": datasets,
                "kernel_sources": []
                if wider_binding is None
                else [WIDER_PARENT["canonical"] + "/3"],
                "competition_sources": [],
            }
        )
    )
    proof = {
        "release_sha256": expected_release_sha256,
        "plan_sha256": sha(plan_raw),
        "reservation_sha256": reservation_sha,
        "rendered_runner_sha256": sha(rendered),
        "payload": {n: {"bytes": len(raw), "sha256": sha(raw)} for n, raw in payload.items()},
        "dispatch_consumed": False,
        "provider_mutation_performed": False,
    }
    exclusive(output / "local-stage-proof.json", proof)
    exclusive(
        ledger / "bound-stage.json",
        {
            "release_sha256": expected_release_sha256,
            "plan_sha256": sha(plan_raw),
            "reservation_sha256": reservation_sha,
        },
    )
    return proof
