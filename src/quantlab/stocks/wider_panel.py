"""Separate wider panel: exact original cohorts, authenticated real histories.

This is prospective preparation code. It never reads outcome-value columns and
never changes the original six-feature panel or its historical eligibility.
"""

import json
import math
import platform
import struct
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from quantlab.data.archive import file_sha256
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.feature_reproduction import (
    POLICY,
    POLICY_SHA256,
    WHITELIST_PATH,
    reproduce_raw_six,
    require_full_acceptance_inventory,
)
from quantlab.stocks.features import FEATURE_IDS as SIX
from quantlab.stocks.prepare import (
    _ActualCalendarCache,
    _digest,
    _source_code,
    verify_prepared_panel,
)
from quantlab.stocks.wider_features import FEATURE_IDS, wider_candidate_from_history

SCHEMA = "quant_stock_wider_panel_v1"
PARENT_SHA = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"


def raw(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, default=str, allow_nan=False) + "\n"
    ).encode()


def code_pins():
    root = Path(__file__).resolve().parents[1]
    names = (
        "stocks/wider_panel.py",
        "stocks/wider_features.py",
        "stocks/prepare_wider.py",
        "stocks/feature_reproduction.py",
    )
    return {
        **{"quantlab/" + n: file_sha256(root / n) for n in names},
        "configs/stock-feature-portability-whitelist.json": file_sha256(WHITELIST_PATH),
    }


def runtime_pins():
    import pyarrow

    return {"python": platform.python_version(), "pyarrow": pyarrow.__version__}


def _same(a, b):
    return struct.pack("<d", a) == struct.pack("<d", b)


def rank_added(rows):
    """Rank observed additions within an unchanged complete original cohort.

    Input dictionaries must already contain original exact rank_* fields. Missing
    added raw values remain null; no eligibility change or raw imputation occurs.
    """
    if len({r["asset_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate fixed-cohort asset")
    for feature in FEATURE_IDS:
        if feature in SIX:
            continue
        available = []
        for row in rows:
            value = row["raw_" + feature]
            row["rank_" + feature] = None
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError("Nonfinite observed feature")
                available.append((value, row["asset_id"], row))
        available.sort(key=lambda x: (x[0], x[1]))
        n, left = len(available), 0
        while left < n:
            right = left + 1
            while right < n and available[right][0] == available[left][0]:
                right += 1
            score = (((left + 1 + right) / 2 - 0.5) / n - 0.5) if n > 1 else 0.0
            for _, _, row in available[left:right]:
                row["rank_" + feature] = score
            left = right
    return rows


def prepare_wider_panel(
    snapshot,
    prepared_root,
    schedule,
    output,
    *,
    expected_parent_sha256=PARENT_SHA,
    service_deadline_monotonic=None,
    reproduction_policy=None,
    reproduction_parent_platform=None,
):
    """Build all original pre-holdout feature dates, reading each needed asset once.

    Full parent authentication includes all7195 identities/32 empty histories.
    Only the union of already selected IDs is parsed here; this is computation on
    fixed historical cohorts, never a new availability/survivorship selection.
    No label values or native learned models are read by this preparation stage.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    def budget():
        if service_deadline_monotonic is not None and (
            not math.isfinite(service_deadline_monotonic)
            or time.monotonic() >= service_deadline_monotonic
        ):
            raise TimeoutError("Wider preparation service deadline")

    budget()
    if not isinstance(snapshot, MarketSnapshot) or not isinstance(schedule, XNYSSchedule):
        raise ValueError("Authenticated original snapshot and actual calendar required")
    if expected_parent_sha256 != PARENT_SHA:
        raise ValueError("Exact completed original full-cohort parent required")
    prepared_root, output = Path(prepared_root), Path(output)
    parent = verify_prepared_panel(prepared_root, expected_parent_sha256)
    if parent["config"]["source_code_sha256"] != _source_code():
        raise ValueError("Original preparation implementation changed")
    if (
        parent["config"]["calendar"] != schedule.provenance
        or parent["source_manifest_sha256"] != snapshot.manifest_sha256
        or parent["config"]["archive_sha256"] != snapshot.archive_sha256
    ):
        raise ValueError("Parent/source/calendar identity disagreement")
    snapshot._completion_guard(snapshot.manifest_sha256)
    if reproduction_policy is not None:
        if reproduction_policy != POLICY or reproduction_parent_platform != "Windows":
            raise ValueError("Explicit frozen Windows-to-Linux exact-whitelist policy required")
        if platform.system() != "Linux" or platform.python_version() != "3.12.15":
            raise ValueError(
                "Exact-whitelist portability policy requires actual locked Linux runtime"
            )
    sources, runtime = code_pins(), runtime_pins()
    if any(output.resolve().is_relative_to(p.resolve()) for p in (prepared_root, snapshot.root)):
        raise ValueError("Wider output must be outside immutable parent inputs")
    by_asset, by_date, original_schema = defaultdict(list), {}, None
    for item in parent["outputs"]:
        if item.get("kind") != "features":
            continue
        day = date.fromisoformat(Path(item["path"]).stem)
        if day >= date(2015, 1, 1):
            continue
        budget()
        table = pq.ParquetFile(prepared_root / item["path"]).read()
        if original_schema is not None and table.schema != original_schema:
            raise ValueError("Original feature schemas differ")
        original_schema = table.schema
        original = table.to_pylist()
        if len({r["asset_id"] for r in original}) != len(original):
            raise ValueError("Duplicate original feature key")
        by_date[day] = original
        for row in original:
            if row["signal_session"] != day:
                raise ValueError("Original feature date mismatch")
            by_asset[row["asset_id"]].append(row)
    if not by_date or any(not rows for rows in by_date.values()):
        raise ValueError("Full original past training cohorts required")
    pins = {
        "schema": SCHEMA,
        "parent_prepared_manifest_sha256": expected_parent_sha256,
        "original_archive_sha256": snapshot.archive_sha256,
        "source_normalized_manifest_sha256": snapshot.manifest_sha256,
        "source_snapshot_id": snapshot.snapshot_id,
        "calendar": schedule.provenance,
        "source_code_sha256": sources,
        "original_preparation_source_sha256": _source_code(),
        "runtime": runtime,
        "feature_reproduction_policy": dict(reproduction_policy) if reproduction_policy else None,
        "feature_reproduction_policy_sha256": POLICY_SHA256 if reproduction_policy else None,
        "feature_reproduction_parent_platform": reproduction_parent_platform,
        "registered_at_actual_utc": datetime.now(UTC),
        "scope": "all_original_selected_cohorts_before_2015_including_prior_training_dates",
        "upstream_population": parent["counts"],
        "selected_union_histories": len(by_asset),
        "no_label_value_reads": True,
        "fresh_model_score_reproduction_verified": False,
        "missing_encoding": "raw_nulls_preserved_optional_model_neutral_rank_plus_indicator",
    }
    budget()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(raw(pins))
    current = None
    gate_diagnostic = None
    accepted_differences = []
    try:
        schedule = _ActualCalendarCache(schedule)
        with snapshot.verified_read_session() as verified:
            for asset in sorted(by_asset):
                budget()
                current = asset
                gate_diagnostic = None
                bars = verified.bars("stock", asset.split(":", 1)[1])
                dates = tuple(b.session for b in bars)
                source_hash = snapshot._assets[asset]["source_sha256"]
                for row in by_asset[asset]:
                    budget()
                    day = row["signal_session"]
                    gate_diagnostic = None
                    right = bisect_right(dates, day)
                    enriched = wider_candidate_from_history(
                        asset,
                        bars[max(0, right - 253) : right],
                        schedule,
                        as_of=day,
                        archive_sha256=snapshot.archive_sha256,
                    )
                    candidate = enriched.original
                    computed = dict(candidate.features)
                    digest = _digest(candidate.source_rows, source_hash)
                    gate_diagnostic = {
                        "asset_id": asset,
                        "signal_session": day,
                        "eligible": candidate.eligible,
                        "reasons": candidate.reasons,
                        "source_digest": {
                            "computed": digest,
                            "parent": row["input_rows_sha256"],
                            "equal": digest == row["input_rows_sha256"],
                        },
                        "clocks": {
                            name: {
                                "computed": getattr(candidate, name),
                                "parent": row[name],
                                "equal": getattr(candidate, name) == row[name],
                            }
                            for name in (
                                "signal_at",
                                "intended_execution_at",
                                "max_input_available_at",
                            )
                        },
                        "features": {
                            f: {
                                "computed": computed.get(f),
                                "parent": row["raw_" + f],
                                "computed_float64_hex": struct.pack("<d", computed[f]).hex()
                                if f in computed
                                else None,
                                "parent_float64_hex": struct.pack("<d", row["raw_" + f]).hex(),
                                "exact_equal": f in computed
                                and _same(computed[f], row["raw_" + f]),
                            }
                            for f in SIX
                        },
                    }
                    if (
                        not candidate.eligible
                        or digest != row["input_rows_sha256"]
                        or any(not c["equal"] for c in gate_diagnostic["clocks"].values())
                    ):
                        raise ValueError(
                            "Original eligibility/source rows/availability "
                            "failed EXACT reproduction"
                        )
                    comparison = reproduce_raw_six(
                        tuple((f, row["raw_" + f]) for f in SIX),
                        candidate.features,
                        asset_id=asset,
                        signal_session=day,
                        parent_prepared_manifest_sha256=expected_parent_sha256,
                        source_context={
                            "source_sha256": source_hash,
                            "input_rows_sha256": digest,
                            "clocks": {
                                k: v["computed"] for k, v in gate_diagnostic["clocks"].items()
                            },
                        },
                        policy=reproduction_policy,
                        parent_platform=reproduction_parent_platform,
                        computed_platform=platform.system(),
                        parent_python=parent["config"]["runtime"]["python"],
                        computed_python=platform.python_version(),
                    )
                    accepted_differences.extend(asdict(v) for v in comparison.accepted_differences)
                    # Original raw/rank fields stay exactly as authenticated in the parent row.
                    for value in enriched.values:
                        if value.feature_id not in SIX:
                            row["raw_" + value.feature_id] = value.value
                        row["missing_" + value.feature_id] = value.value is None
                        row["reason_" + value.feature_id] = value.missing_reason
        if reproduction_policy is not None:
            require_full_acceptance_inventory(accepted_differences)
        # Mandatory session exit has reauthenticated metadata/source before output finalization.
        fields = list(original_schema)
        for feature in FEATURE_IDS:
            if feature not in SIX:
                fields.extend(
                    (
                        pa.field("raw_" + feature, pa.float64()),
                        pa.field("rank_" + feature, pa.float64()),
                    )
                )
            fields.extend(
                (
                    pa.field("missing_" + feature, pa.bool_()),
                    pa.field("reason_" + feature, pa.string()),
                )
            )
        schema = pa.schema(fields)
        outputs = []
        for day, rows in sorted(by_date.items()):
            budget()
            rows.sort(key=lambda r: r["asset_id"])
            rank_added(rows)
            path = output / "features" / (str(day) + ".parquet")
            path.parent.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
            outputs.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "rows": len(rows),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        budget()
        verify_prepared_panel(prepared_root, expected_parent_sha256)
        snapshot._completion_guard(snapshot.manifest_sha256)
        snapshot._check_source_contract()
        if (
            sources != code_pins()
            or runtime != runtime_pins()
            or pins["original_preparation_source_sha256"] != _source_code()
        ):
            raise ValueError("Wider source/runtime/original contract changed")
        manifest = {
            "schema": SCHEMA,
            "complete": True,
            "config": pins,
            "decision_count": len(by_date),
            "feature_rows": sum(map(len, by_date.values())),
            "outputs": outputs,
            "no_label_values_read": True,
            "fresh_reference_score_gate_pending": True,
            "accepted_feature_representation_differences": accepted_differences,
        }
        budget()
        (output / "manifest.json").write_bytes(raw(manifest))
        digest = file_sha256(output / "manifest.json")
        budget()
        (output / "COMPLETE").write_text(digest + "\n", encoding="ascii")
        return manifest
    except BaseException as error:
        (output / "FAILURE.json").write_bytes(
            raw(
                {
                    "complete": False,
                    "asset_id": current,
                    "original_gate_diagnostic": gate_diagnostic,
                    "accepted_feature_representation_differences": accepted_differences,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        )
        raise


def verify_wider_panel(root, expected_manifest_sha256):
    """External SHA-bound completed wider inputs; never authorize model score identity."""
    root = Path(root)
    if (root / "FAILURE.json").exists() or (root / "COMPLETE").read_bytes() != (
        expected_manifest_sha256 + "\n"
    ).encode():
        raise ValueError("Complete failure-free wider panel required")
    if file_sha256(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Wider manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        manifest["schema"] != SCHEMA
        or manifest["complete"] is not True
        or manifest["config"]["parent_prepared_manifest_sha256"] != PARENT_SHA
    ):
        raise ValueError("Exact wider panel parent contract required")
    if (root / "config.json").read_bytes() != raw(manifest["config"]):
        raise ValueError("Wider config mismatch")
    if (
        manifest["config"]["source_code_sha256"] != code_pins()
        or manifest["config"]["original_preparation_source_sha256"] != _source_code()
        or manifest["no_label_values_read"] is not True
    ):
        raise ValueError("Wider/original feature implementation or label-read policy differs")
    names, rows = set(), 0
    for item in manifest["outputs"]:
        path = root / item["path"]
        if (
            item["path"] in names
            or not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents))
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("Wider feature partition integrity failure")
        names.add(item["path"])
        rows += item["rows"]
    if len(names) != manifest["decision_count"] or rows != manifest["feature_rows"]:
        raise ValueError("Wider feature support mismatch")
    expected = names | {"manifest.json", "config.json", "COMPLETE"}
    if {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} != expected:
        raise ValueError("Unexpected/missing wider panel files")
    return manifest
