"""Source-only full selected-cohort arithmetic audit, never model fitting.

The child is readable Python source, not an encoded payload. Staging must bind
an actual root approval into both UNBOUND literals before any cloud execution.
All existing source/data bundles are reused unchanged; no new dataset is needed.
"""

import hashlib
import importlib.util
import json
import math
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT_APPROVAL_SHA256 = "UNBOUND"
ROOT_APPROVAL_JSON = "UNBOUND"
PARENT_PLAN_SHA256 = "d01e98b71f9ad9a2aff4f35252af429e14d2a6843e10d2a7f424ec3e6058e824"
SERVICE_SECONDS = 36000

CHILD = r"""
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date
from importlib.metadata import version
from pathlib import Path

PARENT_SHA = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
NORMALIZED_SHA = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
SIX = ("F01", "F07", "F09", "F13", "F15", "F19")
EXPECTED_ROWS = 71309
EXPECTED_DATES = 180


def ordered_float_bits(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)


def feature_comparison(parent, computed):
    finite = computed is not None and math.isfinite(computed) and math.isfinite(parent)
    parent_hex = struct.pack("<d", parent).hex()
    computed_hex = struct.pack("<d", computed).hex() if computed is not None else None
    return {
        "parent": parent if math.isfinite(parent) else None,
        "computed": computed if computed is not None and math.isfinite(computed) else None,
        "parent_float64_hex": parent_hex,
        "computed_float64_hex": computed_hex,
        "exact": parent_hex == computed_hex,
        "finite": finite,
        "ulp_distance": abs(ordered_float_bits(parent) - ordered_float_bits(computed))
        if finite
        else None,
    }


def main():
    context_raw = Path(os.environ["QUANTLAB_AUDIT_CONTEXT"]).read_bytes()
    assert hashlib.sha256(context_raw).hexdigest() == os.environ["QUANTLAB_AUDIT_CONTEXT_SHA"]
    context = json.loads(context_raw)
    deadline = float(os.environ["QUANTLAB_AUDIT_DEADLINE"])
    output = Path(os.environ["QUANTLAB_AUDIT_OUTPUT"])

    def budget():
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Full-cohort audit startup-inclusive deadline")

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                budget()
                digest.update(block)
        return digest.hexdigest()

    def guards():
        budget()
        assert sha(os.environ["QUANTLAB_AUDIT_CONTEXT"]) == os.environ["QUANTLAB_AUDIT_CONTEXT_SHA"]
        assert platform.python_version() == "3.12.15"
        assert Path(os.path.abspath(sys.executable)) == Path(context["locked_interpreter"])
        assert (
            Path(sys.executable)
            .resolve()
            .is_relative_to(Path(context["private_managed_root"]).resolve())
        )
        assert {name: version(name) for name in context["runtime"]} == context["runtime"]
        for name, expected in context["project_source_sha256"].items():
            assert sha(Path(context["project"]) / name) == expected
        for name, expected in context["input_container_sha256"].items():
            assert sha(name) == expected

    def write(name, value):
        budget()
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
            stream.write("\n")

    guards()
    # Authentication and locked-runtime checks finish BEFORE project/native imports.
    sys.path.insert(0, str(Path(context["project"]) / "src"))
    import pyarrow.parquet as pq
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.data.snapshot import MarketSnapshot
    from quantlab.stocks.features import candidate_from_history
    from quantlab.stocks.prepare import _ActualCalendarCache, _digest, verify_prepared_panel

    prepared_root = Path(context["prepared_root"])
    parent = verify_prepared_panel(prepared_root, PARENT_SHA)
    snapshot = MarketSnapshot(context["market_root"], NORMALIZED_SHA)
    schedule = _ActualCalendarCache(XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31)))
    assert parent["config"]["calendar"] == schedule.provenance
    assert parent["config"]["archive_sha256"] == snapshot.archive_sha256
    assert parent["source_manifest_sha256"] == snapshot.manifest_sha256
    by_asset, dates_seen = defaultdict(list), set()
    parent_input_hash = hashlib.sha256()
    count = 0
    for item in parent["outputs"]:
        budget()
        if item.get("kind") != "features":
            continue
        day = date.fromisoformat(Path(item["path"]).stem)
        if day >= date(2015, 1, 1):
            continue  # Gate on partition date BEFORE any feature values are parsed.
        table = pq.ParquetFile(prepared_root / item["path"]).read()
        rows = table.to_pylist()
        assert len({r["asset_id"] for r in rows}) == len(rows)
        dates_seen.add(day)
        for row in rows:
            assert row["signal_session"] == day
            by_asset[row["asset_id"]].append(row)
            material = {
                "asset_id": row["asset_id"],
                "signal_session": str(day),
                "input_rows_sha256": row["input_rows_sha256"],
                "signal_at": str(row["signal_at"]),
                "intended_execution_at": str(row["intended_execution_at"]),
                "max_input_available_at": str(row["max_input_available_at"]),
                "raw_six_hex": {f: struct.pack("<d", row["raw_" + f]).hex() for f in SIX},
            }
            parent_input_hash.update(
                json.dumps(material, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
            count += 1
    assert count == EXPECTED_ROWS and len(dates_seen) == EXPECTED_DATES
    assert min(dates_seen) == date(2000, 1, 31) and max(dates_seen) == date(2014, 12, 31)
    stats = {
        f: {
            "compared": 0,
            "exact": 0,
            "mismatched": 0,
            "nonfinite_or_missing": 0,
            "max_ulp_distance": 0,
            "ulp_distance_counts": Counter(),
            "representatives": [],
        }
        for f in SIX
    }
    counts = Counter()
    completed_histories = []
    calendar_set = set(schedule.sessions)
    mismatch_path = output / "mismatches.jsonl"
    with mismatch_path.open("x") as log:
        with snapshot.verified_read_session() as verified:
            for asset in sorted(by_asset):
                budget()
                bars = verified.bars("stock", asset.split(":", 1)[1])
                source_hash = snapshot._assets[asset]["source_sha256"]
                source_dates = tuple(b.session for b in bars)
                for row in sorted(by_asset[asset], key=lambda r: r["signal_session"]):
                    budget()
                    day = row["signal_session"]
                    index = schedule.sessions.index(day)
                    required = schedule.sessions[index - 252 : index + 1]
                    assert len(required) == 253
                    left, right = (
                        bisect_left(source_dates, required[0]),
                        bisect_right(source_dates, day),
                    )
                    window = bars[left:right]
                    candidate, row_error = None, None
                    try:
                        # EXACT original preparer's calendar-bounded source window.
                        candidate = candidate_from_history(
                            asset,
                            window,
                            schedule,
                            as_of=day,
                            archive_sha256=snapshot.archive_sha256,
                        )
                    except Exception as error:
                        # Audit arithmetic/source-support findings; do not stop at first difference.
                        row_error = {"type": type(error).__name__, "message": str(error)[:1000]}
                    actual = dict(candidate.features) if candidate is not None else {}
                    clocks = {
                        name: {
                            "computed": str(getattr(candidate, name, None)),
                            "parent": str(row[name]),
                            "equal": candidate is not None
                            and getattr(candidate, name) == row[name],
                        }
                        for name in ("signal_at", "intended_execution_at", "max_input_available_at")
                    }
                    digest = (
                        _digest(candidate.source_rows, source_hash)
                        if candidate is not None
                        else None
                    )
                    source_equal = digest == row["input_rows_sha256"]
                    eligible = candidate is not None and candidate.eligible
                    physics_ok = (
                        eligible and source_equal and all(v["equal"] for v in clocks.values())
                    )
                    comparisons = {
                        f: feature_comparison(row["raw_" + f], actual.get(f)) for f in SIX
                    }
                    failed = [f for f, v in comparisons.items() if not v["exact"]]
                    counts["rows_audited"] += 1
                    counts["rows_with_feature_drift"] += bool(failed)
                    counts["source_clock_or_eligibility_failures"] += not physics_ok
                    counts["candidate_exceptions"] += row_error is not None
                    for feature, result in comparisons.items():
                        stat = stats[feature]
                        stat["compared"] += 1
                        stat["exact"] += result["exact"]
                        if not result["exact"]:
                            stat["mismatched"] += 1
                            if result["ulp_distance"] is None:
                                stat["nonfinite_or_missing"] += 1
                            else:
                                stat["max_ulp_distance"] = max(
                                    stat["max_ulp_distance"], result["ulp_distance"]
                                )
                                stat["ulp_distance_counts"][str(result["ulp_distance"])] += 1
                            if len(stat["representatives"]) < 8:
                                stat["representatives"].append(
                                    {"asset_id": asset, "signal_session": str(day), **result}
                                )
                    record = {
                        "asset_id": asset,
                        "signal_session": str(day),
                        "eligible": eligible,
                        "reasons": candidate.reasons if candidate is not None else None,
                        "source_sha256": source_hash,
                        "source_digest": {
                            "computed": digest,
                            "parent": row["input_rows_sha256"],
                            "equal": source_equal,
                        },
                        "clocks": clocks,
                        "features": comparisons,
                        "mismatched_features": failed,
                        "candidate_exception": row_error,
                    }
                    # Audit the prospective last-253-source shortcut independently too.
                    shortcut = bars[max(0, right - 253) : right]
                    same_source_rows = tuple(b.row_sha256 for b in shortcut) == tuple(
                        b.row_sha256 for b in window
                    )
                    if not same_source_rows:
                        counts["last253_source_window_differs_from_calendar_window"] += 1
                        try:
                            short = candidate_from_history(
                                asset,
                                shortcut,
                                schedule,
                                as_of=day,
                                archive_sha256=snapshot.archive_sha256,
                            )
                            short_digest = _digest(short.source_rows, source_hash)
                            record["last253_source_window_audit"] = {
                                "eligible": short.eligible,
                                "reasons": short.reasons,
                                "source_digest_equal": short_digest == row["input_rows_sha256"],
                                "first_source_session": str(shortcut[0].session)
                                if shortcut
                                else None,
                            }
                            if not short.eligible or short_digest != row["input_rows_sha256"]:
                                counts["last253_source_shortcut_support_failures"] += 1
                        except Exception as error:
                            record["last253_source_window_audit"] = {
                                "error_type": type(error).__name__
                            }
                            counts["last253_source_shortcut_support_failures"] += 1
                    if not physics_ok or not same_source_rows:
                        observed = {b.session for b in window}
                        record["source_window"] = {
                            "required_first": str(required[0]),
                            "required_last": str(required[-1]),
                            "source_first": str(window[0].session) if window else None,
                            "source_last": str(window[-1].session) if window else None,
                            "source_observations": len(window),
                            "missing_required_dates": [
                                str(d) for d in required if d not in observed
                            ],
                            "off_calendar_dates": [
                                str(b.session) for b in window if b.session not in calendar_set
                            ],
                            "flagged_source_rows": [
                                {
                                    "session": str(b.session),
                                    "flags": b.quality_flags,
                                    "row_sha256": b.row_sha256,
                                }
                                for b in window
                                if b.quality_flags
                            ],
                        }
                    if failed or not physics_ok or not same_source_rows:
                        log.write(
                            json.dumps(record, sort_keys=True, allow_nan=False, default=str) + "\n"
                        )
                        counts["mismatch_log_records"] += 1
                log.flush()
                completed_histories.append(
                    {
                        "asset_id": asset,
                        "source_sha256": source_hash,
                        "selected_rows": len(by_asset[asset]),
                    }
                )
                if len(completed_histories) % 32 == 0:
                    print(
                        json.dumps(
                            {
                                "histories_audited": len(completed_histories),
                                "histories_required": len(by_asset),
                                **counts,
                            }
                        ),
                        flush=True,
                    )
        # Mandatory verified source session postflight finishes before summary/marker.
    assert counts["rows_audited"] == EXPECTED_ROWS
    assert len(completed_histories) == len(by_asset)
    assert all(v["compared"] == EXPECTED_ROWS for v in stats.values())
    verify_prepared_panel(prepared_root, PARENT_SHA)
    guards()
    summary = {
        "schema": "full_selected_feature_portability_audit_v1",
        "diagnostic_complete": True,
        "scientific_study_complete": False,
        "fits": 0,
        "labels_read": False,
        "enrichment_run": False,
        "all_original_pre2015_selected_rows_audited": True,
        "expected_rows": EXPECTED_ROWS,
        "expected_dates": EXPECTED_DATES,
        "selected_histories": len(by_asset),
        "first_date": str(min(dates_seen)),
        "last_date": str(max(dates_seen)),
        "parent_prepared_manifest_sha256": PARENT_SHA,
        "normalized_manifest_sha256": NORMALIZED_SHA,
        "original_archive_sha256": snapshot.archive_sha256,
        "calendar": schedule.provenance,
        "parent_selected_input_sha256": parent_input_hash.hexdigest(),
        "source_plan_sha256": context["source_plan_sha256"],
        "runtime": context["runtime"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_clock_eligibility_all_exact": counts["source_clock_or_eligibility_failures"] == 0,
        "all_six_feature_values_bit_exact": counts["rows_with_feature_drift"] == 0,
        "counts": dict(counts),
        "per_feature": stats,
        "completed_source_histories": completed_histories,
        "mismatch_log": {
            "path": mismatch_path.name,
            "sha256": sha(mismatch_path),
            "bytes": mismatch_path.stat().st_size,
        },
        "policy_extension_or_model_selection": False,
    }
    write("summary.json", summary)
    guards()
    summary_sha = sha(output / "summary.json")
    budget()
    with (output / "AUDIT_COMPLETE").open("x") as stream:
        stream.write(summary_sha + "\n")
    print(
        json.dumps({"diagnostic_complete": True, "summary_sha256": summary_sha, **counts}),
        flush=True,
    )


if __name__ == "__main__":
    main()
"""


def file_sha(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError("Full audit deadline")
            digest.update(block)
    return digest.hexdigest()


def discover(root, expected, deadline, *, name, size=None):
    root, matches = Path(root), []
    for path in root.rglob(name):
        if time.monotonic() >= deadline:
            raise TimeoutError("Full audit discovery deadline")
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            continue
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Regular contained source required")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source escapes input root")
        if file_sha(path, deadline) == expected:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("Exactly one SHA-bound original source required")
    return matches[0]


def run(input_root="/kaggle/input", output_root="/kaggle/working/full-portability-audit"):
    if ROOT_APPROVAL_JSON == "UNBOUND" or ROOT_APPROVAL_SHA256 == "UNBOUND":
        raise ValueError("Unbound source-only full audit")
    raw = ROOT_APPROVAL_JSON.encode()
    if hashlib.sha256(raw).hexdigest() != ROOT_APPROVAL_SHA256:
        raise ValueError("Root audit approval SHA mismatch")
    approval = json.loads(raw)
    if (
        approval["schema"] != "quant_full_feature_portability_audit_source_freeze_v1"
        or approval["status"] != "READY"
        or approval["parent_plan_sha256"] != PARENT_PLAN_SHA256
        or approval["child_sha256"] != hashlib.sha256(CHILD.encode()).hexdigest()
        or approval["service_seconds"] != SERVICE_SECONDS
        or approval["expected_selected_rows"] != 71309
        or approval["expected_dates"] != 180
        or approval["fits"] != 0
        or approval["labels_read"] is not False
    ):
        raise ValueError("Exact full audit approval contract required")
    if not Path("/kaggle/input").is_dir():
        raise ValueError("Cloud-only full selected-cohort audit")
    output = Path(output_root)
    deadline = time.monotonic() + SERVICE_SECONDS
    output.mkdir(parents=True, exist_ok=False)
    (output / "STARTED.json").write_text(
        json.dumps(
            {
                "at": datetime.now(UTC).isoformat(),
                "root_approval_sha256": ROOT_APPROVAL_SHA256,
                "parent_plan_sha256": PARENT_PLAN_SHA256,
                "fits": 0,
            }
        )
        + "\n"
    )

    def expired(_signum, _frame):
        raise TimeoutError("Startup-inclusive full audit deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, SERVICE_SECONDS)
    try:
        plan_path = discover(input_root, PARENT_PLAN_SHA256, deadline, name="*.json")
        plan = json.loads(plan_path.read_bytes())
        entry = next(
            i
            for i in plan["source_files"]
            if i["path"] == "cloud/stock_wider_notebook_bootstrap.py"
        )
        bootstrap_path = discover(
            input_root, entry["sha256"], deadline, name=entry["package_name"], size=entry["bytes"]
        )
        spec = importlib.util.spec_from_file_location("frozen_full_audit_bootstrap", bootstrap_path)
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        bootstrap.ROOT_PLAN_SHA256 = PARENT_PLAN_SHA256
        locked, context, env = bootstrap.run(input_root, output / "bootstrap")
        env.update(
            QUANTLAB_AUDIT_CONTEXT=str(context),
            QUANTLAB_AUDIT_CONTEXT_SHA=file_sha(context, deadline),
            QUANTLAB_AUDIT_OUTPUT=str(output),
            QUANTLAB_AUDIT_DEADLINE=str(deadline),
        )
        remaining = deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError("No full audit child service budget")
        subprocess.run([locked, "-c", CHILD], env=env, check=True, timeout=remaining)
        if (output / "AUDIT_COMPLETE").read_text().strip() != file_sha(
            output / "summary.json", deadline
        ):
            raise ValueError("Full audit completion marker mismatch")
        if time.monotonic() >= deadline:
            raise TimeoutError("Full audit final deadline")
    except BaseException as error:
        (output / "FAILURE.json").write_text(
            json.dumps(
                {"diagnostic_complete": False, "fits": 0, "error_type": type(error).__name__}
            )
            + "\n"
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    run()
