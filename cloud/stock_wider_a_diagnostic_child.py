# ruff: noqa: E402
import hashlib
import json
import os
import platform
import struct
import sys
import time
from bisect import bisect_right
from datetime import date
from importlib.metadata import version
from pathlib import Path

context_raw = Path(os.environ["QUANTLAB_A_CONTEXT"]).read_bytes()
assert hashlib.sha256(context_raw).hexdigest() == os.environ["QUANTLAB_A_CONTEXT_SHA"]
context = json.loads(context_raw)
deadline = float(os.environ["QUANTLAB_A_DEADLINE"])
sys.path.insert(0, str(Path(context["project"]) / "src"))


def guards():
    assert time.monotonic() < deadline
    assert (
        hashlib.sha256(Path(os.environ["QUANTLAB_A_CONTEXT"]).read_bytes()).hexdigest()
        == os.environ["QUANTLAB_A_CONTEXT_SHA"]
    )
    assert platform.python_version() == "3.12.15"
    assert Path(os.path.abspath(sys.executable)) == Path(context["locked_interpreter"])
    assert (
        Path(sys.executable)
        .resolve()
        .is_relative_to(Path(context["private_managed_root"]).resolve())
    )
    assert {n: version(n) for n in context["runtime"]} == context["runtime"]
    for name, expected in context["project_source_sha256"].items():
        assert (
            hashlib.sha256((Path(context["project"]) / name).read_bytes()).hexdigest() == expected
        )
    for name, expected in context["input_container_sha256"].items():
        h = hashlib.sha256()
        with Path(name).open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                assert time.monotonic() < deadline
                h.update(block)
        assert h.hexdigest() == expected


guards()
import pyarrow.parquet as pq

from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.features import FEATURE_IDS
from quantlab.stocks.prepare import _ActualCalendarCache, _digest, verify_prepared_panel
from quantlab.stocks.wider_features import wider_candidate_from_history

root = Path(context["project"])
parent = Path(context["prepared_root"])
m = verify_prepared_panel(
    parent, "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
)
selected_rows = []
for item in m["outputs"]:
    if item.get("kind") != "features" or Path(item["path"]).stem >= "2015-01-01":
        continue
    rows = pq.ParquetFile(parent / item["path"]).read().to_pylist()
    selected_rows.extend(r for r in rows if r["asset_id"] == "stock:a.us")
assert len(selected_rows) == 150
assert min(r["signal_session"] for r in selected_rows) == date(2001, 6, 29)
assert max(r["signal_session"] for r in selected_rows) == date(2014, 12, 31)
snapshot = MarketSnapshot(
    context["market_root"], "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
)
bars = snapshot.bars("stock", "a.us")
dates = tuple(b.session for b in bars)
schedule = _ActualCalendarCache(XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31)))
assert m["config"]["calendar"] == schedule.provenance
assert m["config"]["archive_sha256"] == snapshot.archive_sha256
assert m["source_manifest_sha256"] == snapshot.manifest_sha256
results = []
for selected in sorted(selected_rows, key=lambda x: x["signal_session"]):
    assert time.monotonic() < deadline
    day = selected["signal_session"]
    right = bisect_right(dates, day)
    window = bars[max(0, right - 253) : right]
    c = wider_candidate_from_history(
        "stock:a.us", window, schedule, as_of=day, archive_sha256=snapshot.archive_sha256
    ).original
    pairs = {
        k: {
            "actual": str(getattr(c, k)),
            "parent": str(selected[k]),
            "equal": getattr(c, k) == selected[k],
        }
        for k in ["signal_at", "intended_execution_at", "max_input_available_at"]
    }
    a = _digest(c.source_rows, snapshot._assets["stock:a.us"]["source_sha256"])
    pairs["digest"] = {
        "actual": a,
        "parent": selected["input_rows_sha256"],
        "equal": a == selected["input_rows_sha256"],
    }
    for f in FEATURE_IDS:
        v = dict(c.features).get(f)
        old = selected["raw_" + f]
        pairs[f] = {
            "actual": v,
            "parent": old,
            "parent_float64_hex": struct.pack("<d", old).hex(),
            "actual_float64_hex": struct.pack("<d", v).hex() if v is not None else None,
            "equal": v is not None and struct.pack(">d", v) == struct.pack(">d", old),
        }
    bad = [k for k, v in pairs.items() if not v["equal"]]
    item = {
        "date": str(day),
        "eligible": c.eligible,
        "reasons": c.reasons,
        "mismatched_fields": bad,
        "checks": pairs,
    }
    if bad or not c.eligible:
        index = schedule.sessions.index(day)
        expected = schedule.sessions[index - 252 : index + 1]
        observed = {b.session for b in window}
        item.update(
            checks=pairs,
            source_window=[
                {
                    "date": str(b.session),
                    "row_sha256": b.row_sha256,
                    "source_sha256": b.source_sha256,
                    "close": b.close,
                    "volume": b.volume,
                    "flags": b.quality_flags,
                }
                for b in window
            ],
            source_slice_first=str(window[0].session),
            required_first=str(expected[0]),
            off_calendar_source_dates=[
                str(b.session) for b in window if b.session not in set(schedule.sessions)
            ],
            missing_required_dates=[str(d) for d in expected if d not in observed],
        )
        from bisect import bisect_left

        exact_window = bars[bisect_left(dates, expected[0]) : right]
        fixed = wider_candidate_from_history(
            "stock:a.us", exact_window, schedule, as_of=day, archive_sha256=snapshot.archive_sha256
        ).original
        item["calendar_bound_window"] = {
            "source_observations": len(exact_window),
            "eligible": fixed.eligible,
            "digest_equal": _digest(
                fixed.source_rows, snapshot._assets["stock:a.us"]["source_sha256"]
            )
            == selected["input_rows_sha256"],
            "all_raw_six_bits_equal": all(
                struct.pack(">d", v) == struct.pack(">d", selected["raw_" + f])
                for f, v in fixed.features
            ),
            "features_count": len(fixed.features),
        }
    results.append(item)
    print(
        json.dumps({"date": str(day), "eligible": c.eligible, "mismatched_fields": bad}), flush=True
    )
out = {
    "asset": "stock:a.us",
    "selected_dates_checked": len(results),
    "failures": [r for r in results if r["mismatched_fields"] or not r["eligible"]],
    "all_date_results": results,
    "first_failure": next(
        (r for r in results if r["mismatched_fields"] or not r["eligible"]), None
    ),
    "runtime": {n: version(n) for n in context["runtime"]},
    "python": platform.python_version(),
    "parent_prepared_manifest_sha256": (
        "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
    ),
    "normalized_manifest_sha256": snapshot.manifest_sha256,
    "calendar": schedule.provenance,
    "fits": 0,
    "labels_read": False,
    "diagnostic_complete": True,
    "scientific_study_complete": False,
    "scope": (
        "All original selected A-only pre2015 source feature rows; "
        "no labels, models or full preparation"
    ),
}
snapshot._completion_guard(snapshot.manifest_sha256)
guards()
p = Path(os.environ["QUANTLAB_A_OUTPUT"]) / "report.json"
p.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps({"selected_dates_checked": len(results), "failures": out["failures"]}), flush=True)

guards()
assert time.monotonic() < deadline
(Path(os.environ["QUANTLAB_A_OUTPUT"]) / "DIAGNOSTIC_COMPLETE").write_text(
    hashlib.sha256(p.read_bytes()).hexdigest() + "\n"
)
