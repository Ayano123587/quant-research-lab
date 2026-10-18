"""Real-source common-probe risk forecasts, separate from portfolio accounting.

The constant-weight daily probe is a statistical target, not a holdings ledger.
Source partitions can be authenticated in full; future proxy arithmetic is never
performed until its exact scheduled maturity is inside the development cutoff.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import platform
import shutil
import time
import tomllib
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np

from quantlab.data import archive, calendar, ingest, returns
from quantlab.data import provenance as provenance_module
from quantlab.data import snapshot as snapshot_module
from quantlab.data import yahoo as yahoo_module
from quantlab.data.archive import Bar, file_sha256
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity, identity_from_source, resolve_source_identity
from quantlab.data.snapshot import MarketSnapshot
from quantlab.risk import covariance
from quantlab.risk.covariance import ReturnWindow, estimate_covariance


def _raw(value):
    return (
        json.dumps(
            value, sort_keys=True, indent=2, allow_nan=False, default=lambda x: x.isoformat()
        )
        + "\n"
    ).encode()


def _digest(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending-" + uuid4().hex)
    with pending.open("xb") as handle:
        handle.write(_raw(value))
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    pending.rename(path)


def _source_closure():
    return {
        name: file_sha256(Path(module.__file__))
        for name, module in (
            ("study", __import__(__name__, fromlist=["__file__"])),
            ("covariance", covariance),
            ("archive", archive),
            ("calendar", calendar),
            ("ingest", ingest),
            ("returns", returns),
            ("snapshot", snapshot_module),
            ("provenance", provenance_module),
            ("yahoo", yahoo_module),
        )
    }


def load_spec(path):
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    p, e, f = spec["panels"], spec["estimators"], spec["forecast_scoring"]
    if (
        spec["selection_information_cutoff_session"] != "2014-12-31"
        or spec["signal_start"] != "2007-02-28"
        or p["etf_symbols"] != ["SPY", "EFA", "EEM", "TLT", "LQD", "GLD", "DBC", "VNQ"]
        or p["stock_asset_counts"] != [50, 100, 150]
        or p["return_observations"] != [63, 126, 252]
        or p["stock_order"] != "descending_past_63_session_source_activity_then_ascending_asset_id"
        or e["methods"]
        != ["sample_ddof1", "ewma", "ledoit_wolf", "oas", "constant_correlation", "pca_factor"]
        or e["pca_components"] != [1, 3, 5]
        or e["ewma_half_life_observations"] != 63
        or e["ewma_finite_sample_correction"] != "effective_weight_ddof1"
        or e["constant_correlation_shrinkage"] != 0.5
        or e["pca_basis"] != "daily_return_covariance_no_standardisation"
        or e["pca_residual_floor"] != 0
        or e["matrix_ridge"] != 0
        or f["reference_weights"]
        != "equal_weight_selected_past_only_panel_common_to_all_estimators"
        or f["horizon"]
        != "21_scheduled_close_to_close_intervals_starting_execution_close_22_actual_marks"
        or f["realised_proxy"] != "sample_ddof1_daily_reference_portfolio_return_variance"
        or f["primary_selection"] != "mean_QLIKE_on_common_mature_dates_with_positive_forecasts"
        or f["losses"]
        != ["squared_variance_error", "log_forecast_variance_plus_realised_over_forecast"]
        or f["tie_rule"] != "ascending_fixed_candidate_id"
        or f["selection_scope"]
        != "separate_per_panel_and_observation_window_no_pooling_unlike_variance_scales"
        or f["probe_interpretation"]
        != "constant_weight_daily_return_risk_probe_not_a_tradable_holdings_ledger"
        or spec["integration_dimensions"]
        != {
            "etf_asset_count": 8,
            "etf_return_observations": 63,
            "stock_asset_count": 100,
            "stock_return_observations": 252,
            "dimension_selection": (
                "declared_before_scores_not_chosen_from_best_development_or_holdout_performance"
            ),
        }
    ):
        raise ValueError("Registered development risk protocol required before scores")
    return spec


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    method: str
    parameters: tuple[tuple[str, float | int], ...] = ()


def recipes(spec):
    e = spec["estimators"]
    return (
        Recipe("sample_ddof1", "sample_ddof1"),
        Recipe(
            "ewma_halflife63",
            "ewma",
            (("ewma_lambda", 2 ** (-1 / e["ewma_half_life_observations"])),),
        ),
        Recipe("ledoit_wolf", "ledoit_wolf"),
        Recipe("oas", "oas"),
        Recipe(
            "constant_correlation_fixed0p5",
            "constant_correlation",
            (("constant_correlation_intensity", e["constant_correlation_shrinkage"]),),
        ),
        *(
            Recipe(f"pca_K{k}", "pca_factor", (("n_factors", k), ("residual_floor", 0.0)))
            for k in e["pca_components"]
        ),
    )


def panel_grid(spec):
    return tuple(
        (name, size, observations)
        for name, size in (
            ("etf8", 8),
            *((f"stock{p}", p) for p in spec["panels"]["stock_asset_counts"]),
        )
        for observations in spec["panels"]["return_observations"]
    )


@dataclass(frozen=True)
class AssetWindow:
    asset_id: str
    signal_session: date
    activity: float
    returns252: tuple[float, ...]
    source_rows: tuple[tuple[str, int, str], ...]
    input_sha256: str
    future_rows: tuple[Bar, ...]


def _identity(bar):
    return bar.source_member, bar.row_number, bar.row_sha256


def proxy_schedule(schedule, signal_session):
    """Calendar clocks only: no mark or outcome values are accessed here."""
    index = bisect_left(schedule.sessions, signal_session)
    if index == len(schedule.sessions) or schedule.sessions[index] != signal_session:
        raise ValueError("Actual scheduled signal session required")
    days = schedule.sessions[index + 1 : index + 23]
    if len(days) != 22:
        raise ValueError("Calendar must cover full declared proxy horizon")
    return tuple(days), schedule.session(days[-1]).decision_at


def asset_window(asset_id, past_rows, schedule, *, signal_session, future_rows=()):
    """Low-level trusted actual rows: caller slices precisely, never fills gaps."""
    signal = schedule.session(signal_session)
    index = bisect_left(schedule.sessions, signal_session)
    if index < 252:
        return None, ("INSUFFICIENT_CALENDAR_HISTORY",)
    expected = schedule.sessions[index - 252 : index + 1]
    rows = tuple(past_rows)
    if not rows:
        return None, ("EMPTY_PAST_HISTORY",)
    if any(b.asset_id != asset_id or b.session > signal_session for b in rows):
        raise ValueError("Past-only source identity/cutoff violated")
    if tuple(b.session for b in rows) != expected:
        return None, ("MISSING_EXACT_253_SCHEDULED_MARKS",)
    if any(b.quality_flags for b in rows):
        return None, ("FLAGGED_PAST_SOURCE_MARK",)
    if any(not math.isfinite(b.close) or b.close <= 0 for b in rows):
        return None, ("INVALID_PAST_CLOSE",)
    if any(not math.isfinite(b.volume) or b.volume < 0 for b in rows[-63:]):
        return None, ("INVALID_OBSERVED_ACTIVITY",)
    activity_values = [b.close * b.volume for b in rows[-63:]]
    if not all(math.isfinite(v) for v in activity_values):
        return None, ("NONFINITE_SOURCE_ACTIVITY",)
    activity = float(np.median(activity_values))
    values = tuple(b.close / a.close - 1 for a, b in zip(rows[:-1], rows[1:], strict=True))
    identities = tuple(_identity(b) for b in rows)
    payload = {
        "asset_id": asset_id,
        "signal_session": signal_session,
        "max_available_at": signal.decision_at,
        "source_rows": identities,
        "returns252": values,
        "activity63": activity,
    }
    return AssetWindow(
        asset_id, signal_session, activity, values, identities, _digest(payload), tuple(future_rows)
    ), ()


def select_nested(candidates, sizes=(50, 100, 150)):
    """Single past-only ordering; no future support field is inspected."""
    candidates = tuple(candidates)
    if len({c.asset_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate historical source identity")
    ordered = tuple(sorted(candidates, key=lambda c: (-c.activity, c.asset_id)))
    return {size: ordered[:size] if len(ordered) >= size else () for size in sizes}


def return_window(
    panel,
    schedule,
    count,
    *,
    signal_session,
    source_snapshot_id=None,
    archive_sha256=None,
    source_identity=None,
):
    panel = tuple(panel)
    if not panel or count not in (63, 126, 252):
        raise ValueError("Declared nonempty panel and observation count required")
    if any(c.signal_session != signal_session for c in panel):
        raise ValueError("Mixed signal dates")
    if source_identity is None:
        source_identity = SourceIdentity(
            "original_archive",
            schedule.provenance["schedule_sha256"],
            "source_OHLC_adjustment_unverified",
            manifest_sha256=source_snapshot_id,
            archive_sha256=archive_sha256,
        )
    source_identity = resolve_source_identity(
        schedule.provenance["schedule_sha256"],
        source_identity=source_identity,
        archive_sha256=archive_sha256,
    )
    if source_snapshot_id is not None and source_snapshot_id != source_identity.source_id:
        raise ValueError("Contradictory risk-window source snapshot identity")
    end = bisect_left(schedule.sessions, signal_session)
    days = schedule.sessions[end - count : end + 1]
    timing = tuple(schedule.session(day) for day in days)
    return ReturnWindow(
        np.asarray([c.returns252[-count:] for c in panel]).T,
        tuple(c.asset_id for c in panel),
        tuple(t.close_at for t in timing[:-1]),
        tuple(t.close_at for t in timing[1:]),
        tuple(t.decision_at for t in timing[1:]),
        schedule.session(signal_session).decision_at,
        source_identity.source_id,
        "close_simple_daily_return_" + source_identity.price_basis,
        "exact253_past_closes_activity63_nested_source_population_v1",
        _digest({"input_hashes": [c.input_sha256 for c in panel], "count": count}),
        {
            **source_identity.as_provenance(),
            "past_asset_inputs_sha256": _digest([c.input_sha256 for c in panel]),
        },
    )


def future_proxy(panel, schedule, *, signal_session, information_cutoff):
    """Clock gate precedes *all* future mark arithmetic, including coverage checks."""
    if information_cutoff > schedule.session(date(2014, 12, 31)).decision_at:
        raise ValueError("Development refuses 2015+ outcome/proxy construction")
    days, maturity = proxy_schedule(schedule, signal_session)
    base = {
        "signal_session": signal_session,
        "planned_available_at": maturity,
        "start_session": days[0],
        "end_session": days[-1],
        "observations": 21,
        "probe": "fixed_equal_weight_daily_return_probe_not_holdings_ledger",
        "units": covariance.UNITS,
    }
    if maturity > information_cutoff:
        return {
            **base,
            "status": "not_mature_by_information_cutoff",
            "value": None,
            "missing": [],
            "source_rows_sha256": None,
        }
    if not panel:
        return {
            **base,
            "status": "panel_unavailable",
            "value": None,
            "missing": [],
            "source_rows_sha256": None,
        }
    columns, missing, inputs = [], [], []
    for candidate in panel:
        rows = candidate.future_rows
        # No substitute, gap collapse, shortened endpoint or dropped asset.
        if tuple(b.session for b in rows) != days:
            missing.append((candidate.asset_id, "missing_exact_future_marks"))
            continue
        if any(b.asset_id != candidate.asset_id for b in rows):
            raise ValueError("Future source identity mismatch")
        inputs.append((candidate.asset_id, tuple(_identity(b) for b in rows)))
        if any(b.quality_flags for b in rows):
            missing.append((candidate.asset_id, "flagged_future_mark"))
            continue
        if any(not math.isfinite(b.close) or b.close <= 0 for b in rows):
            missing.append((candidate.asset_id, "invalid_future_close"))
            continue
        columns.append([b.close / a.close - 1 for a, b in zip(rows[:-1], rows[1:], strict=True)])
    if missing:
        return {
            **base,
            "status": "unresolved_future_panel",
            "value": None,
            "missing": missing,
            "source_rows_sha256": _digest(inputs),
        }
    observed = np.asarray(columns).T @ np.full(len(panel), 1 / len(panel))
    value = float(np.var(observed, ddof=1))
    return {
        **base,
        "status": "valid",
        "value": value,
        "missing": [],
        "source_rows_sha256": _digest(inputs),
        "asset_ids": [c.asset_id for c in panel],
        "observed_probe_returns_sha256": _digest(observed.tolist()),
    }


def score_forecast(forecast, proxy):
    if proxy["status"] != "valid":
        return {"status": proxy["status"], "squared_variance_error": None, "qlike": None}
    if forecast is None or not math.isfinite(forecast) or forecast <= 0:
        return {
            "status": "nonpositive_or_invalid_forecast",
            "squared_variance_error": None,
            "qlike": None,
        }
    realised = proxy["value"]
    if realised is None or not math.isfinite(realised) or realised < 0:
        raise ValueError("Finite nonnegative observed variance proxy required")
    return {
        "status": "valid",
        "squared_variance_error": (forecast - realised) ** 2,
        "qlike": math.log(forecast) + realised / forecast,
    }


class _WorstFirst:
    """Heap root is the last ranked member, with ascending-ID ties preserved."""

    def __init__(self, candidate):
        self.candidate = candidate

    def __lt__(self, other):
        a, b = self.candidate, other.candidate
        return (a.activity < b.activity) or (a.activity == b.activity and a.asset_id > b.asset_id)


def _calendar_dates(schedule, spec):
    start = date.fromisoformat(spec["signal_start"])
    end = date.fromisoformat(spec["selection_information_cutoff_session"])
    result = []
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            day = schedule.month_end(year, month).session
            if start <= day <= end:
                result.append(day)
    return tuple(result)


def _materialize(
    snapshot,
    schedule,
    dates,
    cutoff,
    audit_path,
    spec,
    budget_check=lambda: None,
    etf_snapshot=None,
):
    """Read each authenticated history once; retain only current top150 windows."""
    heaps, etfs, counts = {day: [] for day in dates}, {day: {} for day in dates}, Counter()
    etf_ids = tuple("etf:" + symbol.lower() + ".us" for symbol in spec["panels"]["etf_symbols"])
    source_ids = tuple(a for a in snapshot.asset_ids if a.startswith("stock:"))
    if len(source_ids) != 7195:
        raise ValueError("Full 7195-identity supplied stock population required")
    eligible_by_date = Counter()
    required_ids = (*source_ids, *etf_ids)
    etf_source = etf_snapshot if etf_snapshot is not None else snapshot
    stock_identity, etf_identity = (
        identity_from_source(snapshot, schedule),
        identity_from_source(etf_source, schedule),
    )
    missing_ids = set(etf_ids) - set(etf_source.asset_ids)
    if missing_ids:
        raise ValueError("Declared ETF source identities absent: " + str(sorted(missing_ids)))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("xb") as audit, snapshot.verified_read_session() as verified:
        for asset_id in required_ids:
            budget_check()
            kind, symbol = asset_id.split(":", 1)
            reader = etf_source if kind == "etf" and etf_snapshot is not None else verified
            history = reader.bars(kind, symbol.removesuffix(".us"))
            source_identity = etf_identity if kind == "etf" else stock_identity
            if kind == "stock" and not history:
                counts["empty_stock_sources"] += 1
            days = tuple(b.session for b in history)
            if days != tuple(sorted(set(days))):
                raise ValueError("Ordered unique authenticated source history required")
            for day in dates:
                i = bisect_left(schedule.sessions, day)
                start_day = schedule.sessions[i - 252] if i >= 252 else schedule.sessions[0]
                past = history[bisect_left(days, start_day) : bisect_right(days, day)]
                future_days, maturity = proxy_schedule(schedule, day)
                # The entire raw partition was authenticated. No future outcome
                # arithmetic is performed here, and immature future marks are not retained.
                future = (
                    history[bisect_left(days, future_days[0]) : bisect_right(days, future_days[-1])]
                    if maturity <= cutoff
                    else ()
                )
                candidate, reasons = asset_window(
                    asset_id, past, schedule, signal_session=day, future_rows=future
                )
                audit.write(
                    (
                        json.dumps(
                            {
                                "asset_id": asset_id,
                                "source_identity": source_identity.as_dict(),
                                "signal_session": day,
                                "eligible": candidate is not None,
                                "reasons": reasons,
                                "input_sha256": candidate.input_sha256 if candidate else None,
                                "activity": candidate.activity if candidate else None,
                            },
                            sort_keys=True,
                            allow_nan=False,
                            default=lambda x: x.isoformat(),
                        )
                        + "\n"
                    ).encode()
                )
                counts["eligible" if candidate else "excluded"] += 1
                if kind == "etf":
                    if candidate:
                        etfs[day][asset_id] = candidate
                    continue
                if candidate is not None:
                    eligible_by_date[day.isoformat()] += 1
                    heap = heaps[day]
                    item = _WorstFirst(candidate)
                    if len(heap) < 150:
                        heapq.heappush(heap, item)
                    elif heap[0] < item:
                        heapq.heapreplace(heap, item)
            counts["source_histories"] += 1
        audit.flush()
        os.fsync(audit.fileno())
    # Mandatory verified-session postflight has exited before inputs are consumed.
    if (
        identity_from_source(snapshot, schedule) != stock_identity
        or identity_from_source(etf_source, schedule) != etf_identity
    ):
        raise ValueError("Stock/ETF source identity changed during materialization")
    panels = {}
    for day in dates:
        nested = select_nested(item.candidate for item in heaps[day])
        panels[day] = {f"stock{size}": nested[size] for size in (50, 100, 150)}
        panels[day]["etf8"] = (
            tuple(etfs[day][asset] for asset in etf_ids) if len(etfs[day]) == 8 else ()
        )
    return panels, {
        **counts,
        "stock_source_identities": len(source_ids),
        "stock_eligible_by_date": dict(eligible_by_date),
        "candidate_audit_sha256": file_sha256(audit_path),
        "stock_source_identity": stock_identity.as_dict(),
        "etf_source_identity": etf_identity.as_dict(),
    }


def _restore(path, expected_sha, root, pins):
    path = Path(path).resolve()
    origin, chain, cells, seen = path.parent.parent, [], [], set()
    while path is not None:
        if path in seen or origin not in path.parents or file_sha256(path) != expected_sha:
            raise ValueError("Checkpoint chain hash/path mismatch")
        seen.add(path)
        receipt = json.loads(path.read_text())
        if receipt["pins"] != json.loads(_raw(pins)) or receipt["complete"] is not False:
            raise ValueError("Checkpoint source/config/input/runtime pins mismatch")
        cells[0:0] = receipt["cells"]
        chain.append(path)
        parent = receipt.get("parent")
        path = (origin / parent["path"]).resolve() if parent else None
        expected_sha = parent["sha256"] if parent else None
    keys = [(c["signal_session"], c["panel_id"], c["observations"], c["recipe_id"]) for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate checkpoint cell")
    for cell in cells:
        for artifact in cell["artifacts"]:
            source, target = (
                (origin / artifact["path"]).resolve(),
                (root / artifact["path"]).resolve(),
            )
            if origin not in source.parents or root not in target.parents or target.exists():
                raise ValueError("Checkpoint artifact path/fresh-output violation")
            if file_sha256(source) != artifact["sha256"]:
                raise ValueError("Checkpoint artifact SHA mismatch")
    for cell in cells:
        for artifact in cell["artifacts"]:
            target = root / artifact["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / artifact["path"], target)
    for path in chain:
        target = root / path.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return cells


def paired_summary(rows, dates, spec):
    """Select per declared panel/window; unlike risk probes are never pooled."""
    recipe_ids = tuple(r.recipe_id for r in recipes(spec))
    indexed = {
        (r["signal_session"], r["panel_id"], r["observations"], r["recipe_id"]): r for r in rows
    }
    expected = {
        (d.isoformat(), panel, t, recipe)
        for d in dates
        for panel, _, t in panel_grid(spec)
        for recipe in recipe_ids
    }
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Partial research grid: final comparison refused")
    summaries = []
    for panel, _, t in panel_grid(spec):
        common = [
            d.isoformat()
            for d in dates
            if all(
                indexed[d.isoformat(), panel, t, r]["score"]["status"] == "valid"
                for r in recipe_ids
            )
        ]
        means = {
            r: math.fsum(indexed[d, panel, t, r]["score"]["qlike"] for d in common) / len(common)
            if common
            else None
            for r in recipe_ids
        }
        summaries.append(
            {
                "panel_id": panel,
                "observations": t,
                "common_dates": common,
                "expected_dates": len(dates),
                "common_date_count": len(common),
                "all_observed_support": {
                    r: dict(
                        Counter(
                            indexed[d.isoformat(), panel, t, r]["score"]["status"] for d in dates
                        )
                    )
                    for r in recipe_ids
                },
                "mean_qlike_common": means,
                "selected_recipe_id": min(recipe_ids, key=lambda r: (means[r], r))
                if common
                else None,
            }
        )
    all_common = [
        d.isoformat()
        for d in dates
        if all(
            indexed[d.isoformat(), p, t, r]["score"]["status"] == "valid"
            for p, _, t in panel_grid(spec)
            for r in recipe_ids
        )
    ]
    return {
        "panels": summaries,
        "fully_paired_96_cell_dates": all_common,
        "global_winner": None,
        "selection_scope": "separate_panel_and_window_development_only",
        "predeclared_integration_panels": {"etf_scaler": "etf8_T63", "stock_risk": "stock100_T252"},
        "economic_evaluation": "not_run",
        "uncertainty": "not_run",
    }


def run_covariance_study(
    snapshot,
    schedule,
    spec_path,
    output,
    *,
    resume_checkpoint=None,
    expected_resume_sha256=None,
    service_deadline_monotonic=None,
    etf_snapshot=None,
):
    """Full actual-population cloud run; host checks use the pure primitives only."""
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Full risk studies require reviewed Kaggle runtime")
    if not isinstance(snapshot, MarketSnapshot) or not isinstance(schedule, XNYSSchedule):
        raise ValueError("Authenticated full source snapshot and actual calendar required")
    if (resume_checkpoint is None) != (expected_resume_sha256 is None):
        raise ValueError("Explicit checkpoint path and external SHA required together")
    spec = load_spec(spec_path)
    stock_identity = identity_from_source(snapshot, schedule)
    etf_source = etf_snapshot if etf_snapshot is not None else snapshot
    etf_identity = identity_from_source(etf_source, schedule)
    config_sha, closure = file_sha256(Path(spec_path)), _source_closure()
    cutoff = schedule.session(date(2014, 12, 31)).decision_at
    dates = _calendar_dates(schedule, spec)
    runtime = {
        name: version(name)
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "pyarrow")
    }
    runtime.update(
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        threads=4,
    )
    pins = {
        "stock_source_identity": stock_identity.as_dict(),
        "etf_source_identity": etf_identity.as_dict(),
        "calendar": schedule.provenance,
        "spec_sha256": config_sha,
        "source": closure,
        "runtime": runtime,
        "dates": dates,
        "recipes": [asdict(r) for r in recipes(spec)],
    }
    root = Path(output).resolve()
    if any(
        root == source.root.resolve() or source.root.resolve() in root.parents
        for source in (snapshot, etf_source)
    ):
        raise ValueError("Output must be outside immutable source")
    root.mkdir(parents=True, exist_ok=False)
    cells, results = [], []
    current = None

    def budget_check():
        if (
            service_deadline_monotonic is not None
            and time.monotonic() >= service_deadline_monotonic
        ):
            raise TimeoutError(
                "Service budget exhausted; partial checkpoint is not research completion"
            )

    try:
        budget_check()
        _write(
            root / "config.json",
            {
                "pins": pins,
                "spec": spec,
                "complete": False,
                "resume_checkpoint_sha256": expected_resume_sha256,
            },
        )
        panels, support = _materialize(
            snapshot,
            schedule,
            dates,
            cutoff,
            root / "candidate-audit.jsonl",
            spec,
            budget_check,
            etf_snapshot=etf_snapshot,
        )
        _write(root / "input-support.json", support)
        pins["candidate_audit_sha256"] = support["candidate_audit_sha256"]
        if resume_checkpoint is not None:
            cells = _restore(resume_checkpoint, expected_resume_sha256, root, pins)
        restored = {
            (c["signal_session"], c["panel_id"], c["observations"], c["recipe_id"]): c
            for c in cells
        }
        from threadpoolctl import threadpool_limits

        for day in dates:
            for panel_id, size, count in panel_grid(spec):
                panel = panels[day][panel_id]
                window = (
                    return_window(
                        panel,
                        schedule,
                        count,
                        signal_session=day,
                        source_identity=etf_identity if panel_id == "etf8" else stock_identity,
                    )
                    if panel
                    else None
                )
                for recipe in recipes(spec):
                    budget_check()
                    key = (day.isoformat(), panel_id, count, recipe.recipe_id)
                    current = key
                    relative = f"cells/{day}/{panel_id}_T{count}/{recipe.recipe_id}.json"
                    forecast_path = root / relative
                    estimate_input = (
                        window.window_hash
                        if window
                        else _digest({"panel": panel_id, "day": day, "support": "unavailable"})
                    )
                    if key in restored:
                        saved = restored[key]
                        if saved["forecast_path"] != relative or relative not in {
                            a["path"] for a in saved["artifacts"]
                        }:
                            raise ValueError("Resumed forecast semantic role mismatch")
                        forecast = json.loads(forecast_path.read_text())
                        if (
                            forecast["input_sha256"] != estimate_input
                            or forecast["asset_ids"] != [c.asset_id for c in panel]
                            or forecast["recipe"] != json.loads(_raw(asdict(recipe)))
                            or forecast["signal_session"] != day.isoformat()
                            or forecast["panel_id"] != panel_id
                            or forecast["observations"] != count
                            or forecast["source_identity"]
                            != (etf_identity if panel_id == "etf8" else stock_identity).as_dict()
                        ):
                            raise ValueError("Resumed forecast input/order/recipe mismatch")
                        if (
                            forecast["units"] != covariance.UNITS
                            or forecast["max_available_at"]
                            != schedule.session(day).decision_at.isoformat()
                        ):
                            raise ValueError("Resumed forecast units/availability mismatch")
                        if forecast["status"] == "valid":
                            matrix = np.asarray(forecast["matrix"], dtype=float)
                            if matrix.shape != (size, size) or not np.isfinite(matrix).all():
                                raise ValueError("Resumed covariance shape/finite-value mismatch")
                            tolerance = max(1e-15, 1e-10 * float(np.linalg.norm(matrix, ord=2)))
                            if (
                                np.linalg.norm(matrix - matrix.T, ord=2) > tolerance
                                or np.linalg.eigvalsh(matrix)[0] < -tolerance
                            ):
                                raise ValueError("Resumed covariance symmetry/PSD mismatch")
                            q = np.full(size, 1 / size)
                            if not math.isclose(
                                float(q @ matrix @ q),
                                forecast["variance"],
                                rel_tol=1e-12,
                                abs_tol=1e-18,
                            ):
                                raise ValueError(
                                    "Resumed common-probe forecast differs from matrix"
                                )
                    else:
                        forecast = {
                            "signal_session": day.isoformat(),
                            "panel_id": panel_id,
                            "observations": count,
                            "requested_assets": size,
                            "actual_assets": len(panel),
                            "asset_ids": [c.asset_id for c in panel],
                            "recipe": asdict(recipe),
                            "input_sha256": estimate_input,
                            "max_available_at": schedule.session(day).decision_at,
                            "variance": None,
                            "status": "past_panel_unavailable",
                            "units": covariance.UNITS,
                            "source_identity": (
                                etf_identity if panel_id == "etf8" else stock_identity
                            ).as_dict(),
                        }
                        if window is not None:
                            try:
                                with threadpool_limits(limits=4):
                                    estimate = estimate_covariance(
                                        window, recipe.method, **dict(recipe.parameters)
                                    )
                                forecast.update(
                                    status=estimate.status,
                                    diagnostics=estimate.diagnostics,
                                    parameters=estimate.parameters,
                                    configuration_hash=estimate.configuration_hash,
                                    fitted_at=estimate.fitted_at,
                                    training_start=estimate.training_start,
                                    training_end=estimate.training_end,
                                    max_available_at=estimate.max_available_at,
                                    matrix=estimate.matrix.tolist(),
                                    location=estimate.location.tolist(),
                                    runtime=estimate.runtime,
                                    variance=estimate.portfolio_variance(
                                        np.full(size, 1 / size), window.asset_ids
                                    )
                                    if estimate.status == "valid"
                                    else None,
                                )
                            except ValueError as error:
                                # Numerical failure is an explicit cell, never a fallback.
                                forecast.update(status="estimator_unavailable", reason=str(error))
                        _write(forecast_path, forecast)
                    # Forecast file is sealed before any future target is calculated.
                    proxy = future_proxy(
                        panel, schedule, signal_session=day, information_cutoff=cutoff
                    )
                    scored = score_forecast(forecast["variance"], proxy)
                    if forecast["status"] != "valid":
                        scored = {
                            "status": forecast["status"],
                            "squared_variance_error": None,
                            "qlike": None,
                        }
                    result = {
                        "signal_session": day.isoformat(),
                        "panel_id": panel_id,
                        "observations": count,
                        "recipe_id": recipe.recipe_id,
                        "forecast_sha256": file_sha256(forecast_path),
                        "proxy": proxy,
                        "score": scored,
                    }
                    result_path = root / relative.replace("cells/", "scores/")
                    if key in restored:
                        if result_path.relative_to(root).as_posix() not in {
                            a["path"] for a in restored[key]["artifacts"]
                        }:
                            raise ValueError("Resumed score semantic role lacks bound artifact")
                        old = json.loads(result_path.read_text())
                        if _raw(old) != _raw(result):
                            raise ValueError(
                                "Resumed score/proxy differs from actual clock-gated inputs"
                            )
                    else:
                        _write(result_path, result)
                        cells.append(
                            {
                                "signal_session": day.isoformat(),
                                "panel_id": panel_id,
                                "observations": count,
                                "recipe_id": recipe.recipe_id,
                                "forecast_path": relative,
                                "artifacts": [
                                    {
                                        "path": p.relative_to(root).as_posix(),
                                        "sha256": file_sha256(p),
                                    }
                                    for p in (forecast_path, result_path)
                                ],
                            }
                        )
                        previous = root / "checkpoints" / f"{len(cells) - 1:06d}.json"
                        _write(
                            root / "checkpoints" / f"{len(cells):06d}.json",
                            {
                                "complete": False,
                                "pins": pins,
                                "cells": [cells[-1]],
                                "parent": {
                                    "path": previous.relative_to(root).as_posix(),
                                    "sha256": file_sha256(previous),
                                }
                                if len(cells) > 1
                                else None,
                            },
                        )
                    results.append(result)
        summary = paired_summary(results, dates, spec)
        budget_check()
        if _source_closure() != closure or file_sha256(Path(spec_path)) != config_sha:
            raise ValueError("Source/spec changed during risk study")
        with snapshot.verified_read_session():
            pass
        if identity_from_source(etf_source, schedule) != etf_identity:
            raise ValueError("ETF source changed during risk study")
        _write(root / "forecast-comparison.json", summary)
        manifest = {
            "schema": "risk_common_probe_development_v1",
            "complete": True,
            "pins": pins,
            "expected_cells": len(dates) * 96,
            "actual_cells": len(results),
            "support": support,
            "economic_evaluation": "not_run",
            "bootstrap_uncertainty": "not_run",
            "holdout_outcomes_constructed": False,
            "population": {
                "stock": "conditional_on_supplied_archive",
                "etf": "eight_prespecified_sleeves_from_declared_authenticated_source",
            },
            "price_basis_by_panel": {
                "stock": stock_identity.price_basis,
                "etf": etf_identity.price_basis,
            },
            "artifacts": [
                {"path": p.relative_to(root).as_posix(), "sha256": file_sha256(p)}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ],
        }
        budget_check()
        _write(root / "manifest.json", manifest)
        digest = file_sha256(root / "manifest.json")
        with (root / "COMPLETE").open("xb") as handle:
            handle.write((digest + "\n").encode())
        return {**manifest, "manifest_sha256": digest}
    except BaseException as error:
        _write(
            root / "FAILURE.json",
            {
                "complete": False,
                "current_cell": current,
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_cells": len(cells),
            },
        )
        raise
