"""Prospective matched covariance economics: 26 arms × four fees × two panels.

Completed statistical forecasts supply matrices, never outcome values or winner
selection. Each path owns its positions, signal-close turnover penalty, actual
cost funding and missing-mark failures in the unchanged shared engine.

Current resume authenticates completed paths only. Unsupported terminal evidence
is preserved, but a fresh resumed family reattempts those paths once; terminal
unsupported-cache reuse needs a separately reviewed externally frozen contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import time
import tomllib
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import numpy as np

from quantlab.allocation import study as allocation_study
from quantlab.backtest.engine import Decision, run_history
from quantlab.data.archive import file_sha256
from quantlab.data.provenance import identity_from_source
from quantlab.evaluation.attribution import monetary_contributions
from quantlab.risk import execution, optimization, study
from quantlab.risk.baseline_execution import project_baseline
from quantlab.risk.covariance import UNITS, CovarianceEstimate
from quantlab.risk.signal_weights import POLICY, POLICY_SHA256, project_signal_weights

FEES = (0.0, 5.0, 10.0, 25.0)
ETAS = (0.0, 0.01, 0.1)
PANELS = (("etf8", 8, 63, 0.35), ("stock100", 100, 252, 0.10))
START, END = date(2007, 2, 28), date(2014, 12, 31)


def _raw(value):
    return (
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False, default=lambda v: v.isoformat()
        )
        + "\n"
    ).encode()


def _hash(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_raw(value))


def _relative(root, name):
    value = PurePosixPath(name)
    target = (root / name).resolve()
    if (
        value.is_absolute()
        or ".." in value.parts
        or "\\" in name
        or ":" in name
        or value.as_posix() != name
        or not target.is_relative_to(root.resolve())
        or (root / name).is_symlink()
    ):
        raise ValueError("Regular relative authenticated artifact required")
    return target


def source_closure():
    root = Path(__file__).resolve().parents[1]
    names = set(allocation_study._sources()) | {
        "risk/study.py",
        "risk/optimization.py",
        "risk/execution.py",
        "risk/baseline_execution.py",
        "risk/economic_study.py",
        "risk/signal_weights.py",
        "data/returns.py",
        "evaluation/attribution.py",
    }
    return {name: file_sha256(root / name) for name in sorted(names)}


def load_spec(path):
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    checks = {
        "schema": "risk_economic_development_v1",
        "start_signal": str(START),
        "end_close": str(END),
        "initial_cash": 100000.0,
        "fees_bps": list(FEES),
        "turnover_etas": list(ETAS),
        "expected_paths": 208,
        "etf_panel": "etf8",
        "etf_observations": 63,
        "etf_cap": 0.35,
        "stock_panel": "stock100",
        "stock_observations": 252,
        "stock_cap": 0.10,
        "baselines": ["equal_weight", "capped_inverse_volatility_not_risk_parity"],
        "inverse_volatility_reference": "sample_ddof1",
        "objective": (
            "daily_variance_plus_eta_times_mean_daily_diagonal_variance_times_signal_L1_union_turnover"
        ),
        "solver_failure": "HOLD_actual_previous_positions_and_cash_initially",
        "data_math_provenance_runtime_failure": "hard_stop_not_solver_fallback",
        "missing_held_or_fill_mark": "incomplete_valid_prefix_no_gap_repair",
        "signal_weight_policy": POLICY["policy_id"],
        "signal_weight_maximum_l1_adjustment": POLICY["maximum_l1_adjustment"],
        "signal_weight_nav_reconciliation_ulps": POLICY["nav_reconciliation_ulps"],
        "unresolved_economic_path": (
            "terminal_unsupported_preserve_partial_evidence_continue_other_paths_no_full_score"
        ),
        "projection": "registered_tiny_clip_reduce_overbudget_no_topup_no_matrix_change",
        "cash": "explicit_zero_interest_assumption",
        "minimum_seconds_before_path": 120,
        "resume": "externally_frozen_completed_paths_only_no_midpath_resume",
        "selection": "none_all_208_paths_and_all_four_assumed_fees_reported",
        "forecast_outcomes_parsed": False,
        "holdout_outcomes_parsed": False,
    }
    if any(spec.get(key) != value for key, value in checks.items()):
        raise ValueError("Economic family/protocol changed; resolve before any scores")
    return spec


@dataclass(frozen=True)
class Arm:
    arm_id: str
    baseline: str | None
    recipe_id: str
    eta: float


def arms(forecast_spec):
    return (
        Arm("equal_weight", "equal_weight", "sample_ddof1", 0.0),
        Arm("capped_inverse_volatility", "capped_inverse_volatility", "sample_ddof1", 0.0),
        *(
            Arm(
                recipe.recipe_id + "_eta" + format(eta, "g").replace(".", "p"),
                None,
                recipe.recipe_id,
                eta,
            )
            for recipe in study.recipes(forecast_spec)
            for eta in ETAS
        ),
    )


def authenticate_forecasts(root, expected_sha, schedule, forecast_spec_path):
    """Authenticate the fixed integration subset; parse no outcomes or rankings.

    The complete upstream manifest authenticates the 9,120-cell study. This
    consumer requires only all 1,520 declared integration forecasts and their
    source support, without claiming to have downloaded the other cells.
    """
    root = Path(root)
    if (
        file_sha256(root / "manifest.json") != expected_sha
        or (root / "COMPLETE").read_bytes() != (expected_sha + "\n").encode()
        or (root / "FAILURE.json").exists()
    ):
        raise ValueError("Externally pinned completed full forecast study required")
    manifest = json.loads((root / "manifest.json").read_bytes())
    pins = manifest["pins"]
    if not (
        manifest["complete"] is True
        and manifest["actual_cells"] == 9120
        and manifest["expected_cells"] == 9120
        and pins["calendar"] == schedule.provenance
        and pins["source"] == study._source_closure()
        and pins["spec_sha256"] == file_sha256(Path(forecast_spec_path))
    ):
        raise ValueError("Forecast source/config/calendar/full-grid contract differs")
    inventory = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    if len(inventory) != len(manifest["artifacts"]):
        raise ValueError("Duplicate forecast artifact")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    dates = tuple(pins["dates"])
    recipe_ids = tuple(r.recipe_id for r in study.recipes(study.load_spec(forecast_spec_path)))
    required = {
        f"cells/{day}/{panel}_T{count}/{recipe}.json"
        for day in dates
        for panel, _, count, _ in PANELS
        for recipe in recipe_ids
    } | {"config.json", "candidate-audit.jsonl", "input-support.json"}
    allowed = required | {"forecast-comparison.json"}
    if (
        len(dates) != 95
        or len(recipe_ids) != 8
        or not required <= set(inventory)
        or not required <= actual
        or not actual <= allowed | {"manifest.json", "COMPLETE"}
    ):
        raise ValueError("Exact predeclared integration forecast profile required")
    for name in actual - {"manifest.json", "COMPLETE"}:
        if file_sha256(_relative(root, name)) != inventory[name]:
            raise ValueError("Forecast artifact hash mismatch")
    return manifest, inventory


def rehydrate_estimate(record, window, recipe):
    """Recover a saved actual estimate only after rebuilding its exact past inputs."""
    if not (
        record["status"] == "valid"
        and record["units"] == UNITS
        and record["input_sha256"] == window.window_hash
        and tuple(record["asset_ids"]) == window.asset_ids
        and record["recipe"] == json.loads(_raw(asdict(recipe)))
        and record["signal_session"] == window.signal_at.date().isoformat()
        and record["observations"] == len(window.values)
    ):
        raise ValueError("Saved estimate input/order/method/units/status mismatch")
    matrix, location = (
        np.asarray(record["matrix"], dtype=float),
        np.asarray(record["location"], dtype=float),
    )
    if (
        matrix.shape != (len(window.asset_ids),) * 2
        or location.shape != (len(window.asset_ids),)
        or not np.isfinite(matrix).all()
        or not np.isfinite(location).all()
    ):
        raise ValueError("Invalid saved actual covariance/location")
    clocks = [
        datetime.fromisoformat(record[name])
        for name in ("training_start", "training_end", "max_available_at", "fitted_at")
    ]
    if (
        clocks[0] != window.interval_start[0]
        or clocks[1] != window.interval_end[-1]
        or clocks[2] != max(window.available_at)
        or any(t.tzinfo is None or t.utcoffset() is None for t in clocks)
    ):
        raise ValueError("Saved covariance availability/window clocks differ")
    config = hashlib.sha256(
        json.dumps(
            {"method": recipe.method, "parameters": record["parameters"]}, sort_keys=True
        ).encode()
    ).hexdigest()
    if config != record["configuration_hash"]:
        raise ValueError("Saved estimator configuration hash differs")
    provenance = {
        **dict(window.source_provenance),
        "source_snapshot_id": window.source_snapshot_id,
        "return_convention": window.return_convention,
        "eligibility_rule_id": window.eligibility_rule_id,
        "coverage_report_id": window.coverage_report_id,
    }
    matrix.setflags(write=False)
    location.setflags(write=False)
    estimate = CovarianceEstimate(
        matrix,
        location,
        window.asset_ids,
        recipe.method,
        record["parameters"],
        record["diagnostics"],
        window.window_hash,
        window.signal_at,
        *clocks,
        record["runtime"],
        config,
        provenance,
    )
    optimization._covariance(estimate)  # Same strict PSD, units and historical-availability gate.
    return estimate


def decide(
    view, holdings, arm, estimate, reference, cap, solver_policy, projection_path, projection_sha
):
    """One dated estimate and frozen signal holdings; no future proxy or source handle."""
    if (
        estimate.signal_at != view.cutoff
        or reference.signal_at != view.cutoff
        or estimate.asset_ids != reference.asset_ids
        or holdings.valued_at != reference.training_end
        or holdings.nav <= 0
    ):
        raise ValueError("Exact current dated estimates and signal-close holdings required")
    representation = project_signal_weights(holdings)
    old = dict(representation.applied_weights)
    penalty_bound = (
        arm.eta * float(np.diag(estimate.matrix).mean()) * representation.total_l1_adjustment
    )
    weight_audit = {
        **representation.to_record(),
        "turnover_penalty_absolute_impact_bound": penalty_bound,
        "penalty_basis": "eta_times_mean_daily_diagonal_times_L1_oldweight_change",
    }
    if arm.baseline:
        raw = (
            optimization.equal_weight(reference.asset_ids, cap)
            if arm.baseline == "equal_weight"
            else optimization.capped_inverse_volatility(reference, cap)
        )
        projection = project_baseline(
            raw, reference, policy_path=projection_path, expected_policy_sha256=projection_sha
        )
    else:
        raw = optimization.minimum_variance(
            estimate, cap, old, arm.eta, frozen_solver_policy=solver_policy
        )
        if raw.status == "solver_failed":
            # An explicit prior policy: retain each path's actual exposure, never substitute cash.
            return Decision(
                "HOLD",
                None,
                view.cutoff,
                view.next_open_at,
                "declared_solver_failure_HOLD_actual_previous",
            ), {
                "raw_solution": raw.to_record(),
                "action": "HOLD",
                "signal_close_weights": old,
                "signal_weight_representation": weight_audit,
                "projection": None,
            }
        if raw.status != "valid_raw_target":
            raise ValueError("Non-solver mathematical failure cannot become a HOLD fallback")
        projection = execution.project_for_execution(
            raw, estimate, policy_path=projection_path, expected_policy_sha256=projection_sha
        )
    target = tuple(zip(projection.asset_ids, projection.applied_weights, strict=True))
    return Decision("REBALANCE", target, view.cutoff, view.next_open_at, arm.arm_id), {
        "raw_solution": raw.to_record(),
        "projection": projection.to_record(),
        "action": "REBALANCE",
        "signal_close_weights": old,
        "signal_weight_representation": weight_audit,
        "target_annual_volatility": math.sqrt(
            252 * estimate.portfolio_variance(tuple(w for _, w in target), estimate.asset_ids)
        ),
    }


def run_path(
    histories,
    schedule,
    *,
    arm,
    estimates,
    cap,
    source_identity,
    fee_bps,
    solver_policy,
    projection_path,
    projection_sha,
    start_signal,
    end,
    deadline=None,
    event_sink=None,
):
    """Low-level path also permits short actual engineering checks, never a hidden full family."""
    if start_signal >= date(2015, 1, 1) or end >= date(2015, 1, 1):
        raise ValueError("Development economic path refuses shared holdout outcomes")
    if deadline is not None and (
        isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= time.monotonic()
    ):
        raise ValueError("Future finite path deadline required")
    records = []

    def callback(view, holdings):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Incomplete current path; earlier completed paths remain reusable")
        if view.session not in estimates:
            return Decision(
                "HOLD", None, view.cutoff, view.next_open_at, "between_declared_signals"
            )
        dated = estimates[view.session]  # Exactly current date; no outcome or future lookup.
        decision, audit = decide(
            view,
            holdings,
            arm,
            dated[arm.recipe_id],
            dated["sample_ddof1"],
            cap,
            solver_policy,
            projection_path,
            projection_sha,
        )
        audit.update(
            signal_at=view.cutoff.isoformat(),
            intended_execution_at=view.next_open_at.isoformat(),
            arm_id=arm.arm_id,
            source_identity=source_identity.as_dict(),
            execution_within_requested_period=view.next_open_at.date() <= end,
        )
        records.append(audit)
        if event_sink:
            event_sink(audit)
        return decision

    result = run_history(
        histories,
        schedule,
        start=start_signal,
        end=end,
        initial_cash=100000.0,
        fee_bps=fee_bps,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=callback,
        source_identity=source_identity,
    )
    expected = tuple(d for d in schedule.sessions if start_signal <= d <= end)
    if result.failure or result.incomplete_day or result.economic_support != expected:
        return {"complete": False, "engine": result, "targets": records}
    measured = result.daily[1:]
    return {
        "complete": True,
        "engine": result,
        "targets": records,
        "measured": measured,
        "metrics": allocation_study._metrics(measured, 100000.0),
        "attribution": monetary_contributions(measured, initial_nav=100000.0),
    }


def _prepare_inputs(snapshot, etf_source, schedule, forecasts, root, recipes, identities, dates):
    """Union is an IO optimization only; dated panels come from full-population past selection."""
    records, required = {}, {panel: set() for panel, *_ in PANELS}
    for panel, count, observations, _ in PANELS:
        for day in dates:
            current = {}
            for recipe in recipes:
                path = root / f"cells/{day}/{panel}_T{observations}/{recipe.recipe_id}.json"
                record = json.loads(path.read_bytes())
                ids = tuple(record["asset_ids"])
                if not (
                    record["panel_id"] == panel
                    and record["requested_assets"] == count
                    and record["actual_assets"] == count
                    and len(set(ids)) == count
                    and record["source_identity"] == identities[panel].as_dict()
                ):
                    raise ValueError("Actual dated full panel/source identity differs")
                if current and ids != tuple(next(iter(current.values()))["asset_ids"]):
                    raise ValueError("Estimators do not share the exact ordered past-only panel")
                current[recipe.recipe_id] = record
                required[panel].update(ids)
            records[panel, day] = current
    stock_ids = tuple(a for a in snapshot.asset_ids if a.startswith("stock:"))
    if (
        len(stock_ids) != 7195
        or not required["stock100"] <= set(stock_ids)
        or forecasts["support"]["stock_source_identities"] != 7195
    ):
        raise ValueError("Original full 7195-identity stock population proof required")
    histories = {panel: {} for panel, *_ in PANELS}
    begin = schedule.sessions[schedule.sessions.index(dates[0]) - 252]
    with snapshot.verified_read_session() as verified:
        for panel in required:
            source = etf_source if panel == "etf8" else verified
            for asset in sorted(required[panel]):
                kind, symbol = asset.split(":", 1)
                history = source.bars(kind, symbol.removesuffix(".us"))
                days = tuple(b.session for b in history)
                # Observed records only, never inferred endpoints or future-coverage eligibility.
                histories[panel][asset] = history[
                    bisect_left(days, begin) : bisect_right(days, END)
                ]
    estimates = {panel: {} for panel, *_ in PANELS}
    for panel, _, observations, _ in PANELS:
        for day in dates:
            ids = tuple(records[panel, day]["sample_ddof1"]["asset_ids"])
            position = schedule.sessions.index(day)
            start = schedule.sessions[position - 252]
            candidates = []
            for asset in ids:
                history = histories[panel][asset]
                days = tuple(b.session for b in history)
                candidate, reasons = study.asset_window(
                    asset,
                    history[bisect_left(days, start) : bisect_right(days, day)],
                    schedule,
                    signal_session=day,
                )
                if candidate is None:
                    raise ValueError(
                        "Authenticated past window no longer available: " + str(reasons)
                    )
                candidates.append(candidate)
            window = study.return_window(
                candidates,
                schedule,
                observations,
                signal_session=day,
                source_identity=identities[panel],
            )
            estimates[panel][day] = {
                recipe.recipe_id: rehydrate_estimate(
                    records[panel, day][recipe.recipe_id], window, recipe
                )
                for recipe in recipes
            }
    inputs = {
        panel: {asset: _hash([b.row_sha256 for b in history]) for asset, history in values.items()}
        for panel, values in histories.items()
    }
    return histories, estimates, inputs


def _path_manifest(root, expected_sha, pins):
    """A complete immutable path is reusable only with exact current experiment pins."""
    if (
        file_sha256(root / "manifest.json") != expected_sha
        or (root / "COMPLETE").read_bytes() != (expected_sha + "\n").encode()
        or (root / "FAILURE.json").exists()
    ):
        raise ValueError("Only externally frozen complete economic paths are reusable")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        manifest["complete"] is not True
        or manifest["pins"] != pins
        or manifest["sessions"] != 1975
        or manifest["first_session"] != "2007-03-01"
        or manifest["last_session"] != "2014-12-31"
    ):
        raise ValueError("Completed path source/config/input/runtime/full-support pins differ")
    expected = manifest["outputs_sha256"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(expected) | {"manifest.json", "COMPLETE"}:
        raise ValueError("Completed path artifact inventory differs")
    for name, digest in expected.items():
        if file_sha256(_relative(root, name)) != digest:
            raise ValueError("Completed path artifact hash differs")
    import pyarrow.parquet as pq

    sessions = pq.read_table(root / "ledger.parquet", columns=["session"])["session"].to_pylist()
    if [day.isoformat() for day in sessions] != pins["economic_sessions"]:
        raise ValueError("Completed path actual ledger session support differs")
    if json.loads((root / "metrics.json").read_bytes()) != manifest["metrics"]:
        raise ValueError("Completed path metric artifact role differs")
    return manifest


def _restore_paths(freeze_path, expected_sha, origin, destination, pins):
    if file_sha256(Path(freeze_path)) != expected_sha:
        raise ValueError("External completed-path resume freeze SHA differs")
    freeze = json.loads(Path(freeze_path).read_bytes())
    if (
        freeze["schema"] != "risk_economic_completed_paths_freeze_v1"
        or freeze["status"] != "READY_FOR_COMPLETED_PATH_REUSE"
        or freeze["pins"] != pins
    ):
        raise ValueError("Actual externally frozen completed-path inventory required")
    restored = {}
    for item in freeze["paths"]:
        path_id = item["path_id"]
        if path_id in restored or item["manifest_path"] != f"paths/{path_id}/manifest.json":
            raise ValueError("Duplicate or mismatched resumed path role")
        parent = _relative(Path(origin), item["manifest_path"]).parent
        manifest = _path_manifest(parent, item["manifest_sha256"], pins)
        if manifest["path_id"] != path_id:
            raise ValueError("Resumed path identity differs")
        target = _relative(destination, "paths/" + path_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(parent, target)
        _path_manifest(target, item["manifest_sha256"], pins)
        restored[path_id] = manifest
    return restored


def _save_path(root, path_id, result, pins):
    import pyarrow as pa
    import pyarrow.parquet as pq

    engine = result["engine"]
    _write(
        root / "endpoint.json",
        {
            "endpoint_status": engine.endpoint_status,
            "pending_status": engine.pending_status,
            "pending": asdict(engine.pending) if engine.pending else None,
            "failure": asdict(engine.failure) if engine.failure else None,
            "incomplete_day": asdict(engine.incomplete_day) if engine.incomplete_day else None,
        },
    )
    if not result["complete"]:
        _write(root / "partial-owned-ledger.json", [asdict(row) for row in engine.daily])
        _write(
            root / "FAILURE.json",
            {
                "complete": False,
                "support": engine.economic_support,
                "failure": asdict(engine.failure) if engine.failure else None,
            },
        )
        terminal = {
            "schema": "risk_economic_unsupported_path_v1",
            "complete": False,
            "terminal": True,
            "status": "unsupported_unresolved_economic_mark",
            "path_id": path_id,
            "pins": pins,
            "full_period_metrics": None,
            "valid_prefix_sessions": len(engine.daily),
            "outputs_sha256": {
                p.relative_to(root).as_posix(): file_sha256(p)
                for p in root.rglob("*") if p.is_file()
            },
        }
        _write(root / "manifest.json", terminal)
        (root / "UNSUPPORTED").write_bytes(
            (file_sha256(root / "manifest.json") + "\n").encode()
        )
        return terminal
    records = result["measured"]
    if len(records) != 1975 or records[0].session != date(2007, 3, 1) or records[-1].session != END:
        raise ValueError("Complete 1975-session measured path required")
    rows, ownership, trades = [], [], []
    for record in records:
        weights = [(asset, value / record.state.nav) for asset, value in record.state.values]
        concentration = math.fsum(w * w for _, w in weights)
        rows.append(
            {
                "session": record.session,
                "close_nav": record.state.nav,
                "cash": record.state.cash,
                "fee": record.execution.fee,
                "pretrade_nav": record.execution.nav_before,
                "traded_notional": record.execution.traded_notional,
                "full_turnover": record.execution.full_turnover,
                "money_pnl": record.money_pnl,
                "reconciliation_residual": record.reconciliation_residual,
                "weight_sum_squares": concentration,
                "effective_name_count": 1 / concentration if concentration > 0 else None,
                "holdings": [
                    {"asset_id": asset, "value": value, "close_weight": value / record.state.nav}
                    for asset, value in record.state.values
                ],
            }
        )
        overnight, intraday = dict(record.overnight.asset_pnl), dict(record.intraday.asset_pnl)
        ownership.extend(
            {
                "session": record.session,
                "asset_id": asset,
                "overnight_pnl": overnight.get(asset, 0.0),
                "intraday_pnl": intraday.get(asset, 0.0),
            }
            for asset in sorted(set(overnight) | set(intraday))
        )
        trades.extend({"session": record.session, **asdict(leg)} for leg in record.execution.legs)
    for name, values in (("ledger", rows), ("owned-asset-pnl", ownership), ("trades", trades)):
        if values:
            pq.write_table(
                pa.Table.from_pylist(values), root / (name + ".parquet"), compression="zstd"
            )
        else:
            _write(
                root / (name + "-empty.json"),
                {"rows": 0, "reason": "no_exposure_no_invented_records"},
            )
    _write(root / "metrics.json", result["metrics"])
    _write(root / "monetary-attribution.json", result["attribution"])
    manifest = {
        "schema": "risk_economic_complete_path_v1",
        "complete": True,
        "path_id": path_id,
        "pins": pins,
        "sessions": 1975,
        "first_session": "2007-03-01",
        "last_session": "2014-12-31",
        "metrics": result["metrics"],
        "solver_failure_HOLD_events": sum(r["action"] == "HOLD" for r in result["targets"]),
        "outputs_sha256": {
            p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()
        },
    }
    _write(root / "manifest.json", manifest)
    (root / "COMPLETE").write_bytes((file_sha256(root / "manifest.json") + "\n").encode())
    return manifest


def run_economic_study(
    snapshot,
    etf_source,
    schedule,
    forecast_root,
    expected_forecast_sha256,
    forecast_spec_path,
    spec_path,
    expected_spec_sha256,
    output,
    *,
    frozen_solver_policy,
    expected_solver_policy_id,
    projection_path,
    expected_projection_sha256,
    service_deadline_monotonic,
    resume_freeze_path=None,
    expected_resume_sha256=None,
    resume_origin=None,
):
    """Cloud-only full family; preserve every completed path and never select a partial family."""
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Full 208-path economic studies require reviewed Kaggle runtime")
    if (
        not isinstance(service_deadline_monotonic, (int, float))
        or isinstance(service_deadline_monotonic, bool)
        or not math.isfinite(service_deadline_monotonic)
    ):
        raise ValueError("Future finite service deadline required")

    def budget(reserve=0):
        if time.monotonic() + reserve >= service_deadline_monotonic:
            raise TimeoutError("Partial economic family; no selection or COMPLETE permitted")

    budget(120)
    if (
        not isinstance(frozen_solver_policy, optimization.SolverPolicy)
        or frozen_solver_policy.policy_id != expected_solver_policy_id
        or frozen_solver_policy.frozen_at > datetime.now(UTC)
    ):
        raise ValueError("Externally frozen actual solver policy required before data reads")
    if file_sha256(Path(spec_path)) != expected_spec_sha256:
        raise ValueError("Externally pinned economic configuration required")
    spec, forecast_spec = load_spec(spec_path), study.load_spec(forecast_spec_path)
    execution._policy(projection_path, expected_projection_sha256, datetime.now(UTC))
    original, etf = (
        identity_from_source(snapshot, schedule),
        identity_from_source(etf_source, schedule),
    )
    if original.source_type != "original_archive" or etf.source_type != "yahoo_etf_release":
        raise ValueError("Original stock and distinct Yahoo ETF source types required")
    identities = {"stock100": original, "etf8": etf}
    manifest, _ = authenticate_forecasts(
        forecast_root, expected_forecast_sha256, schedule, forecast_spec_path
    )
    if (
        manifest["pins"]["stock_source_identity"] != original.as_dict()
        or manifest["pins"]["etf_source_identity"] != etf.as_dict()
    ):
        raise ValueError("Statistical and economic source identities differ")
    dates = tuple(date.fromisoformat(value) for value in manifest["pins"]["dates"])
    if dates != study._calendar_dates(schedule, forecast_spec) or len(dates) != 95:
        raise ValueError("All 95 declared actual monthly signal dates required")
    recipes = study.recipes(forecast_spec)
    family = arms(forecast_spec)
    if len(family) != 26:
        raise ValueError("Exactly 26 fixed arms required")
    histories, estimates, inputs = _prepare_inputs(
        snapshot, etf_source, schedule, manifest, Path(forecast_root), recipes, identities, dates
    )
    budget(120)
    runtime = {name: version(name) for name in optimization.RUNTIME_PACKAGES}
    if tuple(sorted(runtime.items())) != frozen_solver_policy.expected_versions:
        raise ValueError("Actual numerical runtime differs from frozen solver policy")
    runtime.update(
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    sources = source_closure()
    pins = json.loads(
        _raw(
            {
                "schema": "risk_economic_family_pins_v1",
                "spec_sha256": expected_spec_sha256,
                "forecast_manifest_sha256": expected_forecast_sha256,
                "identities": {key: value.as_dict() for key, value in identities.items()},
                "input_rows_sha256": inputs,
                "source": sources,
                "runtime": runtime,
                "solver_policy_id": expected_solver_policy_id,
                "projection_sha256": expected_projection_sha256,
                "signal_weight_policy_sha256": POLICY_SHA256,
                "dates": dates,
                "economic_sessions": [day for day in schedule.sessions if START < day <= END],
                "panels": PANELS,
                "fees": FEES,
                "arms": [asdict(a) for a in family],
            }
        )
    )
    root = Path(output).resolve()
    if any(
        root == Path(source).resolve() or Path(source).resolve() in root.parents
        for source in (snapshot.root, etf_source.root, forecast_root)
    ):
        raise ValueError("Economic output must be outside immutable inputs")
    root.mkdir(parents=True, exist_ok=False)
    completed, current = {}, None
    try:
        _write(
            root / "contract.json",
            {
                "pins": pins,
                "spec": spec,
                "expected_paths": 208,
                "forecast_outcomes_parsed": False,
                "holdout_evaluated": False,
                "interpretation": (
                    "research_price_proxies_no_dividend_cash_or_observed_auction_fills"
                ),
            },
        )
        if not (
            (resume_freeze_path is None)
            == (expected_resume_sha256 is None)
            == (resume_origin is None)
        ):
            raise ValueError("Resume requires external freeze SHA and actual origin together")
        if resume_freeze_path is not None:
            completed = _restore_paths(
                resume_freeze_path, expected_resume_sha256, resume_origin, root, pins
            )
        requested = {
            f"{panel}/cost{fee:g}/{arm.arm_id}"
            for panel, *_ in PANELS
            for fee in FEES
            for arm in family
        }
        if not set(completed) <= requested:
            raise ValueError("Resumed inventory contains an unregistered economic path")
        for panel, _, _, cap in PANELS:
            for fee in FEES:
                for arm in family:
                    current = f"{panel}/cost{fee:g}/{arm.arm_id}"
                    if current in completed:
                        continue
                    budget(spec["minimum_seconds_before_path"])
                    path = root / "paths" / current
                    path.mkdir(parents=True, exist_ok=False)

                    def sink(record, path=path):
                        _write(path / "decisions" / (record["signal_at"][:10] + ".json"), record)

                    result = run_path(
                        histories[panel],
                        schedule,
                        arm=arm,
                        estimates=estimates[panel],
                        cap=cap,
                        source_identity=identities[panel],
                        fee_bps=fee,
                        solver_policy=frozen_solver_policy,
                        projection_path=projection_path,
                        projection_sha=expected_projection_sha256,
                        start_signal=START,
                        end=END,
                        deadline=service_deadline_monotonic,
                        event_sink=sink,
                    )
                    completed[current] = _save_path(path, current, result, pins)
                    _write(
                        root / "checkpoints" / f"{len(completed):03d}.json",
                        {
                            "complete_family": False,
                            "pins": pins,
                            "completed_paths": [
                                {
                                    "path_id": key,
                                    "manifest_path": f"paths/{key}/manifest.json",
                                    "manifest_sha256": file_sha256(
                                        root / "paths" / key / "manifest.json"
                                    ),
                                }
                                for key in sorted(completed) if completed[key]["complete"]
                            ],
                        },
                    )
        if set(completed) != requested or len(completed) != 208:
            raise ValueError("Partial economic family cannot produce comparison or COMPLETE")
        budget()
        if (
            source_closure() != sources
            or file_sha256(Path(spec_path)) != expected_spec_sha256
            or identity_from_source(snapshot, schedule) != original
            or identity_from_source(etf_source, schedule) != etf
        ):
            raise ValueError("Economic source/config/input changed during execution")
        authenticate_forecasts(
            forecast_root, expected_forecast_sha256, schedule, forecast_spec_path
        )
        _write(
            root / "all-path-metrics.json",
            {key: value["metrics"] for key, value in completed.items() if value["complete"]},
        )
        final = {
            "schema": "risk_economic_development_family_v1",
            "complete": all(value["complete"] for value in completed.values()),
            "pins": pins,
            "expected_paths": 208,
            "attempted_terminal_paths": 208,
            "completed_paths": sum(value["complete"] for value in completed.values()),
            "unsupported_paths": [
                key for key in sorted(completed) if not completed[key]["complete"]
            ],
            "comparison": "full_support_paths_only_no_partial_prefix_scores",
            "selection": "none_all_prespecified_paths_and_fees_reported",
            "holdout_evaluated": False,
            "forecast_outcomes_parsed": False,
            "paths": [
                {
                    "path_id": key,
                    "manifest_sha256": file_sha256(root / "paths" / key / "manifest.json"),
                }
                for key in sorted(completed)
            ],
            "all_path_metrics_sha256": file_sha256(root / "all-path-metrics.json"),
            "artifacts": [
                {"path": p.relative_to(root).as_posix(), "sha256": file_sha256(p)}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ],
        }
        budget()
        _write(root / "manifest.json", final)
        marker = "COMPLETE" if final["complete"] else "ATTEMPTED_UNSUPPORTED"
        (root / marker).write_bytes((file_sha256(root / "manifest.json") + "\n").encode())
        return final
    except BaseException as error:
        _write(
            root / "FAILURE.json",
            {
                "complete": False,
                "current_path": current,
                "completed_paths": len(completed),
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
