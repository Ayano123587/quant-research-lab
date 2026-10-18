"""Shared release gates and portfolio orchestration, without recipe selection.

Callbacks are trusted, source-authenticated research code, not a Python sandbox.
This module does not compute holdout metrics or covariance outcomes. Those
separate consumers must verify every declared comparison before global release.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from quantlab.backtest.engine import Decision, EngineResult, run_history
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity

START = date(2014, 12, 31)
END = date(2017, 11, 10)
COMPONENTS = {"stocks", "etf", "covariance", "integrations"}
REFITS = {"none", "initial_only_mature", "monthly_expanding_mature"}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        or value == "0" * 64
    ):
        raise ValueError("Actual externally pinned SHA256 required")
    return value


def _aware(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timezone-aware information clock required")
    return value


def _bound(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or path.is_symlink()
        or any(p.is_symlink() for p in path.parents if p != root.parent)
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
    ):
        raise ValueError("Regular contained artifact required")
    return path


def source_closure(project, entry_sources):
    """Hash transitive local imports and initializers without importing models."""
    project = Path(project).resolve()
    pending, seen = list(entry_sources), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = _bound(project, name)
        relative = path.relative_to(project).as_posix()
        if not relative.startswith("src/quantlab/") or path.suffix != ".py":
            raise ValueError("Audited quantlab production modules required")
        seen.add(relative)
        parts = list(Path(relative).relative_to("src").with_suffix("").parts)
        package = parts[:-1]
        for length in range(1, len(package) + 1):
            pending.append("src/" + "/".join(package[:length]) + "/__init__.py")
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join([*base, *([] if node.module is None else node.module.split("."))])
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "import_module":
                    if not node.args or not isinstance(node.args[0], ast.Constant):
                        raise ValueError(
                            "Dynamic import requires a reviewed explicit source closure"
                        )
                    modules = [node.args[0].value]
            for module in modules:
                if not isinstance(module, str) or not module.startswith("quantlab"):
                    continue
                stem = project / "src" / module.replace(".", "/")
                candidate = stem.with_suffix(".py")
                if not candidate.is_file():
                    candidate = stem / "__init__.py"
                if candidate.is_file():
                    pending.append(candidate.relative_to(project).as_posix())
    return {name: file_sha(project / name) for name in sorted(seen)}


@dataclass(frozen=True)
class Window:
    initial_signal: date
    first_execution: date
    terminal_close: date
    measured_sessions: tuple[date, ...]
    rebalance_signals: tuple[date, ...]
    calendar_sha256: str


def calendar_window(schedule, initial_signal, terminal_close):
    """Use exchange sessions, never the next observed source row."""
    if not isinstance(schedule, XNYSSchedule):
        raise ValueError("Actual XNYS schedule required")
    schedule.session(initial_signal)
    schedule.session(terminal_close)
    if initial_signal >= terminal_close:
        raise ValueError("Initial cash close must precede terminal valuation")
    first = schedule.next_session(initial_signal).session
    measured = tuple(d for d in schedule.sessions if first <= d <= terminal_close)
    if not measured or measured[0] != first or measured[-1] != terminal_close:
        raise ValueError("Complete scheduled measurement boundary required")
    # The schedule must extend beyond the endpoint to establish real month ends.
    schedule.next_session(terminal_close)
    signals = tuple(
        d
        for d in schedule.sessions
        if initial_signal <= d <= terminal_close
        and (d == initial_signal or schedule.next_session(d).session.month != d.month)
    )
    return Window(
        initial_signal,
        first,
        terminal_close,
        measured,
        signals,
        schedule.provenance["schedule_sha256"],
    )


@dataclass(frozen=True)
class LabelClock:
    asset_id: str
    signal_session: date
    available_at: datetime | None
    status: str
    source_sha256: str

    @property
    def key(self):
        return self.signal_session, self.asset_id


@dataclass(frozen=True)
class MatureKeys:
    keys: tuple[tuple[date, str], ...]
    sample_weights: tuple[float, ...]
    maximum_available_at: datetime | None
    excluded_status_counts: tuple[tuple[str, int], ...]


def mature_training_keys(clocks, *, signal_session, decision_at):
    """Select clock metadata before a provider is allowed to fetch label values."""
    _aware(decision_at)
    clocks = tuple(clocks)
    if any(not isinstance(row, LabelClock) for row in clocks):
        raise ValueError("Explicit label metadata required, not outcome values")
    if len({row.key for row in clocks}) != len(clocks):
        raise ValueError("Duplicate training label identity")
    selected, excluded = [], Counter()
    for row in sorted(clocks, key=lambda r: r.key):
        _sha(row.source_sha256)
        if row.signal_session >= signal_session:
            excluded["not_a_prior_signal"] += 1
        elif row.available_at is None or _aware(row.available_at) > decision_at:
            excluded["not_mature_at_fit"] += 1
        elif row.status != "valid":
            excluded[row.status] += 1
        else:
            selected.append(row)
    counts = Counter(row.signal_session for row in selected)
    n, months = len(selected), len(counts)
    weights = tuple(n / (months * counts[row.signal_session]) for row in selected)
    return MatureKeys(
        tuple(row.key for row in selected),
        weights,
        max((row.available_at for row in selected), default=None),
        tuple(sorted(excluded.items())),
    )


def risk_probe_clock(schedule, execution_session, *, source_endpoint):
    """Plan the unchanged 21-interval proxy without reading a future price."""
    start = schedule.sessions.index(execution_session)
    marks = schedule.sessions[start : start + 22]
    if len(marks) != 22:
        raise ValueError("Calendar must expose the full planned proxy horizon")
    mature = marks[-1] <= source_endpoint
    return {
        "closing_sessions": marks,
        "intervals": 21,
        "available_at": schedule.session(marks[-1]).decision_at,
        "status": "planned_within_source_endpoint" if mature else "right_censored",
        "outcome_values_read": False,
    }


@dataclass(frozen=True)
class FitReceipt:
    fit_id: str
    cutoff: datetime
    used_keys: tuple[tuple[date, str], ...]
    maximum_feature_available_at: datetime


@dataclass(frozen=True)
class EmptySourceMember:
    asset_id: str
    source_member: str
    source_sha256: str
    source_rows: int = 0


@dataclass(frozen=True)
class LoadedHistory:
    histories: Mapping
    source_identity: SourceIdentity
    source_empty_assets: tuple[str, ...] = ()
    source_empty_evidence: tuple[EmptySourceMember, ...] = ()


@dataclass(frozen=True)
class PortfolioNotExecuted:
    """No invented ledger when the frozen forecast itself lacks verified support."""

    initial_signal: date
    terminal_close: date
    unsupported_forecasts: tuple[tuple[date, str], ...]
    status: str = "not_executed_due_to_typed_forecast_support"
    full_period_performance_allowed: bool = False


@dataclass(frozen=True)
class Arm:
    arm_id: str
    component: str
    recipe_sha256: str
    load: Callable
    decide: Callable
    clocks: Callable | None = None
    fit: Callable | None = None


def economic_source_receipt(loaded):
    """Authenticate the sole permitted economic population exclusion."""
    empty = tuple(sorted(a for a, rows in loaded.histories.items() if not rows))
    if (
        empty != loaded.source_empty_assets
        or tuple(sorted(r.asset_id for r in loaded.source_empty_evidence)) != empty
        or any(r.source_rows != 0 for r in loaded.source_empty_evidence)
    ):
        raise ValueError("Exact authenticated globally empty source declarations required")
    for row in loaded.source_empty_evidence:
        if not row.source_member:
            raise ValueError("Empty source member identity required")
        _sha(row.source_sha256)
    return {
        "forecast_population_count": len(loaded.histories),
        "economic_history_count": len(loaded.histories) - len(empty),
        "omitted_globally_empty_assets": empty,
        "omitted_count": len(empty),
        "empty_source_evidence": loaded.source_empty_evidence,
        "source_identity": loaded.source_identity.as_dict(),
        "policy": "only_authenticated_global_source_rowcount_zero",
    }


def _execute(loaded, schedule, window, arm, policy, initial_cash, fee_bps):
    fits = []
    source_receipt = economic_source_receipt(loaded)
    empty = source_receipt["omitted_globally_empty_assets"]

    def decision(view, state):
        due = policy == "initial_only_mature" and view.session == window.initial_signal
        due |= policy == "monthly_expanding_mature" and view.session in window.rebalance_signals
        if due:
            selected = mature_training_keys(
                arm.clocks(view), signal_session=view.session, decision_at=view.cutoff
            )
            # The audited fit provider receives permitted keys, never all labels.
            receipt = arm.fit(view, selected)
            if (
                not isinstance(receipt, FitReceipt)
                or receipt.used_keys != selected.keys
                or receipt.cutoff != view.cutoff
                or _aware(receipt.maximum_feature_available_at) > view.cutoff
            ):
                raise ValueError("Refit receipt crossed the declared information boundary")
            _sha(receipt.fit_id)
            fits.append(receipt)
        value = arm.decide(view, state)
        if not isinstance(value, Decision):
            raise ValueError("Explicit HOLD or REBALANCE decision required")
        if value.target is not None and any(asset in empty for asset, _ in value.target):
            raise ValueError("Target mentions an authenticated globally empty source member")
        return value

    result = run_history(
        {a: rows for a, rows in loaded.histories.items() if rows},
        schedule,
        start=window.initial_signal,
        end=window.terminal_close,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=decision,
        source_identity=loaded.source_identity,
    )
    if result.daily:
        first = result.daily[0]
        if first.state.nav != initial_cash or first.execution.fee or first.state.values:
            raise ValueError("Initial signal close must contain cash only, with no inherited P&L")
    return result, tuple(fits)


def run_development_engineering_window(
    loaded,
    schedule,
    *,
    initial_signal,
    terminal_close,
    arm,
    initial_cash,
    fee_bps,
):
    """Real-development checks only; this entrypoint cannot expose the holdout."""
    if initial_signal >= date(2015, 1, 1) or terminal_close >= date(2015, 1, 1):
        raise ValueError("Engineering entrypoint refuses shared final period")
    if arm.clocks is not None or arm.fit is not None:
        raise ValueError("This engineering path performs no model fitting")
    return _execute(
        loaded,
        schedule,
        calendar_window(schedule, initial_signal, terminal_close),
        arm,
        "none",
        initial_cash,
        fee_bps,
    )[0]


def _authenticate(path, expected_sha256, project, inputs, registry, schedule):
    if file_sha(path) != _sha(expected_sha256):
        raise ValueError("Actual root release SHA differs")
    freeze = json.loads(Path(path).read_bytes())
    if (
        freeze.get("schema") != "shared_quant_holdout_release_v1"
        or freeze.get("phase") != "all_components_frozen_before_outcomes"
        or freeze.get("initial_signal") != START.isoformat()
        or freeze.get("terminal_close") != END.isoformat()
        or set(freeze.get("development_evidence", {})) != COMPONENTS
        or not freeze.get("comparisons")
        or not freeze.get("input_files")
        or freeze.get("cash_policy") != "explicit_zero_interest_assumption"
        or freeze.get("terminal_policy") != "mark_actual_positions_no_implicit_liquidation"
        or freeze.get("missing_mark_policy") != "preserve_incomplete_valid_prefix_no_gap_repair"
        or freeze.get("first_interval") != "include_initial_fee_and_first_open_to_close"
        or freeze.get("label_policy") != "prior_signals_and_available_at_no_later_than_fit"
        or freeze.get("training_sample_weight") != "equal_signal_date_mean_one"
    ):
        raise ValueError("Actual complete shared release required; development is not a freeze")
    approved = _aware(datetime.fromisoformat(freeze["approved_at"]))
    if approved > datetime.now(UTC):
        raise ValueError("Future-dated research approval")
    if freeze["historical_information_cutoff"] != schedule.session(START).decision_at.isoformat():
        raise ValueError("Historical information cutoff differs from initial signal")
    runtime = freeze["runtime"]
    actual = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in runtime["packages"]},
    }
    if actual != runtime:
        raise ValueError("Frozen runtime differs")
    sources = freeze["source_sha256"]
    own = Path(__file__).resolve().relative_to(Path(project).resolve()).as_posix()
    if own not in sources:
        raise ValueError("This release gate must itself be externally source-bound")
    for name, digest in sources.items():
        if file_sha(_bound(project, name)) != _sha(digest):
            raise ValueError("Frozen production source differs")
    for binding in freeze["input_files"].values():
        p = _bound(inputs, binding["path"])
        if p.stat().st_size != binding["bytes"] or file_sha(p) != _sha(binding["sha256"]):
            raise ValueError("Frozen input bytes differ before deserialization")
    for evidence in freeze["development_evidence"].values():
        if evidence["reviewed_terminal_evidence"] is not True:
            raise ValueError("Pending development comparisons cannot open final evaluation")
        support = evidence["economic_support"]
        if (
            support["classification"]
            not in {"supported_common_cohort", "unsupported_preserved_prefix", "not_applicable"}
            or not isinstance(support["original_status"], str)
            or not support["original_status"]
            or support["full_period_performance_allowed"]
            is not (support["classification"] == "supported_common_cohort")
        ):
            raise ValueError("Terminal review must not impersonate supported economic performance")
        if file_sha(_bound(inputs, evidence["path"])) != _sha(evidence["sha256"]):
            raise ValueError("Actual development evidence differs")
    declarations = freeze["portfolio_arms"] + freeze.get("forecast_arms", [])
    forecast_ids = {r["arm_id"] for r in freeze.get("forecast_arms", [])}
    if not declarations or len({r["arm_id"] for r in declarations}) != len(declarations):
        raise ValueError("Unique complete portfolio arm register required")
    if set(registry) != {row["arm_id"] for row in declarations}:
        raise ValueError("Missing or extra registered arms cannot silently complete")
    if not math.isfinite(freeze["initial_cash"]) or freeze["initial_cash"] <= 0:
        raise ValueError("Explicit positive initial cash required")
    entry_sources = [own]
    for row in declarations:
        arm = registry[row["arm_id"]]
        if (arm.arm_id, arm.component, arm.recipe_sha256) != (
            row["arm_id"],
            row["component"],
            _sha(row["recipe_sha256"]),
        ) or row["refit_policy"] not in REFITS:
            raise ValueError("Recipe or refit policy differs")
        if row["input_id"] not in freeze["input_files"]:
            raise ValueError("Every loader must name an authenticated input")
        recipe = _bound(project, row["recipe_path"])
        if (
            file_sha(recipe) != arm.recipe_sha256
            or sources.get(row["recipe_path"]) != arm.recipe_sha256
        ):
            raise ValueError("Actual frozen recipe bytes required, not just a matching label")
        if row["arm_id"] in forecast_ids:
            if row.get("fee_bps") is not None or row["refit_policy"] != "none":
                raise ValueError(
                    "Forecast-only declarations cannot request fees or supervised fits"
                )
        elif not math.isfinite(row["fee_bps"]) or row["fee_bps"] < 0:
            raise ValueError("Explicit finite nonnegative cost assumption required")
        identity = SourceIdentity(**row["source_identity"])
        if identity.calendar_sha256 != schedule.provenance["schedule_sha256"]:
            raise ValueError("Frozen source calendar differs")
        callbacks = [arm.load, arm.decide]
        if row["refit_policy"] != "none":
            if arm.clocks is None or arm.fit is None:
                raise ValueError("Registered mature-label refit providers required")
            callbacks += [arm.clocks, arm.fit]
        elif arm.clocks is not None or arm.fit is not None:
            raise ValueError("Undeclared fit provider forbidden")
        for callback in callbacks:
            source = Path(inspect.getsourcefile(callback)).resolve()
            relative = source.relative_to(Path(project).resolve()).as_posix()
            if relative not in sources or file_sha(source) != sources[relative]:
                raise ValueError("Every provider/decision implementation must be source-bound")
            entry_sources.append(relative)
    closure = source_closure(project, entry_sources)
    if any(sources.get(name) != digest for name, digest in closure.items()):
        raise ValueError("Complete transitive production source closure required")
    return freeze


def run_shared_holdout(
    release_path,
    expected_sha256,
    *,
    project,
    inputs,
    registry,
    schedule,
    dispatch_ticket_path,
    expected_dispatch_ticket_sha256,
):
    """Authenticate first, reserve once, then execute frozen portfolio arms only.

    No global scientific COMPLETE is emitted: prediction, risk-proxy and reporting
    consumers must separately satisfy all comparisons in the same actual freeze.
    """
    freeze = _authenticate(release_path, expected_sha256, project, inputs, registry, schedule)
    # The future publisher must exclusively consume the durable HOST ledger
    # entry before ONE cloud push. This local marker cannot police fresh VMs.
    # The externally pinned stage ticket binds that same reserved/consumed entry.
    if file_sha(dispatch_ticket_path) != _sha(expected_dispatch_ticket_sha256):
        raise ValueError("Actual externally bound dispatch ticket required")
    ticket = json.loads(Path(dispatch_ticket_path).read_bytes())
    release_id = freeze["release_id"]
    if (
        not isinstance(release_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", release_id) is None
        or ticket.get("schema") != "shared_quant_single_dispatch_ticket_v1"
        or ticket.get("release_id") != release_id
        or ticket.get("release_sha256") != expected_sha256
        or ticket.get("scope") != "one_shared_final_evaluation_no_automatic_retry"
    ):
        raise ValueError("Dispatch ticket is not bound to this shared frozen release")
    _sha(ticket["host_ledger_entry_sha256"])
    if platform.system() != "Linux" or not Path("/kaggle/working").is_dir():
        raise ValueError("Full-period scientific execution is cloud-only")
    marker = Path("/kaggle/working/shared-holdout") / release_id / "started.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(
            {"release_sha256": expected_sha256, "started_at": datetime.now(UTC).isoformat()}, stream
        )
    window = calendar_window(schedule, START, END)
    results, fits, histories = {}, {}, {}
    # All source-bound forecast/refit streams are durably exported by their
    # registered loaders before any portfolio can expose an economic outcome.
    for row in freeze["portfolio_arms"] + freeze.get("forecast_arms", []):
        arm = registry[row["arm_id"]]
        loaded = arm.load(row["input_id"], window)
        if not isinstance(loaded, LoadedHistory):
            raise ValueError("Explicit source-bound history provider required")
        if loaded.source_identity.as_dict() != row["source_identity"]:
            raise ValueError("Actual loaded source identity differs from the freeze")
        histories[arm.arm_id] = loaded
    from quantlab.evaluation.holdout_registry import prepare_registered_streams

    prepare_registered_streams(registry, histories, window)
    for row in freeze["portfolio_arms"]:
        arm = registry[row["arm_id"]]
        wrapper = arm.load.__self__
        missing = []
        for day, status in wrapper.stream_status.items():
            if status == "unsupported_forecast":
                item = wrapper.stream_exports[day]
                if file_sha(item["path"]) != item["sha256"]:
                    raise ValueError("Typed unsupported forecast receipt changed")
                record = json.loads(Path(item["path"]).read_bytes())
                from quantlab.evaluation.holdout_registry import ForecastSupportError

                if record["support_code"] not in ForecastSupportError.CODES:
                    raise ValueError("Unknown unsupported forecast cannot close a portfolio arm")
                missing.append((day, record["support_code"]))
        if missing:
            results[arm.arm_id] = PortfolioNotExecuted(START, END, tuple(missing))
            fits[arm.arm_id] = tuple(wrapper.fit_receipts.values())
            continue
        results[arm.arm_id], fits[arm.arm_id] = _execute(
            histories[arm.arm_id],
            schedule,
            window,
            arm,
            row["refit_policy"],
            freeze["initial_cash"],
            row["fee_bps"],
        )
    _authenticate(release_path, expected_sha256, project, inputs, registry, schedule)
    complete = all(
        isinstance(result, EngineResult)
        and result.failure is None
        and result.incomplete_day is None
        and tuple(r.session for r in result.daily[1:]) == window.measured_sessions
        for result in results.values()
    )
    return {
        "release_sha256": expected_sha256,
        "window": window,
        "portfolio_execution_support_complete": complete,
        "global_holdout_complete": False,
        "outcome_evaluation": "separate_after_frozen_predictions_not_run_here",
        "results": results,
        "fits": fits,
        "economic_source_receipts": {
            row["arm_id"]: economic_source_receipt(histories[row["arm_id"]])
            for row in freeze["portfolio_arms"]
        },
        "forecast_only_arms": tuple(r["arm_id"] for r in freeze.get("forecast_arms", [])),
    }
