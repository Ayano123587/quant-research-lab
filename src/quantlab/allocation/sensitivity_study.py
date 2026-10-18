"""Full common-cohort ETF one-change economics, restricted to guarded cloud runs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import time
import tomllib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from quantlab.allocation.policies import ASSET_IDS
from quantlab.allocation.sensitivity import SCENARIOS, sensitivity_target
from quantlab.allocation.study import _metrics
from quantlab.backtest.engine import Decision, run_history
from quantlab.data.provenance import identity_from_source
from quantlab.data.yahoo import YahooETFRelease


@contextmanager
def authenticated_histories(snapshot):
    """Read the actual Yahoo interface and re-authenticate after execution."""
    if not isinstance(snapshot, YahooETFRelease):
        raise ValueError("Authenticated Yahoo ETF release required")
    histories = {
        asset: snapshot.bars("etf", asset.split(":", 1)[1]) for asset in ASSET_IDS
    }
    try:
        yield histories
    finally:
        for asset in ASSET_IDS:
            snapshot.rows(asset)


def encoded(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda v: v.isoformat(),
        )
        + "\n"
    ).encode()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_closure():
    src = Path(__file__).resolve().parents[2]
    pending, found = [Path(__file__).resolve()], {}
    while pending:
        path = pending.pop()
        name = path.relative_to(src).as_posix()
        if name in found:
            continue
        found[name] = file_sha(path)
        for parent in path.parents:
            if parent == src:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer != path:
                pending.append(initializer)
        package = ".".join(path.relative_to(src).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(
                        (*parts[: len(parts) - node.level + 1], base)
                    ).strip(".")
                modules = [base, *(base + "." + a.name for a in node.names)]
            for module in modules:
                if module == "quantlab" or module.startswith("quantlab."):
                    target = src.joinpath(*module.split("."))
                    for candidate in (
                        target.with_suffix(".py"),
                        target / "__init__.py",
                    ):
                        if candidate.is_file():
                            pending.append(candidate)
    return dict(sorted(found.items()))


def runtime_versions():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "scipy",
                "scikit-learn",
                "exchange-calendars",
            )
        },
    }


def execute_scenario(
    histories,
    schedule,
    source_identity,
    scenario,
    *,
    start,
    end,
    initial_cash,
    fee_bps,
    budget_callback=None,
):
    """Small real-history engineering entrypoint, never selecting a scenario."""
    if scenario not in SCENARIOS or set(histories) != set(ASSET_IDS):
        raise ValueError("Exact declared scenario/eight actual sleeves required")
    if start >= date(2015, 1, 1) or end >= date(2015, 1, 1):
        raise ValueError("Shared final period is sealed")
    targets = []

    def decide(view, state):
        if budget_callback is not None:
            budget_callback(view, state)
        if (
            schedule.month_end(view.session.year, view.session.month).session
            != view.session
        ):
            return Decision(
                "HOLD", None, view.cutoff, view.next_open_at, "between_month_ends"
            )
        target = sensitivity_target(view, schedule, source_identity, scenario)
        targets.append(target)
        return target.decision

    result = run_history(
        histories,
        schedule,
        start=start,
        end=end,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=decide,
        source_identity=source_identity,
    )
    return result, tuple(targets)


def output_inventory(output):
    return {
        p.relative_to(output).as_posix(): {
            "sha256": file_sha(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(Path(output).rglob("*"))
        if p.is_file() and p.name not in ("manifest.json", "COMPLETE")
    }


def target_record(target):
    original = target.source_target
    identities = [
        (row.asset_id, str(row.session), row.source_sha256, row.row_sha256)
        for row in original.source_rows
    ]
    covariance = original.covariance
    return {
        "strategic_certificate": (
            asdict(target.strategic_certificate)
            if target.strategic_certificate is not None
            else None
        ),
        "scenario": asdict(target.scenario),
        "decision": asdict(target.decision),
        "strategic_weights": target.strategic_weights,
        "active": target.active,
        "forecast_before": target.forecast_before,
        "forecast_after": target.forecast_after,
        "scale": target.scale,
        "weights": target.weights,
        "cash_weight": target.cash_weight,
        "status": target.status,
        "max_input_available_at": original.max_input_available_at,
        "source_identity": original.source_identity.as_dict(),
        "input_rows": len(identities),
        "input_row_identity_sha256": hashlib.sha256(encoded(identities)).hexdigest(),
        "covariance_method": covariance.method if covariance else None,
        "covariance_configuration_hash": (
            covariance.configuration_hash if covariance else None
        ),
        "covariance_window_hash": covariance.window_hash if covariance else None,
    }


def run_sensitivity(
    snapshot, schedule, config_path, output, *, service_deadline_monotonic
):
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Full sensitivity study is cloud-only")
    if not math.isfinite(service_deadline_monotonic):
        raise ValueError("Startup-inclusive finite external deadline required")

    def budget():
        if time.monotonic() >= service_deadline_monotonic:
            raise TimeoutError("Sensitivity service deadline exhausted")

    budget()
    sources, runtime, config_sha = (
        source_closure(),
        runtime_versions(),
        file_sha(config_path),
    )
    config = tomllib.loads(Path(config_path).read_text())
    expected = {
        "schema_version": 1,
        "phase": "development_only_no_selection_or_holdout",
        "initial_signal": "2007-02-28",
        "end_close": "2014-12-31",
        "measured_sessions": 1975,
        "fee_bps": 10,
        "cash": "explicit_zero_interest_assumption",
        "comparison": "seven_one_change_scenarios_common_support_no_winner",
        "weighting": (
            "eight_strategic_sleeves_equal_or_declared_capped_method_then_trend_risk_no_leverage"
        ),
        "covariance_window": 63,
        "ewma_half_life": 63,
        "price_basis": "provider_adjusted_close_ratio_price_index_research_proxy",
        "terminal": "observed_close_no_liquidation_final_target_pending",
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact declared sensitivity semantics required")
    capital = config["initial_cash"]
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("Finite positive initial capital required")
    identity = identity_from_source(snapshot, schedule)
    if identity.price_basis != config["price_basis"]:
        raise ValueError("Actual reviewed Yahoo price index required")
    start, end = date(2007, 2, 28), date(2014, 12, 31)
    sessions = tuple(d for d in schedule.sessions if start < d <= end)
    if len(sessions) != 1975:
        raise ValueError("Exact common 1975 sessions required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "etf_sensitivity_development_v1",
        "complete": False,
        "source_identity": identity.as_dict(),
        "calendar": schedule.provenance,
        "source_closure_sha256": sources,
        "runtime": runtime,
        "config_sha256": config_sha,
        "scenarios": [asdict(s) for s in SCENARIOS],
        "paths": [],
        "metrics": None,
        "initial_signal": str(start),
        "first_measured": str(sessions[0]),
        "end_close": str(end),
        "fee_bps": 10,
        "initial_cash": capital,
        "holdout_evaluated": False,
        "winner_selected": False,
        "inverse_volatility_and_capped_risk_budget_comparison": True,
        "exact_erc_under_caps_claimed": False,
        "terminal": config["terminal"],
    }

    def postflight():
        if (
            source_closure() != sources
            or runtime_versions() != runtime
            or file_sha(config_path) != config_sha
        ):
            raise ValueError("Sensitivity source/runtime/config changed")
        manifest["artifacts"] = output_inventory(output)
        manifest["finished_at"] = datetime.now(UTC)
        if time.monotonic() >= service_deadline_monotonic:
            manifest.update(
                complete=False, status="INCOMPLETE_SERVICE_BUDGET", metrics=None
            )
        (output / "manifest.json").write_bytes(encoded(manifest))
        if manifest["complete"]:
            (output / "COMPLETE").write_text(file_sha(output / "manifest.json") + "\n")
        return manifest

    metrics, supported = {}, []
    with authenticated_histories(snapshot) as histories:
        for scenario in SCENARIOS:

            def check(view, state, scenario=scenario):
                if time.monotonic() >= service_deadline_monotonic:
                    (output / "interrupted-close.json").write_bytes(
                        encoded(
                            {
                                "scenario": scenario.scenario_id,
                                "session": view.session,
                                "state": asdict(state),
                                "resume_cursor": False,
                            }
                        )
                    )
                budget()

            try:
                budget()
                result, targets = execute_scenario(
                    histories,
                    schedule,
                    identity,
                    scenario,
                    start=start,
                    end=end,
                    initial_cash=capital,
                    fee_bps=10,
                    budget_callback=check,
                )
            except TimeoutError:
                manifest.update(
                    status="INCOMPLETE_SERVICE_BUDGET",
                    interrupted_scenario=scenario.scenario_id,
                )
                return postflight()
            records = result.daily[1:]
            if result.daily and (
                result.daily[0].session != start
                or result.daily[0].state.nav != capital
                or result.daily[0].execution.fee != 0
                or result.daily[0].state.values
            ):
                raise ValueError("Initial close must be cash only")
            complete = (
                result.failure is None
                and result.incomplete_day is None
                and tuple(r.session for r in records) == sessions
            )
            folder = output / scenario.scenario_id
            folder.mkdir()
            with (folder / "money-ledger.jsonl").open("xb") as stream:
                for row in result.daily:
                    stream.write(encoded(asdict(row)).replace(b"\n", b" ") + b"\n")
            (folder / "monthly-targets.json").write_bytes(
                encoded([target_record(t) for t in targets])
            )
            status = {
                "scenario": asdict(scenario),
                "economic_support": (
                    "COMPLETE" if complete else "UNSUPPORTED_VALID_PREFIX"
                ),
                "valid_measured_sessions": len(records),
                "failure": asdict(result.failure) if result.failure else None,
                "incomplete_opening": (
                    asdict(result.incomplete_day) if result.incomplete_day else None
                ),
                "pending_decision": asdict(result.pending) if result.pending else None,
                "endpoint_status": result.endpoint_status,
                "pending_status": result.pending_status,
            }
            (folder / "path-status.json").write_bytes(encoded(status))
            manifest["paths"].append(status)
            supported.append(complete)
            if complete:
                metrics[scenario.scenario_id] = _metrics(records, capital)
    complete = len(supported) == 7 and all(supported)
    manifest.update(
        complete=complete,
        status=(
            "complete_seven_common_paths" if complete else "unsupported_full_comparison"
        ),
        metrics=metrics if complete else None,
    )
    return postflight()
