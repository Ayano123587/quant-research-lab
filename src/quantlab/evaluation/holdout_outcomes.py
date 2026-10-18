"""Outcome-only registered comparison closure; never selects or refits recipes."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quantlab.evaluation.holdout import file_sha
from quantlab.risk.scoring import (
    build_future_proxy,
    paired_loss_difference,
    score_forecasts_outcomes_only,
)
from quantlab.stocks.development import date_ic


def read_prediction_exports(
    exports, *, expected_arms, expected_sessions, unsupported_forecasts=None
):
    """Authenticate exclusive prediction snapshots before opening label values."""
    if set(exports) != set(expected_arms):
        raise ValueError("Every frozen prediction arm export required")
    result = {}
    for arm in expected_arms:
        declaration = exports[arm]
        path = Path(declaration["path"])
        if (
            path.is_symlink()
            or path.stat().st_size != declaration["bytes"]
            or file_sha(path) != declaration["sha256"]
        ):
            raise ValueError("Actual prediction export bytes differ")
        payload = json.loads(path.read_bytes())
        if (
            payload["schema"] != "shared_holdout_stock_predictions_v1"
            or payload["contains_outcomes"] is not False
            or payload["candidate_id"] != declaration["candidate_id"]
        ):
            raise ValueError("Frozen prediction identity differs")
        for record in payload["predictions"]:
            day = date.fromisoformat(record["session"])
            key = arm, day
            if key in result or day not in expected_sessions or not record["fit_id"]:
                raise ValueError("Prediction date/fit receipt differs")
            result[key] = tuple((a, float(v)) for a, v in record["scores"])
    expected = {(a, d) for a in expected_arms for d in expected_sessions}
    unsupported = _support_codes(unsupported_forecasts, expected)
    if set(result) | set(unsupported) != expected:
        raise ValueError("Missing prediction dates cannot be silently omitted")
    return result


def _support_codes(records, expected):
    from quantlab.evaluation.holdout_registry import ForecastSupportError

    result = dict(records or {})
    if not set(result) <= set(expected) or any(
        v not in ForecastSupportError.CODES for v in result.values()
    ):
        raise ValueError(
            "Only authenticated known forecast-support records may close missing dates"
        )
    return result


def covariance_outcomes(
    forecasts_by_session,
    references_by_session,
    histories,
    schedule,
    *,
    expected_sessions,
    expected_estimators,
    registered_pairs,
    archive_sha256,
    archive_endpoint,
    scoring_cutoff,
    evaluation_plan_id,
    recipe_frozen_at,
    recipe_information_cutoff,
    unsupported_forecasts=None,
):
    """One identical q/proxy per scheduled date for every declared estimator.

    This consumer must run after frozen forecasts have been durably exported.
    Incomplete and immature outcomes remain explicit rows, never zero losses.
    """
    days = tuple(expected_sessions)
    names = tuple(expected_estimators)
    unsupported = _support_codes(unsupported_forecasts, days)
    if (
        not days
        or len(set(days)) != len(days)
        or not names
        or len(set(names)) != len(names)
        or set(forecasts_by_session) | set(unsupported) != set(days)
        or set(references_by_session) != set(forecasts_by_session)
    ):
        raise ValueError("Full registered forecast/reference cohort required")
    if any(a not in names or b not in names or a == b for a, b in registered_pairs):
        raise ValueError("Registered estimator pair differs")
    rows, differences = {}, {}
    for day in days:
        if day not in forecasts_by_session:
            rows[day] = {"status": "unsupported_forecast", "support_code": unsupported[day]}
            differences[day] = None
            continue
        estimates = forecasts_by_session[day]
        if set(estimates) != set(names):
            raise ValueError("Missing estimator cannot silently complete comparison")
        reference = references_by_session[day]
        proxy = build_future_proxy(
            {a: histories[a] for a in reference.asset_ids},
            schedule,
            reference,
            schedule.next_session(day).session,
            archive_sha256=archive_sha256,
            archive_endpoint=archive_endpoint,
        )
        scores = score_forecasts_outcomes_only(
            estimates,
            proxy,
            scoring_cutoff=scoring_cutoff,
            evaluation_plan_id=evaluation_plan_id,
            recipe_frozen_at=recipe_frozen_at,
            recipe_information_cutoff=recipe_information_cutoff,
        )
        indexed = {s.estimator_id: s for s in scores}
        if set(indexed) != set(names):
            raise ValueError("Scorer omitted registered estimator")
        rows[day] = tuple(indexed[n] for n in names)
        differences[day] = {
            (a, b, metric): paired_loss_difference(indexed[a], indexed[b], metric)
            for a, b in registered_pairs
            for metric in ("mse", "qlike")
        }
    return {
        "cohort_complete": True,
        "rows": rows,
        "paired_differences": differences,
        "selection_performed": False,
        "unsupported_forecasts": unsupported,
    }


def stock_outcomes(
    predictions,
    outcomes,
    *,
    expected_arms,
    expected_sessions,
    minimum_support,
    scoring_cutoff,
    outcome_status_by_session,
    unsupported_forecasts=None,
):
    """Join only already-exported predictions; preserve all date/arm support rows."""
    arms, days = tuple(expected_arms), tuple(expected_sessions)
    expected = {(a, d) for a in arms for d in days}
    unsupported = _support_codes(unsupported_forecasts, expected)
    if (
        not arms
        or not days
        or len(set(arms)) != len(arms)
        or len(set(days)) != len(days)
        or set(predictions) | set(unsupported) != expected
        or set(outcomes) != set(days)
        or set(outcome_status_by_session) != set(days)
    ):
        raise ValueError("Every frozen arm/date prediction and outcome status required")
    result = {}
    for day in days:
        status = outcome_status_by_session[day]
        if status not in {"complete", "right_censored", "incomplete_source"}:
            raise ValueError("Explicit source/maturity outcome status required")
        if status == "right_censored" and outcomes[day]:
            raise ValueError("Right-censored monthly outcomes cannot become partial returns")
        if any(
            row.signal_session != day or row.available_at > scoring_cutoff for row in outcomes[day]
        ):
            raise ValueError("Outcome values must belong to the dated mature scoring cohort")
        absent = tuple(arm for arm in arms if (arm, day) not in predictions)
        if absent:
            for arm in arms:
                result[arm, day] = {
                    "status": "unsupported_common_forecast",
                    "ic": None,
                    "missing_arms": absent,
                    "support_codes": {a: unsupported[a, day] for a in absent},
                }
            continue
        cohorts = [tuple(a for a, _ in predictions[arm, day]) for arm in arms]
        if any(c != cohorts[0] for c in cohorts):
            raise ValueError("Comparisons require the same full eligible ordered cohort")
        for arm in arms:
            result[arm, day] = date_ic(
                predictions[arm, day], outcomes[day], minimum_support=minimum_support
            )
    return {
        "cohort_complete": True,
        "per_arm_date": result,
        "outcome_status_by_session": dict(outcome_status_by_session),
        "selection_performed": False,
        "unsupported_forecasts": unsupported,
    }


def portfolio_comparison_support(results, *, expected_arms, expected_sessions):
    """Refuse aggregate performance until every economic path has common support."""
    if set(results) != set(expected_arms) or not expected_arms:
        raise ValueError("Missing registered portfolio arm")
    support = {}
    expected = tuple(expected_sessions)
    for name in expected_arms:
        result = results[name]
        from quantlab.evaluation.holdout import PortfolioNotExecuted

        if isinstance(result, PortfolioNotExecuted):
            support[name] = {
                "complete": False,
                "observed_sessions": (),
                "failure": result.status,
                "unsupported_forecasts": result.unsupported_forecasts,
                "endpoint_status": "no_economic_path_invented",
                "pending_status": None,
            }
            continue
        actual = tuple(d.session for d in result.daily)
        # Initial cash close is not a measured return interval.
        measured = actual[1:]
        support[name] = {
            "complete": result.failure is None and measured == expected,
            "observed_sessions": measured,
            "failure": result.failure,
            "endpoint_status": result.endpoint_status,
            "pending_status": result.pending_status,
        }
    complete = all(r["complete"] for r in support.values())
    return {
        "full_common_economic_support": complete,
        "support": support,
        "aggregate_performance_allowed": complete,
        "selection_performed": False,
    }


def portfolio_outcomes(results, *, expected_arms, expected_sessions, initial_cash):
    """Existing money-ledger metrics only on a complete common window."""
    support = portfolio_comparison_support(
        results, expected_arms=expected_arms, expected_sessions=expected_sessions
    )
    if not support["full_common_economic_support"]:
        return {**support, "metrics": None}
    from quantlab.allocation.study import _metrics

    metrics = {}
    for name in expected_arms:
        records = results[name].daily
        first = records[0]
        if (
            first.state.nav != initial_cash
            or first.state.cash != initial_cash
            or first.state.values
            or first.execution.fee
        ):
            raise ValueError("Every path must start with the same uninvested cash close")
        metrics[name] = _metrics(records[1:], initial_cash)
    return {
        **support,
        "metrics": metrics,
        "interpretation": "frozen comparisons only; no performance-driven selection",
    }
