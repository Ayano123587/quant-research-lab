"""Additive readable exports from saved comparison values; no selection or refitting."""

from __future__ import annotations

import dataclasses
import math

REPORTING_POLICY = {
    "schema": "shared_quant_readable_tables_and_fixed_intervals_v1",
    "draws": 2000,
    "seed": 20261017,
    "confidence": 0.95,
    "covariance": {
        "source": "scripts/evaluate_risk_calibration.py",
        "reference": "sample_ddof1",
        "metric": "qlike",
        "include_reference_zero": True,
        "blocks": [3, 6, 12],
        "positions": "all_scheduled_months_censor_positions_retained",
        "mature_gap": "withhold_no_dropping_non_censored_unsupported_dates",
    },
    "stocks": {
        "blocks": [3, 6, 12],
        "aggregation": "equal_date_mean_IC",
        "mature_gap": "withhold_no_dropping_non_censored_unsupported_dates",
    },
    "portfolios": {"blocks": [21, 63, 126], "support": "full_common_window_only"},
    "selection_adjusted": False,
}


def comparison_tables(spec, result):
    """Project actual in-memory keys into explicit fields before JSON serialization."""
    name, kind = spec["comparison_id"], spec["kind"]
    if kind == "stock_prediction":
        return {
            "stock_date_metrics": [
                {
                    **record,
                    "comparison_id": name,
                    "arm_id": arm,
                    "signal_session": day,
                    "outcome_status": result["outcome_status_by_session"][day],
                }
                for (arm, day), record in result["per_arm_date"].items()
            ]
        }
    if kind == "covariance_proxy":
        scores, pairs = [], []
        for day, rows in result["rows"].items():
            if isinstance(rows, dict):
                scores.extend(
                    {
                        "comparison_id": name,
                        "signal_session": day,
                        "estimator_id": estimator,
                        "qlike": None,
                        "mse": None,
                        "forecast_variance": None,
                        "realized_variance": None,
                        **rows,
                    }
                    for estimator in spec["estimator_ids"]
                )
            else:
                scores.extend(
                    {"comparison_id": name, "signal_session": day, **dataclasses.asdict(row)}
                    for row in rows
                )
            differences = result["paired_differences"][day]
            if differences is None:
                pairs.extend(
                    {
                        "comparison_id": name,
                        "signal_session": day,
                        "estimator_a": a,
                        "estimator_b": b,
                        "metric": metric,
                        "difference": None,
                        "status": "unsupported_forecast",
                    }
                    for a, b in spec["pairs"]
                    for metric in ("mse", "qlike")
                )
            else:
                pairs.extend(
                    {
                        "comparison_id": name,
                        "signal_session": day,
                        "estimator_a": a,
                        "estimator_b": b,
                        "metric": metric,
                        **value,
                        "difference": value["difference_left_minus_right"],
                    }
                    for (a, b, metric), value in differences.items()
                )
        return {"covariance_scores": scores, "covariance_paired_losses": pairs}
    if kind == "portfolio":
        return {
            "portfolio_support_metrics": [
                {
                    "comparison_id": name,
                    "arm_id": arm,
                    **support,
                    "observed_session_count": len(support["observed_sessions"]),
                    "full_common_economic_support": result["full_common_economic_support"],
                    "metrics": None if result["metrics"] is None else result["metrics"][arm],
                }
                for arm, support in result["support"].items()
            ]
        }
    raise ValueError("Unknown actual comparison cannot produce report tables")


def daily_tables(results, expected_sessions):
    """Export actual prefixes with support flags, never manufacture missing money rows."""
    from quantlab.evaluation.holdout import PortfolioNotExecuted

    rows = []
    for arm, result in results.items():
        if isinstance(result, PortfolioNotExecuted):
            continue
        full = (
            result.failure is None
            and result.incomplete_day is None
            and tuple(r.session for r in result.daily[1:]) == tuple(expected_sessions)
        )
        previous = None
        for row in result.daily:
            initial = previous is None
            rows.append(
                {
                    "arm_id": arm,
                    "session": row.session,
                    "full_path_support": full,
                    "is_initial_cash_close": initial,
                    "nav": row.state.nav,
                    "cash": row.state.cash,
                    "money_pnl": row.money_pnl,
                    "fee": row.execution.fee,
                    "reconciliation_residual": row.reconciliation_residual,
                    "return_value": None if initial else row.state.nav / previous - 1,
                }
            )
            previous = row.state.nav
    return rows


def _mean_intervals(days, values, lengths):
    """Original risk calibration indices: one RNG, full scheduled positions, shared draws."""
    import numpy as np

    if (
        len(days) < max(lengths)
        or len(set(days)) != len(days)
        or tuple(days) != tuple(sorted(days))
        or not any(v is not None for v in values)
        or not all(math.isfinite(v) for v in values if v is not None)
    ):
        return None
    data = np.asarray(values, dtype=float)
    output = []
    rng = np.random.default_rng(REPORTING_POLICY["seed"])
    for length in lengths:
        starts = rng.integers(0, len(days), size=(2000, (len(days) + length - 1) // length))
        indices = ((starts[:, :, None] + np.arange(length)) % len(days)).reshape(2000, -1)[
            :, : len(days)
        ]
        samples = np.nanmean(data[indices], axis=1)
        low, high = np.quantile(samples, [0.025, 0.975], method="linear")
        output.append(
            {
                "block_length": length,
                "mean_difference": float(np.nanmean(data)),
                "lower": float(low),
                "upper": float(high),
            }
        )
    return output


def comparison_intervals(spec, result, results):
    """Outcome-only fixed registered contrasts; incomplete mature support withholds intervals."""
    from quantlab.evaluation.uncertainty import paired_block_intervals, paired_ic_intervals

    name, kind = spec["comparison_id"], spec["kind"]
    lengths = spec["analysis_policy"]["paired_block_lengths"]
    base = {
        "comparison_id": name,
        "kind": kind,
        "draws": 2000,
        "seed": 20261017,
        "confidence": 0.95,
        "selection_adjusted": False,
    }
    intervals = []
    if kind == "stock_prediction":
        days = sorted(
            d
            for d, status in result["outcome_status_by_session"].items()
            if status != "right_censored"
        )
        arms = spec["arm_ids"]
        valid = bool(days) and all(
            result["per_arm_date"][a, d]["status"] == "valid" for a in arms for d in days
        )
        if not valid or len(days) < max(lengths):
            return [
                {
                    **base,
                    "status": "withheld_incomplete_mature_paired_support",
                    "mature_dates": len(days),
                }
            ]
        values = {a: [result["per_arm_date"][a, d]["ic"] for d in days] for a in arms}
        reference = arms[0]
        for length in lengths:
            record = paired_ic_intervals(days, values, reference=reference, block_months=length)
            intervals.extend(
                {
                    **base,
                    "status": "scored",
                    "arm_id": r["candidate_id"],
                    "reference_arm": reference,
                    "metric": "mean_monthly_IC_difference",
                    "block_length": length,
                    "mean_difference": r["mean_IC_difference"],
                    "lower": r["paired_difference_interval"][0],
                    "upper": r["paired_difference_interval"][1],
                    "mature_dates": len(days),
                    "input_sha256": record["input_sha256"],
                }
                for r in record["candidates"]
            )
    elif kind == "covariance_proxy":
        censored = {
            d
            for d, rows in result["rows"].items()
            if isinstance(rows, tuple)
            and rows
            and all(
                r.status == "unscored_incomplete_outcome" and r.reason.startswith("right_censored:")
                for r in rows
            )
        }
        days = sorted(result["rows"])
        reference, metric = "sample_ddof1", "qlike"
        from quantlab.risk.scoring import paired_loss_difference

        for estimator in spec["estimator_ids"]:
            values, complete = [], True
            for day in days:
                rows = result["rows"][day]
                if day in censored:
                    values.append(None)
                    continue
                if not isinstance(rows, tuple) or any(row.status != "scored" for row in rows):
                    complete = False
                    values.append(None)
                    continue
                indexed = {row.estimator_id: row for row in rows}
                difference = paired_loss_difference(indexed[estimator], indexed[reference], metric)
                values.append(difference["difference_left_minus_right"])
            sampled = _mean_intervals(days, values, lengths) if complete else None
            common = {
                **base,
                "estimator_a": estimator,
                "estimator_b": reference,
                "metric": metric,
                "scheduled_dates": len(days),
                "mature_dates": len(days) - len(censored),
                "right_censored_dates": len(censored),
            }
            intervals.extend(
                [{**common, "status": "withheld_incomplete_mature_paired_support"}]
                if sampled is None
                else [{**common, "status": "scored", **r} for r in sampled]
            )
    elif kind == "portfolio":
        if not result["full_common_economic_support"]:
            return [{**base, "status": "withheld_incomplete_full_window_economic_support"}]
        reference = spec["arm_ids"][0]
        original = results[reference].daily
        days = [r.session for r in original[1:]]
        reference_values = [
            r.state.nav / p.state.nav - 1 for p, r in zip(original[:-1], original[1:], strict=True)
        ]
        for arm in spec["arm_ids"]:
            records = results[arm].daily
            values = [
                r.state.nav / p.state.nav - 1
                for p, r in zip(records[:-1], records[1:], strict=True)
            ]
            interval = paired_block_intervals(days, values, reference_values, block_lengths=lengths)
            for row in interval["block_sensitivity"]:
                for metric, observed, bounds in (
                    (
                        "annual_arithmetic_mean_difference",
                        interval["observed_annual_arithmetic_mean_difference"],
                        row["annual_arithmetic_mean_difference_interval"],
                    ),
                    (
                        "annual_sharpe_difference",
                        interval["observed_annual_sharpe_difference"],
                        row["annual_sharpe_difference_interval"],
                    ),
                ):
                    intervals.append(
                        {
                            **base,
                            "arm_id": arm,
                            "reference_arm": reference,
                            "metric": metric,
                            "block_length": row["block_sessions"],
                            "status": "scored"
                            if bounds is not None
                            else "withheld_undefined_sharpe_draws",
                            "mean_difference": observed,
                            "lower": None if bounds is None else bounds[0],
                            "upper": None if bounds is None else bounds[1],
                            "sessions": len(days),
                            "input_sha256": interval["input_sha256"],
                        }
                    )
    else:
        raise ValueError("Unknown registered interval comparison")
    return intervals
