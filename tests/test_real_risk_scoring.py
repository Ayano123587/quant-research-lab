"""Common-reference scoring checks derived only from unchanged real ETF records."""

import hashlib
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import variance

import numpy as np
import pytest

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.returns import trailing_returns
from quantlab.risk.covariance import ReturnWindow, estimate_covariance
from quantlab.risk.scoring import (
    ROLE,
    ReferencePortfolio,
    build_future_proxy,
    paired_loss_difference,
    score_forecasts_outcomes_only,
)

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def observed():
    path = Path(
        os.environ.get(
            "QUANTLAB_ARCHIVE", str(Path(__file__).resolve().parents[1] / "data/raw/archive.zip")
        )
    )
    if not path.is_file():
        pytest.skip("Actual supplied archive required; no replacement observations generated")
    schedule = XNYSSchedule(date(2006, 1, 1), date(2018, 1, 31))
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        symbols = ("SPY", "IEF", "EFA")
        histories = {
            rows[0].asset_id: rows for symbol in symbols if (rows := source.bars("etf", symbol))
        }
        bil = source.bars("etf", "BIL")
        endpoint = min(rows[-1].session for rows in histories.values())
        signal = schedule.month_end(2010, 1)
        execution = schedule.next_session(signal.session)
        intervals = [
            trailing_returns(rows, schedule, signal.session, 63, archive_endpoint=endpoint)
            for rows in histories.values()
        ]
        values = np.array([[col[i].return_value for col in intervals] for i in range(63)])
        times = intervals[0]
        row_sha = hashlib.sha256(
            repr([[v.start_row for v in col] + [col[-1].end_row] for col in intervals]).encode()
        ).hexdigest()
        window = ReturnWindow(
            values,
            tuple(histories),
            tuple(v.start_at for v in times),
            tuple(v.end_at for v in times),
            tuple(schedule.session(v.end_at.date()).decision_at for v in times),
            signal.decision_at,
            SUPPLIED_ARCHIVE_SHA256,
            "source_close_to_close_adjustment_unverified",
            "fixed_real_three_ETF_scoring_check",
            "all63scheduled_past_intervals_valid",
            {
                "source_rows_sha256": row_sha,
                "schedule_sha256": schedule.provenance["schedule_sha256"],
            },
        )
        reference = ReferencePortfolio(
            tuple(histories),
            (0.4, 0.3, 0.2),
            signal.decision_at,
            signal.decision_at,
            "externally_fixed_common_weights_with_cash",
            "fixed_weight_rule_not_future_selected",
        )
        proxy = build_future_proxy(
            histories,
            schedule,
            reference,
            execution.session,
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
            archive_endpoint=endpoint,
        )
        estimates = {
            "sample": estimate_covariance(window, "sample_ddof1"),
            "ewma": estimate_covariance(window, "ewma"),
        }
        yield dict(
            source=source,
            schedule=schedule,
            histories=histories,
            bil=bil,
            endpoint=endpoint,
            signal=signal,
            execution=execution,
            reference=reference,
            proxy=proxy,
            estimates=estimates,
            window=window,
            actual_recipe_frozen_at=datetime.now(UTC),
        )


def evaluate(data, proxy=None, cutoff=None):
    selected = proxy or data["proxy"]
    return score_forecasts_outcomes_only(
        data["estimates"],
        selected,
        scoring_cutoff=cutoff or selected.available_at,
        evaluation_plan_id="real_2010_software_reconciliation_not_benchmark",
        recipe_frozen_at=data["actual_recipe_frozen_at"],
        recipe_information_cutoff=data["window"].interval_start[0],
    )


def test_real21_returns_need22_exact_scheduled_closes(observed):
    d = observed
    proxy = d["proxy"]
    assert proxy.status == "complete"
    assert len(proxy.calendar_sessions) == 22
    assert len(proxy.portfolio_returns) == proxy.complete_returns == 21
    assert proxy.start_at == d["execution"].close_at
    assert proxy.available_at == d["schedule"].session(proxy.calendar_sessions[-1]).decision_at
    assert proxy.available_at > proxy.end_at
    actual = {a: {b.session: b for b in rows} for a, rows in d["histories"].items()}
    independent = []
    for start, end in zip(proxy.calendar_sessions, proxy.calendar_sessions[1:], strict=False):
        independent.append(
            math.fsum(
                w * (actual[a][end].close / actual[a][start].close - 1)
                for a, w in zip(proxy.reference.asset_ids, proxy.reference.weights, strict=True)
            )
        )
    assert proxy.portfolio_returns == pytest.approx(independent, rel=1e-14, abs=1e-18)
    assert proxy.realized_variance == pytest.approx(variance(independent), rel=1e-13)
    mean = math.fsum(independent) / 21
    assert proxy.realized_variance == pytest.approx(
        math.fsum((v - mean) ** 2 for v in independent) / 20
    )
    assert proxy.units == "decimal_return_squared_per_session" and proxy.ddof == 1
    for asset, intervals in zip(proxy.reference.asset_ids, proxy.intervals, strict=True):
        assert len(intervals) == 21
        for interval in intervals:
            assert interval.start_row[2] == actual[asset][interval.start_at.date()].row_sha256
            assert interval.end_row[2] == actual[asset][interval.end_at.date()].row_sha256
    record = proxy.to_record()
    assert record["role"] == ROLE and record["archive_sha256"] == SUPPLIED_ARCHIVE_SHA256


def test_real_sample_ewma_share_identical_q_outcome_and_exact_losses(observed):
    d = observed
    scores = evaluate(d)
    assert len({s.proxy_id for s in scores}) == len({s.reference_id for s in scores}) == 1
    for score in scores:
        estimate = d["estimates"][score.estimator_id]
        q = d["reference"].weights
        hand = math.fsum(
            q[i] * float(estimate.matrix[i, j]) * q[j] for i in range(len(q)) for j in range(len(q))
        )
        assert score.status == "scored"
        assert score.forecast_variance == pytest.approx(hand, rel=1e-13)
        assert score.realized_variance == d["proxy"].realized_variance
        assert score.mse == pytest.approx((hand - score.realized_variance) ** 2)
        assert score.qlike == pytest.approx(math.log(hand) + score.realized_variance / hand)
        assert score.role == ROLE
    pair = paired_loss_difference(scores[0], scores[1], "qlike")
    assert pair["matched_outcomes"] == 1
    assert pair["difference_left_minus_right"] == pytest.approx(scores[0].qlike - scores[1].qlike)


def test_real_future_variance_refuses_premature_maturity(observed):
    d = observed
    scores = evaluate(
        d, cutoff=d["proxy"].end_at
    )  # Actual close before its declared bar availability.
    assert all(s.status == "unscored_not_mature" for s in scores)
    assert all(s.mse is None and s.qlike is None and s.realized_variance is None for s in scores)
    assert paired_loss_difference(*scores, "mse")["matched_outcomes"] == 0


def test_real_asset_order_and_reference_chronology_refused(observed):
    d = observed
    with pytest.raises(ValueError, match="asset order"):
        build_future_proxy(
            dict(reversed(tuple(d["histories"].items()))),
            d["schedule"],
            d["reference"],
            d["execution"].session,
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
            archive_endpoint=d["endpoint"],
        )
    late = ReferencePortfolio(
        d["reference"].asset_ids,
        d["reference"].weights,
        d["execution"].open_at,
        d["signal"].decision_at,
        "fixed_but_too_late",
        "same_real_inputs",
    )
    with pytest.raises(ValueError, match="before the execution open"):
        build_future_proxy(
            d["histories"],
            d["schedule"],
            late,
            d["execution"].session,
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
            archive_endpoint=d["endpoint"],
        )


def test_real_inception_missing_window_not_gap_collapsed(observed):
    d = observed
    # BIL source genuinely starts during 2007, with no invented pre-inception prices.
    first = d["bil"][0].session
    start_index = d["schedule"].sessions.index(first) - 10
    execution = d["schedule"].session(d["schedule"].sessions[start_index])
    prior = d["schedule"].session(d["schedule"].sessions[start_index - 1])
    reference = ReferencePortfolio(
        (d["bil"][0].asset_id,),
        (1.0,),
        prior.decision_at,
        prior.decision_at,
        "fixed_single_BIL_missing_support_check",
        "actual_inception_metadata",
    )
    proxy = build_future_proxy(
        {reference.asset_ids[0]: d["bil"]},
        d["schedule"],
        reference,
        execution.session,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        archive_endpoint=d["endpoint"],
    )
    assert proxy.status == "incomplete_source_window" and "missing_mark" in proxy.reasons
    assert len(proxy.portfolio_returns) == 21 and proxy.complete_returns < 21
    assert proxy.realized_variance is None
    assert any(interval.start_row is None for interval in proxy.intervals[0])


def test_actual_archive_endpoint_is_right_censored_not_shortened(observed):
    d = observed
    execution = d["schedule"].session(d["endpoint"])
    previous = d["schedule"].session(
        d["schedule"].sessions[d["schedule"].sessions.index(d["endpoint"]) - 1]
    )
    reference = ReferencePortfolio(
        tuple(d["histories"]),
        d["reference"].weights,
        previous.decision_at,
        previous.decision_at,
        "fixed_common_reference_endpoint_check",
        "same_real_asset_files",
    )
    proxy = build_future_proxy(
        d["histories"],
        d["schedule"],
        reference,
        execution.session,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        archive_endpoint=d["endpoint"],
    )
    assert proxy.status == "right_censored" and len(proxy.portfolio_returns) == 21
    assert proxy.complete_returns == 0 and proxy.realized_variance is None


def test_real_all_cash_reference_no_qlike_epsilon(observed):
    d = observed
    reference = ReferencePortfolio(
        tuple(d["histories"]),
        (0.0, 0.0, 0.0),
        d["signal"].decision_at,
        d["signal"].decision_at,
        "explicit_unremunerated_cash_reference",
        "fixed_zero_risky_budget",
    )
    proxy = build_future_proxy(
        d["histories"],
        d["schedule"],
        reference,
        d["execution"].session,
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        archive_endpoint=d["endpoint"],
    )
    scores = evaluate(d, proxy)
    assert proxy.status == "complete" and proxy.realized_variance == 0
    assert all(
        s.status == "unscored_nonpositive_forecast" and s.forecast_variance == 0 for s in scores
    )
    assert all(s.qlike is None and s.mse is None for s in scores)
    with pytest.raises(ValueError, match="identical reference/outcome"):
        paired_loss_difference(evaluate(d)[0], scores[1], "qlike")


def test_actual_research_freeze_clock_is_separate_from_historical_inputs(observed):
    d = observed
    scores = evaluate(d)
    for score in scores:
        assert score.recipe_frozen_at == d["actual_recipe_frozen_at"]
        assert score.recipe_frozen_at > score.signal_at
        assert score.scoring_execution_at >= score.recipe_frozen_at
        assert score.recipe_information_cutoff <= score.signal_at
        assert "do not imply historical preregistration" in score.recipe_clock_scope
        assert score.status == "scored"
    with pytest.raises(ValueError, match="decision-time available"):
        score_forecasts_outcomes_only(
            d["estimates"],
            d["proxy"],
            scoring_cutoff=d["proxy"].available_at,
            evaluation_plan_id="real_historical_cutoff_guard",
            recipe_frozen_at=d["actual_recipe_frozen_at"],
            recipe_information_cutoff=d["execution"].open_at,
        )
