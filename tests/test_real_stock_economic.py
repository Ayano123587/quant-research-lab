"""Small unchanged real-history execution checks, not a performance study."""

import math
import os
import time
from datetime import date
from pathlib import Path

import pytest

from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import identity_from_source
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.economic_study import (
    authenticate_prediction_study,
    completion_budget_guard,
    economic_runtime,
    economic_source_closure,
    execute_candidate_window,
    read_prediction_signal,
    signal_decision,
)
from quantlab.stocks.prepare import SOURCE_MANIFEST_SHA256

pytestmark = pytest.mark.real_data


def test_actual_saved_scores_execute_fixed_target_and_fee_funded_ledger():
    private = Path(__file__).resolve().parents[2] / "quant-research-lab-private"
    study_root = Path(
        os.environ.get(
            "QUANTLAB_PREDICTION_STUDY",
            private
            / "stock-model-publication-v1/scientific-small-output-v1/stock-development/study",
        )
    )
    prepared = Path(
        os.environ.get("QUANTLAB_PREPARED", "data/processed/stock-monthly-v2")
    )
    source = Path(os.environ.get("QUANTLAB_SNAPSHOT", "data/processed/market-v1"))
    if not all((p / "COMPLETE").exists() for p in (study_root, prepared, source)):
        pytest.skip("Reviewed real prediction/prepared/market artifacts required")
    calendar = XNYSSchedule(date(2006, 1, 1), date(2015, 2, 1))
    study = authenticate_prediction_study(
        study_root,
        "b46c22e5a6f46330668f6f3870c0c8cc2ddde15c283d3f9af8637f93557492d6",
        calendar,
    )
    assert len(study.fit_records) == 1536 and len(study.signal_sessions) == 96
    day = date(2010, 1, 29)
    signals = read_prediction_signal(study, day, calendar, prepared)
    signal = signals["control_momentum_F01"]
    target = signal_decision(signal)
    assert len(target.target) == 50
    assert target.target == tuple(
        (a, 0.02)
        for a, _ in sorted(signal.scores, key=lambda pair: (-pair[1], pair[0]))[:50]
    )
    snapshot = MarketSnapshot(source, SOURCE_MANIFEST_SHA256)
    with snapshot.verified_read_session():
        histories = {
            a: snapshot.bars("stock", a.split(":", 1)[1]) for a, _ in target.target
        }
        result = execute_candidate_window(
            histories,
            calendar,
            {day: signal},
            start_signal=day,
            end=date(2010, 2, 3),
            initial_cash=100000.0,
            fee_bps=25,
            source_identity=identity_from_source(snapshot, calendar),
        )
        reached = []
        expired = time.monotonic() - 1

        def deadline(view, state):
            reached.append((view.session, state.nav))
            if time.monotonic() >= expired:
                raise TimeoutError(
                    "Actual reached decision stopped by expired software budget"
                )

        with pytest.raises(TimeoutError):
            execute_candidate_window(
                histories,
                calendar,
                {day: signal},
                start_signal=day,
                end=date(2010, 2, 3),
                initial_cash=100000.0,
                fee_bps=25,
                source_identity=identity_from_source(snapshot, calendar),
                budget_callback=deadline,
            )
        assert reached == [(day, 100000.0)]
    closure = economic_source_closure()
    assert {
        "quantlab/backtest/engine.py",
        "quantlab/backtest/accounting.py",
        "quantlab/data/snapshot.py",
        "quantlab/stocks/economic_study.py",
    } <= set(closure)
    assert economic_runtime()["python"]
    assert result.failure is None and result.incomplete_day is None
    assert tuple(r.session for r in result.daily) == (
        day,
        date(2010, 2, 1),
        date(2010, 2, 2),
        date(2010, 2, 3),
    )
    assert result.daily[0].state.nav == 100000.0
    assert result.daily[0].execution.fee == 0
    assert result.decisions[0] == target
    assert all(d.action == "HOLD" for d in result.decisions[1:])
    assert result.daily[1].execution.fee > 0
    assert all(r.execution.fee == 0 for r in result.daily[2:])
    assert all(
        math.isclose(
            r.state.nav,
            math.fsum(v for _, v in r.state.values) + r.state.cash,
            abs_tol=1e-8,
        )
        for r in result.daily
    )
    assert max(abs(r.reconciliation_residual) for r in result.daily) < 1e-7
    # Actual reached ledger stays available even when final software postflight
    # exhausts the service deadline; no completed research comparison is claimed.
    final_receipt = {
        "complete": True,
        "metrics": None,
        "paths": [{"actual_sessions": [r.session for r in result.daily]}],
    }
    assert completion_budget_guard(final_receipt, time.monotonic() - 1) is False
    assert final_receipt["status"] == "INCOMPLETE_SERVICE_BUDGET"
    assert final_receipt["metrics"] is None
    assert len(final_receipt["paths"][0]["actual_sessions"]) == 4
