"""Adapters on unchanged development records, without fits or final outcomes."""

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from quantlab.allocation.policies import ASSET_IDS, SYMBOLS, allocate
from quantlab.backtest.accounting import rebalance
from quantlab.backtest.engine import HistoricalView, HoldingsState
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity
from quantlab.evaluation.holdout import (
    Arm,
    LoadedHistory,
    file_sha,
    run_development_engineering_window,
)
from quantlab.evaluation.holdout_adapters import (
    CovarianceForecastProvider,
    ETFProvider,
    RiskPortfolioProvider,
    SensitivityProvider,
)
from quantlab.evaluation.holdout_outcomes import (
    covariance_outcomes,
    read_prediction_exports,
    stock_outcomes,
)
from quantlab.risk.scoring import ReferencePortfolio

pytestmark = pytest.mark.real_data
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual():
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", PROJECT / "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supply the unchanged real archive")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        histories = {
            rows[0].asset_id: rows for rows in (source.bars("etf", symbol) for symbol in SYMBOLS)
        }
    schedule = XNYSSchedule(date(2008, 1, 1), date(2010, 6, 1))
    identity = SourceIdentity(
        "original_archive",
        schedule.provenance["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    day = date(2010, 1, 29)
    view = HistoricalView(
        tuple((a, tuple(b for b in histories[a] if b.session <= day)) for a in ASSET_IDS),
        day,
        schedule.session(day).decision_at,
        schedule.next_session(day).open_at,
    )
    return histories, schedule, identity, view


def test_actual_etf_adapter_is_unchanged_control_and_holds_between_signals(actual):
    _, schedule, identity, view = actual
    provider = ETFProvider(
        schedule,
        identity,
        [view.session],
        policy="trend_252_volatility_63",
        covariance_method="sample_ddof1",
    )
    direct = allocate(
        view,
        schedule,
        policy="trend_252_volatility_63",
        source_identity=identity,
        covariance_method="sample_ddof1",
    )
    assert provider.decide(view, None) == direct.decision
    assert provider.targets[0].source_rows == direct.source_rows
    hold = ETFProvider(
        schedule, identity, [], policy="strategic_equal", covariance_method="sample_ddof1"
    )
    assert hold.decide(view, None).action == "HOLD"


def test_actual_covariance_all_estimators_share_same_proxy_and_right_censor(actual):
    histories, schedule, identity, view = actual
    parameters = {
        "min_observations": 63,
        "ewma_lambda": 2 ** (-1 / 63),
        "n_factors": 1,
        "residual_floor": 0,
        "constant_correlation_intensity": None,
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 1e-15,
    }
    provider = CovarianceForecastProvider(
        schedule,
        identity,
        observations=63,
        estimator_parameters={
            "sample": {**parameters, "method": "sample_ddof1"},
            "ewma": {**parameters, "method": "ewma"},
        },
    )
    reference = ReferencePortfolio(
        ASSET_IDS,
        (0.125,) * 8,
        view.cutoff,
        view.cutoff,
        "fixed_equal_eight_engineering",
        SUPPLIED_ARCHIVE_SHA256,
    )
    estimates = provider.forecast(view, reference)
    assert estimates["sample"].window_hash == estimates["ewma"].window_hash
    kwargs = dict(
        expected_sessions=(view.session,),
        expected_estimators=("sample", "ewma"),
        registered_pairs=(("sample", "ewma"),),
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        scoring_cutoff=schedule.session(date(2010, 3, 31)).decision_at,
        evaluation_plan_id="real_development_adapter_check",
        recipe_frozen_at=datetime.now(UTC),
        recipe_information_cutoff=view.cutoff,
    )
    result = covariance_outcomes(
        provider.forecasts,
        provider.references,
        histories,
        schedule,
        archive_endpoint=date(2010, 3, 31),
        **kwargs,
    )
    scores = result["rows"][view.session]
    assert scores[0].proxy_id == scores[1].proxy_id
    assert scores[0].reference_id == scores[1].reference_id
    assert scores[0].realized_variance == scores[1].realized_variance
    assert all(s.status == "scored" for s in scores)
    censored = covariance_outcomes(
        provider.forecasts,
        provider.references,
        histories,
        schedule,
        archive_endpoint=date(2010, 2, 3),
        **kwargs,
    )
    assert all(s.mse is None and s.qlike is None for s in censored["rows"][view.session])
    with pytest.raises(ValueError, match="immutable"):
        provider.forecast(view, reference)


def test_incomplete_registered_comparisons_refuse_before_any_outcome_calculation(actual):
    _, schedule, _, view = actual
    with pytest.raises(ValueError, match="Full registered"):
        covariance_outcomes(
            {},
            {},
            {},
            schedule,
            expected_sessions=(view.session,),
            expected_estimators=("sample",),
            registered_pairs=(),
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
            archive_endpoint=view.session,
            scoring_cutoff=view.cutoff,
            evaluation_plan_id="missing",
            recipe_frozen_at=datetime.now(UTC),
            recipe_information_cutoff=view.cutoff,
        )
    with pytest.raises(ValueError, match="Every frozen"):
        stock_outcomes(
            {},
            {},
            expected_arms=("frozen_unselected_recipe",),
            expected_sessions=(view.session,),
            minimum_support=30,
            scoring_cutoff=view.cutoff,
            outcome_status_by_session={},
        )


def test_actual_saved_sixteen_arm_predictions_export_without_label_reads(tmp_path):
    import pyarrow.parquet as pq

    default = (
        PROJECT.parent
        / "quant-research-lab-private/stock-model-publication-v1"
        / "scientific-small-output-v1/stock-development/study"
    )
    root = Path(os.environ.get("QUANTLAB_STOCK_DEVELOPMENT", default))
    if not root.exists():
        pytest.skip("Supply the authenticated completed development prediction study")
    assert (
        file_sha(root / "manifest.json")
        == "b46c22e5a6f46330668f6f3870c0c8cc2ddde15c283d3f9af8637f93557492d6"
    )
    manifest = json.loads((root / "manifest.json").read_bytes())
    relative = "predictions/2010-06-30.parquet"
    (entry,) = (a for a in manifest["artifacts"] if a["path"] == relative)
    source = root / relative
    assert source.stat().st_size == entry["bytes"] and file_sha(source) == entry["sha256"]
    records = pq.ParquetFile(source).read().to_pylist()
    arms = tuple(manifest["candidate_ids"])
    exports = {}
    for arm in arms:
        rows = sorted((r for r in records if r["candidate_id"] == arm), key=lambda r: r["asset_id"])
        assert len(rows) == 500
        path = tmp_path / (arm + ".json")
        payload = {
            "schema": "shared_holdout_stock_predictions_v1",
            "candidate_id": arm,
            "contains_outcomes": False,
            "predictions": [
                {
                    "session": "2010-06-30",
                    "fit_id": rows[0]["fit_id"],
                    "scores": [(r["asset_id"], r["score"]) for r in rows],
                }
            ],
        }
        path.write_text(json.dumps(payload))
        exports[arm] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha(path),
            "candidate_id": arm,
        }
    restored = read_prediction_exports(
        exports, expected_arms=arms, expected_sessions=(date(2010, 6, 30),)
    )
    assert len(restored) == 16 and all(len(rows) == 500 for rows in restored.values())
    # An absent outcome inventory cannot be mistaken for a completed evaluation.
    with pytest.raises(ValueError, match="Every frozen"):
        stock_outcomes(
            restored,
            {},
            expected_arms=arms,
            expected_sessions=(date(2010, 6, 30),),
            minimum_support=30,
            scoring_cutoff=records[0]["signal_at"],
            outcome_status_by_session={},
        )


def test_actual_risk_baseline_adapter_preserves_registered_projection_and_funding(actual):
    from quantlab.risk.economic_study import Arm as EconomicArm

    _, schedule, identity, view = actual
    parameters = {
        "method": "sample_ddof1",
        "min_observations": 63,
        "ewma_lambda": 2 ** (-1 / 63),
        "n_factors": 1,
        "residual_floor": 0,
        "constant_correlation_intensity": None,
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 1e-15,
    }
    forecasts = CovarianceForecastProvider(
        schedule, identity, observations=63, estimator_parameters={"sample_ddof1": parameters}
    )
    reference = ReferencePortfolio(
        ASSET_IDS,
        (0.125,) * 8,
        view.cutoff,
        view.cutoff,
        "fixed_equal_eight_engineering",
        SUPPLIED_ARCHIVE_SHA256,
    )
    policy = PROJECT / "configs/execution-projection.json"
    policy_sha = "7a732bd76731370b65c805fe8cef5c8e8eaaac49faeff9b641157951e4487685"
    assert file_sha(policy) == policy_sha
    provider = RiskPortfolioProvider(
        forecasts,
        {view.session: reference},
        economic_arm=EconomicArm("equal_weight", "equal_weight", "sample_ddof1", 0.0),
        cap=0.35,
        solver_policy=None,
        projection_path=policy,
        projection_sha256=policy_sha,
        sample_reference_id="sample_ddof1",
    )
    # An explicit cashinitialization assumption, not a generated source mark.
    state = HoldingsState(schedule.session(view.session).close_at, (), 100000.0, 100000.0)
    decision = provider.decide(view, state)
    assert decision.action == "REBALANCE"
    ledger = rebalance({}, state.cash, dict(decision.target), 10.0)
    assert ledger.cash_after >= 0
    assert ledger.nav_after + ledger.fee == pytest.approx(state.nav)
    audit = provider.decisions[0]["audit"]
    assert audit["projection"]["policy_sha256"] == policy_sha
    assert audit["projection"]["total_l1_adjustment"] <= 1e-8
    assert audit["raw_solution"]["method"] == "equal_weight"


def test_actual_all_seven_sensitivity_targets_and_funded_engine_parity(actual):
    from quantlab.allocation.sensitivity import SCENARIOS, sensitivity_target
    from quantlab.backtest.engine import Decision

    histories, schedule, identity, view = actual
    loaded = LoadedHistory(histories, identity)
    recipe_sha = file_sha(PROJECT / "configs/research.toml")
    for scenario in SCENARIOS:
        provider = SensitivityProvider(schedule, identity, [view.session], scenario=scenario)
        direct = sensitivity_target(view, schedule, identity, scenario)
        assert provider.decide(view, None) == direct.decision
        assert provider.targets[0].strategic_weights == direct.strategic_weights
        assert provider.targets[0].weights == direct.weights
        assert provider.targets[0].forecast_after == direct.forecast_after

        def direct_decide(current, state, scenario=scenario):
            if current.session == view.session:
                return sensitivity_target(current, schedule, identity, scenario).decision
            return Decision(
                "HOLD", None, current.cutoff, current.next_open_at, "between_frozen_signals"
            )

        settings = dict(
            initial_signal=view.session,
            terminal_close=date(2010, 2, 3),
            initial_cash=100000.0,
            fee_bps=10.0,
        )
        adapted = run_development_engineering_window(
            loaded,
            schedule,
            arm=Arm(scenario.scenario_id, "etf", recipe_sha, None, provider.decide),
            **settings,
        )
        original = run_development_engineering_window(
            loaded,
            schedule,
            arm=Arm(scenario.scenario_id, "etf", recipe_sha, None, direct_decide),
            **settings,
        )
        assert adapted == original
        assert adapted.failure is None
        assert adapted.daily[0].state.cash == 100000.0
        assert adapted.daily[0].execution.fee == 0
        assert adapted.daily[1].execution.fee > 0
        assert all(abs(row.reconciliation_residual) < 1e-7 for row in adapted.daily)
