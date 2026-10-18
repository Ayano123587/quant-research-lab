"""Real-source provenance and short accounting regressions, never a full study."""

import json
import os
from dataclasses import FrozenInstanceError, replace
from datetime import date
from pathlib import Path

import pytest

from quantlab.allocation.policies import allocate
from quantlab.allocation.study import run_development
from quantlab.backtest.engine import Decision, run_history
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import identity_from_source, resolve_source_identity
from quantlab.data.yahoo import YahooETFRelease
from quantlab.risk import study

pytestmark = pytest.mark.real_data
YAHOO_SHA = "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3"


@pytest.fixture(scope="module")
def source():
    root = Path(os.environ.get("QUANTLAB_YAHOO_RELEASE", "data/processed/yahoo-etf-price-index-v1"))
    if not (root / "COMPLETE").is_file():
        pytest.skip("Approved actual whole Yahoo release absent")
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    release = YahooETFRelease(root, YAHOO_SHA, schedule)
    identity = identity_from_source(release, schedule)
    histories = {asset: release.bars("etf", asset.split(":")[1]) for asset in release.asset_ids}
    return release, schedule, identity, histories


def test_frozen_identity_rejects_archive_alias_and_contradictory_provenance(source):
    release, schedule, identity, _ = source
    assert identity.source_type == "yahoo_etf_release" and identity.archive_sha256 is None
    assert identity.manifest_sha256 == release.manifest_sha256
    assert identity.raw_bundle_sha256 == release.source_bundle_sha256
    with pytest.raises(FrozenInstanceError):
        identity.archive_sha256 = SUPPLIED_ARCHIVE_SHA256
    with pytest.raises(ValueError, match="no archive SHA"):
        replace(identity, archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    with pytest.raises(ValueError, match="Contradictory"):
        resolve_source_identity(
            schedule.provenance["schedule_sha256"],
            source_identity=identity,
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        )
    legacy = resolve_source_identity(
        schedule.provenance["schedule_sha256"], archive_sha256=SUPPLIED_ARCHIVE_SHA256
    )
    assert legacy.archive_sha256 == SUPPLIED_ARCHIVE_SHA256 and legacy.raw_bundle_sha256 is None


def test_actual_targets_risk_windows_and_short_ledger_keep_yahoo_identity(source):
    _, schedule, identity, histories = source
    start, end = date(2010, 6, 30), date(2010, 7, 9)
    saved = []

    def decide(view, holdings):
        assert max(b.session for _, rows in view.histories for b in rows) <= view.session
        assert holdings.valued_at == schedule.session(view.session).close_at < view.cutoff
        if view.session == start:
            target = allocate(
                view, schedule, policy="trend_252_volatility_63", source_identity=identity
            )
            saved.append(target)
            return target.decision
        return Decision("HOLD", None, view.cutoff, view.next_open_at, "software_check_hold")

    result = run_history(
        histories,
        schedule,
        start=start,
        end=end,
        initial_cash=100000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=decide,
        source_identity=identity,
    )
    assert result.failure is None and result.source_identity == identity
    assert result.archive_sha256 is None and len(result.daily) == 7
    assert max(abs(row.reconciliation_residual) for row in result.daily) < 1e-7
    assert all(row.state.cash >= 0 for row in result.daily)
    target = saved[0]
    assert target.source_identity == identity and target.archive_sha256 is None
    assert target.covariance.source_provenance["raw_bundle_sha256"] == identity.raw_bundle_sha256
    assert "archive_sha256" not in target.covariance.source_provenance
    assert target.covariance.max_available_at == schedule.session(start).decision_at


def test_exact_missing_source_mark_still_stops_original_but_actual_yahoo_day_exists(source):
    _, schedule, identity, histories = source
    raw = Path(os.environ.get("QUANTLAB_ARCHIVE", "data/raw/archive.zip"))
    if not raw.is_file():
        pytest.skip("Actual original archive absent")
    with MarketArchive(raw, SUPPLIED_ARCHIVE_SHA256) as archive:
        original = archive.bars("etf", "SPY")
    asset = "etf:spy.us"
    start, end = date(2011, 2, 15), date(2011, 2, 18)

    def decide(view, _):
        return Decision(
            "REBALANCE",
            ((asset, 1.0),),
            view.cutoff,
            view.next_open_at,
            "actual_source_missing_day_engineering_check",
        )

    args = dict(
        start=start,
        end=end,
        initial_cash=10000,
        fee_bps=10,
        cash_policy="explicit_zero_interest_assumption",
        decision_callback=decide,
    )
    old = run_history({asset: original}, schedule, archive_sha256=SUPPLIED_ARCHIVE_SHA256, **args)
    assert old.failure.session == date(2011, 2, 17)
    assert old.daily[-1].session == date(2011, 2, 16)
    assert old.archive_sha256 == SUPPLIED_ARCHIVE_SHA256
    new = run_history({asset: histories[asset]}, schedule, source_identity=identity, **args)
    assert new.failure is None and new.daily[-1].session == end
    assert new.source_identity == identity and new.archive_sha256 is None


def test_actual_risk_panel_window_uses_yahoo_basis_without_original_archive_claim(source):
    _, schedule, identity, histories = source
    day = date(2010, 6, 30)
    position = schedule.sessions.index(day)
    days = set(schedule.sessions[position - 252 : position + 1])
    panel = tuple(
        study.asset_window(
            asset, tuple(b for b in bars if b.session in days), schedule, signal_session=day
        )[0]
        for asset, bars in histories.items()
    )
    window = study.return_window(panel, schedule, 63, signal_session=day, source_identity=identity)
    assert window.source_snapshot_id == YAHOO_SHA
    assert window.source_provenance["raw_bundle_sha256"] == identity.raw_bundle_sha256
    assert "archive_sha256" not in window.source_provenance
    assert window.asset_ids == tuple(histories)
    assert max(window.available_at) == schedule.session(day).decision_at
    assert SUPPLIED_ARCHIVE_SHA256 not in window.source_provenance.values()
    assert "price_index_research_proxy" in window.return_convention
    # Source metadata remains serializable without inventing a Yahoo archive.
    assert json.loads(json.dumps(identity.as_dict()))["archive_sha256"] is None


def test_seven_session_software_driver_receipts_use_explicit_provider_type(source, tmp_path):
    # Accounting/serialization check only, no estimator selection or published research subset.
    release, schedule, identity, _ = source
    output = tmp_path / "seven-session-software-receipts"
    receipt = run_development(
        release, schedule, output, start_signal=date(2010, 6, 30), end=date(2010, 7, 9)
    )
    assert receipt["complete"] is True
    assert receipt["contract"]["source_identity"] == identity.as_dict()
    assert receipt["contract"]["archive_sha256"] is None
    assert receipt["contract"]["source_manifest_sha256"] == YAHOO_SHA
    targets = json.loads((output / "volatility_63-monthly-targets.json").read_bytes())
    assert targets and all(row["source_identity"] == identity.as_dict() for row in targets)
