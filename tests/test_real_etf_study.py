"""Short real 2010 ledger integration checks, never a study-performance result."""

import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path

import pytest

from quantlab.allocation.policies import ASSET_IDS, POLICIES, SYMBOLS
from quantlab.allocation.study import compare_controls, run_development
from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot

pytestmark = pytest.mark.real_data
MANIFEST_SHA256 = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"


@pytest.fixture(scope="module")
def observed():
    project = Path(__file__).resolve().parents[1]
    archive = Path(os.environ.get("QUANTLAB_ARCHIVE", str(project / "data/raw/archive.zip")))
    if not archive.is_file():
        pytest.skip("Authenticated actual archive required; no replacement prices")
    with MarketArchive(archive, SUPPLIED_ARCHIVE_SHA256) as source:
        histories = {a: source.bars("etf", s) for a, s in zip(ASSET_IDS, SYMBOLS, strict=True)}
    schedule = XNYSSchedule(date(2005, 1, 1), date(2018, 12, 31))
    results, targets = compare_controls(
        histories,
        schedule,
        start_signal=date(2010, 1, 29),
        end=date(2010, 2, 3),
        archive_sha256=SUPPLIED_ARCHIVE_SHA256,
    )
    return histories, schedule, results, targets


def test_same_real_sessions_and_cost_funded_first_execution(observed):
    histories, _, results, targets = observed
    expected = (date(2010, 2, 1), date(2010, 2, 2), date(2010, 2, 3))
    for policy in POLICIES:
        result, target = results[policy], targets[policy][0]
        assert tuple(x.session for x in result["measured"]) == expected
        weights = dict(target.decision.target)
        post_fee_nav = 100000 / (1 + 0.001 * math.fsum(weights.values()))
        indexed = {a: {b.session: b for b in bars} for a, bars in histories.items()}
        day = expected[0]
        expected_close_nav = post_fee_nav * (
            target.cash_weight
            + math.fsum(
                w * indexed[a][day].close / indexed[a][day].open for a, w in weights.items()
            )
        )
        first = result["measured"][0]
        assert math.isclose(first.execution.nav_after, post_fee_nav, abs_tol=1e-8)
        assert math.isclose(first.state.nav, expected_close_nav, abs_tol=1e-8)
        assert first.execution.fee > 0
        assert result["metrics"]["measured_sessions"] == 3
        assert all(r.execution.action == "HOLD" for r in result["measured"][1:])


def test_real_drift_and_terminal_mark_not_liquidation(observed):
    histories, _, results, _ = observed
    for result in results.values():
        first, _, last = result["measured"]
        first_values = dict(first.state.values)
        expected = first.state.cash
        for asset, value in first_values.items():
            prices = {b.session: b.close for b in histories[asset]}
            expected += value * prices[last.session] / prices[first.session]
        assert math.isclose(last.state.nav, expected, abs_tol=1e-8)
        assert last.execution.action == "HOLD" and last.execution.fee == 0
        assert math.isclose(result["metrics"]["net_return"], expected / 100000 - 1, abs_tol=1e-12)
        assert max(abs(r.reconciliation_residual) for r in result["measured"]) < 1e-8


def test_development_cannot_score_shared_final_period(observed):
    histories, schedule, _, _ = observed
    with pytest.raises(ValueError, match="shared final period"):
        compare_controls(
            histories,
            schedule,
            start_signal=date(2014, 12, 31),
            end=date(2015, 1, 2),
            archive_sha256=SUPPLIED_ARCHIVE_SHA256,
        )


def test_actual_snapshot_scoped_run_writes_bound_complete_outputs(observed, tmp_path):
    _, schedule, _, _ = observed
    project = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("QUANTLAB_SNAPSHOT", str(project / "data/processed/market-v1")))
    if not (root / "COMPLETE").is_file():
        pytest.skip("Completed authenticated actual normalization required")
    snapshot = MarketSnapshot(root, MANIFEST_SHA256)
    output = tmp_path / "actual-three-session-engine-check"
    result = run_development(
        snapshot, schedule, output, start_signal=date(2010, 1, 29), end=date(2010, 2, 3)
    )
    assert result["complete"] is True and not (output / "FAILURE.json").exists()
    raw = (output / "manifest.json").read_bytes()
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(raw).hexdigest()
    for name, expected in result["outputs_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics) == set(POLICIES)
    assert all(m["measured_sessions"] == 3 for m in metrics.values())
    with pytest.raises(FileExistsError):
        run_development(
            snapshot, schedule, output, start_signal=date(2010, 1, 29), end=date(2010, 2, 3)
        )
