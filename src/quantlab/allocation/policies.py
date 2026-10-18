"""Fixed equal sleeves, positive trailing trend and capped volatility scaling.

These are research controls, not tuned strategies or profitability claims.
Source OHLC adjustment/publication vintages remain unverified. No missing
signal is interpreted as an inactive trend, and no sleeve is redistributed.
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from quantlab.backtest.engine import Decision, HistoricalView
from quantlab.data.archive import Bar
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity, resolve_source_identity
from quantlab.risk.covariance import CovarianceEstimate, ReturnWindow, estimate_covariance

SYMBOLS = ("SPY", "EFA", "EEM", "TLT", "LQD", "GLD", "DBC", "VNQ")
ASSET_IDS = tuple("etf:" + symbol.lower() + ".us" for symbol in SYMBOLS)
POLICIES = ("strategic_equal", "trend_252", "volatility_63", "trend_252_volatility_63")


@dataclass(frozen=True)
class AllocationTarget:
    policy: str
    asset_ids: tuple[str, ...]
    weights: tuple[float, ...]
    cash_weight: float
    strategic_budget: float
    active: tuple[bool, ...]
    trend_returns: tuple[float, ...] | None
    scale: float
    annual_volatility_before_scaling: float | None
    annual_volatility_after_scaling: float | None
    volatility_status: str
    covariance: CovarianceEstimate | None
    decision: Decision
    max_input_available_at: datetime
    source_rows: tuple[Bar, ...]
    archive_sha256: str | None
    calendar_sha256: str
    weight_basis: str = "postcost_NAV"
    cash_policy: str = "inactive_and_scaled_budgets_remain_cash"
    annualisation_factor: int = 252
    target_annual_volatility: float = 0.10
    source_identity: SourceIdentity | None = None


def _window_rows(view, schedule, count):
    index = schedule.sessions.index(view.session)
    if index < count:
        raise ValueError("INSUFFICIENT_CALENDAR_HISTORY: exact scheduled returns required")
    days = schedule.sessions[index - count : index + 1]
    available = dict(view.histories)
    if len(available) != len(view.histories):
        raise ValueError("Duplicate history asset IDs")
    rows = []
    for asset in ASSET_IDS:
        if asset not in available:
            raise ValueError("MISSING_SLEEVE_HISTORY: " + asset)
        bars = available[asset]
        if any(not isinstance(b, Bar) or b.asset_id != asset for b in bars):
            raise ValueError("Identity-aligned source bars required")
        if tuple(b.session for b in bars) != tuple(sorted({b.session for b in bars})):
            raise ValueError("Ordered unique historical source rows required")
        if any(b.session > view.session for b in bars):
            raise ValueError("FUTURE_SOURCE_ROW: restricted historical view required")
        indexed = {b.session: b for b in bars}
        selected = []
        for day in days:
            bar = indexed.get(day)
            if bar is None:
                raise ValueError("MISSING_SCHEDULED_MARK: " + asset + " " + str(day))
            if bar.quality_flags or not math.isfinite(bar.close) or bar.close <= 0:
                raise ValueError("UNUSABLE_SOURCE_MARK: " + asset + " " + str(day))
            selected.append(bar)
        if len({(b.source_member, b.source_sha256) for b in selected}) != 1:
            raise ValueError("Conflicting source vintages within asset")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", b.source_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", b.row_sha256)
            for b in selected
        ):
            raise ValueError("Source and row SHA256 provenance required")
        rows.append(tuple(selected))
    return days, tuple(rows)


def allocate(
    view,
    schedule,
    *,
    policy,
    archive_sha256=None,
    source_identity=None,
    covariance_method="sample_ddof1",
):
    """Construct one eight-sleeve target from a materialized past-only view.

    The calendar supplies only scheduled historical interval boundaries and the
    already-declared next open. All prices come exclusively from view.histories.
    Trend means close[t]/close[t-252]-1 > 0 after validating all 253 marks.
    Volatility is sqrt(252*w' S_daily*w), with scale min(1, 0.10/vol).
    """
    if not isinstance(view, HistoricalView) or not isinstance(schedule, XNYSSchedule):
        raise ValueError("Restricted HistoricalView and actual XNYSSchedule required")
    if policy not in POLICIES:
        raise ValueError("Unknown controlled ETF policy")
    source_identity = resolve_source_identity(
        schedule.provenance["schedule_sha256"],
        source_identity=source_identity,
        archive_sha256=archive_sha256,
    )
    timing = schedule.session(view.session)
    if (
        view.cutoff != timing.decision_at
        or view.next_open_at != schedule.next_session(view.session).open_at
    ):
        raise ValueError("View cutoff/execution differs from actual calendar")
    use_trend = policy in {"trend_252", "trend_252_volatility_63"}
    use_vol = policy in {"volatility_63", "trend_252_volatility_63"}
    count = 252 if use_trend else 63 if use_vol else 0
    days, rows = _window_rows(view, schedule, count)
    trend = tuple(b[-1].close / b[0].close - 1 for b in rows) if use_trend else None
    active = tuple(x > 0 for x in trend) if trend is not None else (True,) * len(ASSET_IDS)
    base = np.array([1 / len(ASSET_IDS) if enabled else 0 for enabled in active])
    covariance, before, after, scale = None, None, None, 1.0
    status = "not_applicable"
    if use_vol:
        risk_days = days[-64:]
        prices = np.array([[b.close for b in asset_rows[-64:]] for asset_rows in rows]).T
        provenance = source_identity.as_provenance()
        for asset, bars in zip(ASSET_IDS, rows, strict=True):
            hashes = {b.source_sha256 for b in bars}
            if len(hashes) != 1:
                raise ValueError("Conflicting source vintages within asset")
            provenance[asset] = next(iter(hashes))
        window = ReturnWindow(
            prices[1:] / prices[:-1] - 1,
            ASSET_IDS,
            tuple(schedule.session(d).close_at for d in risk_days[:-1]),
            tuple(schedule.session(d).close_at for d in risk_days[1:]),
            tuple(schedule.session(d).decision_at for d in risk_days[1:]),
            view.cutoff,
            source_identity.source_id,
            "exact_scheduled_close_to_close_decimal_simple",
            "all_eight_fixed_sleeves_complete_63_intervals",
            "strict_exact_scheduled_marks_no_fill",
            provenance,
        )
        covariance = estimate_covariance(window, method=covariance_method, min_observations=63)
        variance = covariance.portfolio_variance(base, ASSET_IDS)
        if not math.isfinite(variance) or variance < 0:
            raise ValueError("INVALID_PORTFOLIO_VARIANCE")
        before = math.sqrt(252 * variance)
        if before == 0:
            # No exposure needs no leverage; otherwise zero estimated risk
            # leaves the unscaled budget rather than dividing by zero.
            status = "all_cash" if not any(active) else "zero_variance_no_leverage"
        else:
            scale = min(1.0, 0.10 / before)
            status = "scaled_down" if scale < 1 else "unscaled_below_target"
        after = before * scale
    weights = tuple(float(w * scale) for w in base)
    cash = 1 - math.fsum(weights)
    decision = Decision(
        "REBALANCE",
        tuple(zip(ASSET_IDS, weights, strict=True)),
        view.cutoff,
        view.next_open_at,
        policy,
    )
    return AllocationTarget(
        policy,
        ASSET_IDS,
        weights,
        cash,
        1 / len(ASSET_IDS),
        active,
        trend,
        scale,
        before,
        after,
        status,
        covariance,
        decision,
        view.cutoff,
        tuple(b for asset_rows in rows for b in asset_rows),
        source_identity.archive_sha256,
        schedule.provenance["schedule_sha256"],
        source_identity=source_identity,
    )
