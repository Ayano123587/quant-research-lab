"""Causal daily research ledger using unchanged source bars and scheduled events.

Opening OHLC marks are research proxies; adjustment and publication vintages
are not established by this archive. A callback is a trusted Python callable:
its arguments constrain information, not arbitrary closure/global access.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime

from quantlab.backtest.accounting import Execution, MarkedPortfolio, hold, mark_positions, rebalance
from quantlab.data.archive import Bar
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.provenance import SourceIdentity, resolve_source_identity


@dataclass(frozen=True)
class HistoricalView:
    histories: tuple[tuple[str, tuple[Bar, ...]], ...]
    session: date
    cutoff: datetime
    next_open_at: datetime

    def bars(self, asset_id):
        return dict(self.histories)[asset_id]


@dataclass(frozen=True)
class HoldingsState:
    valued_at: datetime
    values: tuple[tuple[str, float], ...]
    cash: float
    nav: float


@dataclass(frozen=True)
class Decision:
    action: str
    target: tuple[tuple[str, float], ...] | None
    signal_at: datetime
    intended_execution_at: datetime
    reason: str

    def __post_init__(self):
        target = self.target
        if hasattr(target, "items"):
            target = target.items()
        if target is not None:
            target = tuple(target)
            if any(isinstance(w, bool) for _, w in target):
                raise ValueError("Target weights must be numeric, not boolean")
            object.__setattr__(self, "target", tuple((a, float(w)) for a, w in target))


@dataclass(frozen=True)
class DailyRecord:
    session: date
    state: HoldingsState
    overnight: MarkedPortfolio
    execution: Execution
    intraday: MarkedPortfolio
    money_pnl: float
    reconciliation_residual: float


@dataclass(frozen=True)
class OpeningRecord:
    """Completed opening events when the subsequent close cannot be valued."""

    session: date
    state: HoldingsState
    overnight: MarkedPortfolio
    execution: Execution
    decision: Decision | None
    money_pnl: float
    reconciliation_residual: float


@dataclass(frozen=True)
class EngineFailure:
    session: date
    event: str
    reason: str
    unresolved_assets: tuple[str, ...]
    last_valid_state: HoldingsState | None


@dataclass(frozen=True)
class EngineResult:
    daily: tuple[DailyRecord, ...]
    decisions: tuple[Decision, ...]
    economic_support: tuple[date, ...]
    failure: EngineFailure | None
    endpoint: HoldingsState | None
    pending: Decision | None
    endpoint_status: str
    pending_status: str
    archive_sha256: str | None
    calendar_sha256: str
    incomplete_day: OpeningRecord | None = None
    source_identity: SourceIdentity | None = None


def run_history(
    histories,
    schedule,
    *,
    start,
    end,
    initial_cash,
    fee_bps,
    cash_policy,
    decision_callback,
    archive_sha256=None,
    source_identity=None,
):
    """Run cash-initialized sessions, stopping before the first unresolved interval.

    Each successful close produces one decision for the actual next scheduled
    open. Final decisions remain unexecuted and right-censored. Historical bars
    may be missing: unused missing assets do not erase held economic exposure.
    """
    if not isinstance(schedule, XNYSSchedule):
        raise ValueError("Actual XNYSSchedule required")
    if cash_policy != "explicit_zero_interest_assumption":
        raise ValueError("Declare the zero-interest cash convention explicitly")
    source_identity = resolve_source_identity(
        schedule.provenance["schedule_sha256"],
        source_identity=source_identity,
        archive_sha256=archive_sha256,
    )
    hold({}, initial_cash)
    rebalance({}, initial_cash, {}, fee_bps)
    if start > end or start not in schedule.sessions or end not in schedule.sessions:
        raise ValueError("Ordered included start/end sessions required")
    days = tuple(d for d in schedule.sessions if start <= d <= end)
    schedule.next_session(end)  # Even final decisions need a declared intended open.
    frozen = {}
    for asset, bars in histories.items():
        bars = tuple(bars)
        if not bars or any(not isinstance(b, Bar) or b.asset_id != asset for b in bars):
            raise ValueError("Nonempty identity-aligned source Bar histories required")
        if tuple(b.session for b in bars) != tuple(sorted({b.session for b in bars})):
            raise ValueError("Ordered unique source sessions required")
        frozen[asset] = bars
    if not frozen:
        raise ValueError("Source histories required")
    ordered_histories = tuple(
        (a, bars, tuple(b.session for b in bars))
        for a, bars in sorted(frozen.items())
    )
    indexed = {a: {b.session: b for b in bars} for a, bars in frozen.items()}
    values, cash, previous_marks = {}, float(initial_cash), {}
    daily, decisions, state, pending, failure = [], [], None, None, None
    incomplete_day = None

    def marks(day, assets, field):
        bad = tuple(
            sorted(
                a
                for a in assets
                if day not in indexed[a]
                # Daily OHLC/volume findings are not known at the open.
                or (field == "close" and indexed[a][day].quality_flags)
                or not math.isfinite(getattr(indexed[a][day], field))
                or getattr(indexed[a][day], field) <= 0
            )
        )
        return ({a: getattr(indexed[a][day], field) for a in assets if a not in bad}, bad)

    for day in days:
        timing = schedule.session(day)
        required = {a for a, v in values.items() if v > 0}
        if pending and pending.action == "REBALANCE":
            required |= {a for a, w in pending.target if w > 0}
        opens, bad_open = marks(day, required, "open")
        if bad_open:
            failure = EngineFailure(day, "open", "UNRESOLVED_SOURCE_MARK", bad_open, state)
            break
        overnight = mark_positions(values, cash, previous_marks, opens, 0.0)
        if pending is not None and pending.intended_execution_at != timing.open_at:
            raise ValueError("Pending execution differs from scheduled open")
        execution = (
            rebalance(overnight.positions, overnight.cash, dict(pending.target), fee_bps)
            if pending and pending.action == "REBALANCE"
            else hold(overnight.positions, overnight.cash)
        )
        opening_state = HoldingsState(
            timing.open_at,
            tuple(sorted(execution.positions.items())),
            execution.cash_after,
            execution.nav_after,
        )
        opening_pnl = execution.nav_after - overnight.nav_before
        opening_residual = opening_pnl - math.fsum(
            [*(v for _, v in overnight.asset_pnl), overnight.cash_pnl, -execution.fee]
        )
        if abs(opening_residual) > max(1.0, overnight.nav_before) * 4e-12:
            raise ValueError("Opening money reconciliation failed")
        # Only owners after the trade need this session's closing mark. Assets
        # sold at the open have no remaining intraday exposure.
        close_required = {a for a, v in execution.positions.items() if v > 0}
        closes, bad_close = marks(day, close_required, "close")
        if bad_close:
            incomplete_day = OpeningRecord(
                day, opening_state, overnight, execution, pending, opening_pnl, opening_residual
            )
            state = opening_state
            pending = None  # This target already executed; it must never be retried.
            failure = EngineFailure(day, "close", "UNRESOLVED_SOURCE_MARK", bad_close, state)
            break
        intraday = mark_positions(execution.positions, execution.cash_after, opens, closes, 0.0)
        pnl = intraday.nav_after - overnight.nav_before
        residual = pnl - math.fsum(
            [
                *(v for _, v in overnight.asset_pnl),
                overnight.cash_pnl,
                -execution.fee,
                *(v for _, v in intraday.asset_pnl),
                intraday.cash_pnl,
            ]
        )
        if abs(residual) > max(1.0, overnight.nav_before) * 4e-12:
            raise ValueError("Daily money reconciliation failed")
        state = HoldingsState(timing.close_at, intraday.values, intraday.cash, intraday.nav_after)
        daily.append(DailyRecord(day, state, overnight, execution, intraday, pnl, residual))
        values, cash, previous_marks = intraday.positions, intraday.cash, closes
        next_open = schedule.next_session(day).open_at
        view = HistoricalView(
            tuple(
                (a, bars[:bisect_right(sessions, day)])
                for a, bars, sessions in ordered_histories
            ),
            day,
            timing.decision_at,
            next_open,
        )
        decision = decision_callback(view, state)
        if not isinstance(decision, Decision):
            raise ValueError("Callback must return Decision")
        if decision.signal_at != view.cutoff or decision.intended_execution_at != next_open:
            raise ValueError("Decision must name exact current cutoff and next scheduled open")
        if decision.action not in {"HOLD", "REBALANCE"} or not isinstance(decision.reason, str):
            raise ValueError("Explicit action and reason required")
        if decision.action == "HOLD":
            if decision.target is not None:
                raise ValueError("HOLD cannot resubmit weights")
        else:
            if decision.target is None or len(dict(decision.target)) != len(decision.target):
                raise ValueError("Unique asset-keyed target required")
            if set(dict(decision.target)) - frozen.keys():
                raise ValueError("Unknown target source identity")
            rebalance({}, initial_cash, dict(decision.target), fee_bps)
        pending = decision
        decisions.append(decision)
    return EngineResult(
        tuple(daily),
        tuple(decisions),
        tuple(r.session for r in daily),
        failure,
        state,
        pending,
        (
            "incomplete_open_marked"
            if incomplete_day
            else "incomplete_valid_prefix"
            if failure
            else "requested_close_marked"
        ),
        (
            "executed_before_unresolved_close"
            if incomplete_day
            else "blocked_unexecuted"
            if failure
            else "right_censored_unexecuted"
        ),
        source_identity.archive_sha256,
        schedule.provenance["schedule_sha256"],
        incomplete_day,
        source_identity,
    )
