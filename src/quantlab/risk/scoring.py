"""Outcome-only common-reference daily variance forecast diagnostics.

This module reads realized outcomes explicitly. It must never be used by a
historical decision callback, constructor or recipe selector. The future sample
variance is a noisy price-return proxy, not true latent conditional variance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime

from quantlab.data.archive import Bar
from quantlab.data.returns import ReturnInterval, close_intervals
from quantlab.risk.covariance import UNITS, CovarianceEstimate

HORIZON = 21
ROLE = "outcome_evaluation_only_not_a_strategy_or_selection_input"


def _aware(value):
    return (
        isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None
    )


def _json(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _identity(value):
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ReferencePortfolio:
    """Externally fixed q; timestamps attest its declared past-only information set."""

    asset_ids: tuple[str, ...]
    weights: tuple[float, ...]
    known_at: datetime
    max_input_available_at: datetime
    rule_id: str
    input_id: str

    def __post_init__(self):
        ids, raw = tuple(self.asset_ids), tuple(self.weights)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(a, str) or not a for a in ids)
            or len(ids) != len(raw)
        ):
            raise ValueError("Unique explicit asset order must match reference weights")
        if any(isinstance(w, bool) for w in raw):
            raise ValueError("Reference weights must be numeric, not boolean")
        weights = tuple(float(w) for w in raw)
        if any(not math.isfinite(w) or w < 0 for w in weights) or math.fsum(weights) > 1:
            raise ValueError("Finite long-only q with sum at most one required")
        if (
            not _aware(self.known_at)
            or not _aware(self.max_input_available_at)
            or self.max_input_available_at > self.known_at
            or not self.rule_id
            or not self.input_id
        ):
            raise ValueError(
                "Reference q requires past-only availability and rule/input identities"
            )
        object.__setattr__(self, "asset_ids", ids)
        object.__setattr__(self, "weights", weights)

    @property
    def reference_id(self):
        return _identity(asdict(self))


@dataclass(frozen=True)
class FutureVarianceProxy:
    reference: ReferencePortfolio
    execution_session: date
    execution_open_at: datetime
    start_at: datetime
    end_at: datetime | None
    available_at: datetime | None
    calendar_sessions: tuple[date, ...]
    intervals: tuple[tuple[ReturnInterval, ...], ...]
    portfolio_returns: tuple[float | None, ...]
    realized_variance: float | None
    status: str
    reasons: tuple[str, ...]
    archive_sha256: str
    archive_endpoint: date
    source_files: tuple[tuple[str, str, str], ...]
    calendar_provenance: tuple[tuple[str, str], ...]
    required_returns: int = HORIZON
    ddof: int = 1
    units: str = UNITS
    return_convention: str = "source_close_to_close_adjustment_unverified"
    cash_convention: str = "q cash remainder has explicit zero return in this common risk probe"

    @property
    def proxy_id(self):
        return _identity(asdict(self))

    @property
    def complete_returns(self):
        return sum(value is not None for value in self.portfolio_returns)

    def to_record(self):
        return _json(
            {
                **asdict(self),
                "proxy_id": self.proxy_id,
                "reference_id": self.reference.reference_id,
                "role": ROLE,
            }
        )


def build_future_proxy(
    histories, schedule, reference, execution_session, *, archive_sha256, archive_endpoint
):
    """Build exactly 21 future intervals starting at execution-session CLOSE.

    The preceding overnight/open-to-close move is intentionally excluded. A
    missing scheduled mark remains a missing interval; later rows never bridge
    it. Upstream archive authentication owns the supplied Bar provenance.
    """
    if not isinstance(reference, ReferencePortfolio):
        raise ValueError("Frozen ex-ante reference portfolio required")
    if not isinstance(histories, Mapping) or tuple(histories) != reference.asset_ids:
        raise ValueError("History asset order differs from the fixed reference")
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(c not in "0123456789abcdef" for c in archive_sha256)
        or archive_sha256 == "0" * 64
    ):
        raise ValueError("Actual archive SHA256 required")
    if type(archive_endpoint) is not date:
        raise ValueError("Verified archive endpoint session required")
    timing = schedule.session(execution_session)
    if reference.known_at >= timing.open_at:
        raise ValueError("Reference q must be fixed before the execution open")
    index = schedule.sessions.index(execution_session)
    days = tuple(schedule.sessions[index : index + HORIZON + 1])
    files, rows = [], {}
    for asset in reference.asset_ids:
        bars = tuple(histories[asset])
        if (
            not bars
            or any(not isinstance(b, Bar) or b.asset_id != asset for b in bars)
            or tuple(b.session for b in bars) != tuple(sorted({b.session for b in bars}))
        ):
            raise ValueError("Unchanged ordered single-asset actual Bar histories required")
        identities = {(b.source_member, b.source_sha256) for b in bars}
        if len(identities) != 1:
            raise ValueError("One immutable source member per asset required")
        member, source_sha = next(iter(identities))
        files.append((asset, member, source_sha))
        rows[asset] = bars
    provenance = tuple(sorted((str(k), str(v)) for k, v in schedule.provenance.items()))
    if len(days) < HORIZON + 1:
        return FutureVarianceProxy(
            reference,
            execution_session,
            timing.open_at,
            timing.close_at,
            None,
            None,
            days,
            (),
            (),
            None,
            "calendar_window_incomplete",
            ("Exactly 22 scheduled closing marks required; calendar ends early",),
            archive_sha256,
            archive_endpoint,
            tuple(files),
            provenance,
        )
    final = schedule.session(days[-1])
    intervals = tuple(
        close_intervals(rows[a], schedule, days[0], days[-1], archive_endpoint=archive_endpoint)
        for a in reference.asset_ids
    )
    if any(len(values) != HORIZON for values in intervals):
        raise ValueError("Scheduled future interval count differs")
    portfolio, reasons = [], set()
    for i in range(HORIZON):
        batch = [values[i] for values in intervals]
        if any(v.status != "valid" for v in batch):
            reasons.update(v.status for v in batch if v.status != "valid")
            portfolio.append(None)
        else:
            portfolio.append(
                math.fsum(
                    w * value.return_value
                    for w, value in zip(reference.weights, batch, strict=True)
                )
            )
    realized = None
    status = "complete"
    if reasons:
        status = "right_censored" if "right_censored" in reasons else "incomplete_source_window"
    else:
        mean = math.fsum(portfolio) / HORIZON
        realized = math.fsum((value - mean) ** 2 for value in portfolio) / (HORIZON - 1)
        if not math.isfinite(realized):
            raise ValueError("Future centered sample variance is nonfinite")
    return FutureVarianceProxy(
        reference,
        execution_session,
        timing.open_at,
        timing.close_at,
        final.close_at,
        final.decision_at,
        days,
        intervals,
        tuple(portfolio),
        realized,
        status,
        tuple(sorted(reasons)),
        archive_sha256,
        archive_endpoint,
        tuple(files),
        provenance,
    )


@dataclass(frozen=True)
class VarianceScore:
    estimator_id: str
    method: str
    status: str
    reason: str
    forecast_variance: float | None
    realized_variance: float | None
    mse: float | None
    qlike: float | None
    reference_id: str
    proxy_id: str
    asset_ids: tuple[str, ...]
    weights: tuple[float, ...]
    signal_at: datetime
    execution_open_at: datetime
    outcome_start_at: datetime
    outcome_end_at: datetime | None
    outcome_available_at: datetime | None
    scoring_information_cutoff: datetime
    recipe_frozen_at: datetime
    scoring_execution_at: datetime
    recipe_information_cutoff: datetime | None
    evaluation_plan_id: str
    forecast_window_hash: str
    forecast_configuration_hash: str
    archive_sha256: str
    observed_returns: int
    outcome_required_returns: int = HORIZON
    units: str = UNITS
    role: str = ROLE
    recipe_clock_scope: str = (
        "Actual research freeze/execution timestamps do not imply historical preregistration"
    )

    def to_record(self):
        return _json(asdict(self))


def score_forecasts_outcomes_only(
    estimates,
    proxy,
    *,
    scoring_cutoff,
    evaluation_plan_id,
    recipe_frozen_at,
    recipe_information_cutoff=None,
):
    """Score a frozen recipe; never return a candidate selection or change q.

    recipe_frozen_at is an actual research audit-wall-clock timestamp, checked
    against the actual scoring execution time. It never claims the recipe was
    published or preregistered in the historical market period. The optional
    recipe_information_cutoff is a separate historical selection/input boundary
    and must precede the affected forecast signal (a final-phase runner must
    require that boundary). scoring_cutoff is the historical outcome-maturity
    boundary. Incomplete/immature outcomes produce no shortened-window loss.
    """
    if (
        not isinstance(proxy, FutureVarianceProxy)
        or not isinstance(estimates, Mapping)
        or not estimates
        or not evaluation_plan_id
        or not _aware(scoring_cutoff)
        or not _aware(recipe_frozen_at)
    ):
        raise ValueError("Explicit frozen evaluation plan, cutoffs and common proxy required")
    scoring_execution_at = datetime.now(UTC)
    if recipe_frozen_at > scoring_execution_at:
        raise ValueError("Actual research recipe freeze cannot follow scoring execution")
    if recipe_information_cutoff is not None and not _aware(recipe_information_cutoff):
        raise ValueError("Historical recipe information cutoff must be timezone-aware")
    if proxy.units != UNITS or proxy.ddof != 1 or proxy.required_returns != HORIZON:
        raise ValueError("Common outcome must use daily variance units and exactly21 ddof1 returns")
    if proxy.status == "complete":
        if (
            len(proxy.calendar_sessions) != HORIZON + 1
            or len(proxy.portfolio_returns) != HORIZON
            or len(proxy.intervals) != len(proxy.reference.asset_ids)
            or any(
                len(rows) != HORIZON or any(r.status != "valid" for r in rows)
                for rows in proxy.intervals
            )
            or any(v is None or not math.isfinite(v) for v in proxy.portfolio_returns)
        ):
            raise ValueError("Complete common outcome requires all21 valid aligned intervals")
        mean = math.fsum(proxy.portfolio_returns) / HORIZON
        replayed = math.fsum((v - mean) ** 2 for v in proxy.portfolio_returns) / (HORIZON - 1)
        if replayed != proxy.realized_variance:
            raise ValueError("Future proxy differs from its centered ddof1 return variance")
    common_signal = None
    output = []
    for name, estimate in estimates.items():
        if not isinstance(name, str) or not name or not isinstance(estimate, CovarianceEstimate):
            raise ValueError("Named actual CovarianceEstimate objects required")
        if (
            estimate.asset_ids != proxy.reference.asset_ids
            or estimate.units != UNITS
            or estimate.status != "valid"
        ):
            raise ValueError("Forecast order, daily units or covariance status differs")
        if (
            not _aware(estimate.signal_at)
            or not _aware(estimate.max_available_at)
            or estimate.max_available_at > estimate.signal_at
            or estimate.training_end > estimate.max_available_at
            or proxy.reference.known_at > estimate.signal_at
            or (
                recipe_information_cutoff is not None
                and recipe_information_cutoff > estimate.signal_at
            )
            or estimate.signal_at >= proxy.execution_open_at
            or estimate.source_provenance.get("source_snapshot_id") != proxy.archive_sha256
            or estimate.source_provenance.get("return_convention") != proxy.return_convention
        ):
            raise ValueError(
                "Forecast/reference/recipe must use only decision-time available source inputs"
            )
        if common_signal is not None and estimate.signal_at != common_signal:
            raise ValueError("Paired estimators require the same forecast decision timestamp")
        common_signal = estimate.signal_at
        forecast = None
        realized = None
        mse = qlike = None
        status, reason = "scored", "Complete mature common-reference 21-session outcome"
        if proxy.status != "complete":
            status, reason = (
                "unscored_incomplete_outcome",
                proxy.status + ": " + ",".join(proxy.reasons),
            )
        elif proxy.available_at is None or scoring_cutoff < proxy.available_at:
            status, reason = (
                "unscored_not_mature",
                "Scoring cutoff precedes the final close availability convention",
            )
        else:
            forecast = estimate.portfolio_variance(
                proxy.reference.weights, proxy.reference.asset_ids
            )
            realized = proxy.realized_variance
            if not math.isfinite(forecast) or forecast <= 0:
                status, reason = (
                    "unscored_nonpositive_forecast",
                    "Positive finite variance required; no epsilon or clipping applied",
                )
            elif realized is None or not math.isfinite(realized) or realized < 0:
                raise ValueError(
                    "Complete outcome must contain a finite nonnegative daily variance"
                )
            else:
                mse = (forecast - realized) ** 2
                qlike = math.log(forecast) + realized / forecast
                if not math.isfinite(mse) or not math.isfinite(qlike):
                    status, reason = (
                        "unscored_nonfinite_loss",
                        "Loss arithmetic overflow; no clipping applied",
                    )
                    mse = qlike = None
        output.append(
            VarianceScore(
                name,
                estimate.method,
                status,
                reason,
                forecast,
                realized,
                mse,
                qlike,
                proxy.reference.reference_id,
                proxy.proxy_id,
                proxy.reference.asset_ids,
                proxy.reference.weights,
                estimate.signal_at,
                proxy.execution_open_at,
                proxy.start_at,
                proxy.end_at,
                proxy.available_at,
                scoring_cutoff,
                recipe_frozen_at,
                scoring_execution_at,
                recipe_information_cutoff,
                evaluation_plan_id,
                estimate.window_hash,
                estimate.configuration_hash,
                proxy.archive_sha256,
                proxy.complete_returns,
            )
        )
    return tuple(output)


def paired_loss_difference(left, right, metric):
    """One matched outcome difference, not evidence of statistical superiority."""
    if metric not in {"mse", "qlike"}:
        raise ValueError("Only prespecified MSE or QLIKE comparisons supported")
    keys = (
        "reference_id",
        "proxy_id",
        "asset_ids",
        "weights",
        "signal_at",
        "units",
        "evaluation_plan_id",
        "recipe_frozen_at",
        "recipe_information_cutoff",
        "scoring_information_cutoff",
    )
    if any(getattr(left, k) != getattr(right, k) for k in keys):
        raise ValueError(
            "Paired losses require identical reference/outcome/decision/evaluation support"
        )
    complete = left.status == right.status == "scored"
    return {
        "status": "paired" if complete else "unpaired_unscored",
        "metric": metric,
        "left_estimator": left.estimator_id,
        "right_estimator": right.estimator_id,
        "left_status": left.status,
        "right_status": right.status,
        "difference_left_minus_right": getattr(left, metric) - getattr(right, metric)
        if complete
        else None,
        "proxy_id": left.proxy_id,
        "reference_id": left.reference_id,
        "matched_outcomes": 1 if complete else 0,
        "role": ROLE,
        "inference_claim": "One-outcome arithmetic; no significance or selection claim",
    }
