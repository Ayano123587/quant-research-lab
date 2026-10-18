"""Prospective wider observed-source features; the frozen six-feature study is unchanged.

F22--F25 remain unavailable until a separate reviewed source-basis proof contract
exists. A neutral model encoding is never written into a missing raw observation.
"""

import math
from dataclasses import dataclass
from statistics import median, stdev

from quantlab.stocks.features import FeatureCandidate, candidate_from_history

FEATURE_IDS = tuple(f"F{i:02d}" for i in range(1, 26))
FORMULA_VERSION = "prospective_wider_observed_features_v1"
ZERO_DENOMINATOR_TOLERANCE = 0.0
VOLUME_CONVENTION = "observed_source_volume_units_unverified_adjustment_not_capacity"
BASIS_GATED = ("F22", "F23", "F24", "F25")


@dataclass(frozen=True)
class FeatureValue:
    feature_id: str
    value: float | None
    missing_reason: str | None
    units: str
    source_convention: str


@dataclass(frozen=True)
class WiderCandidate:
    original: FeatureCandidate
    values: tuple[FeatureValue, ...]
    formula_version: str = FORMULA_VERSION
    zero_denominator_tolerance: float = ZERO_DENOMINATOR_TOLERANCE
    basis_gated_features: tuple[str, ...] = BASIS_GATED


def wider_candidate_from_history(asset_id, bars, schedule, *, as_of, archive_sha256):
    """Trusted-history primitive; trim to the signal before readiness or formulas.

    Original exact253-session readiness, quality, row provenance and six formula
    results are reused verbatim. New optional denominator failures do not change
    the original historical population. No label or future coverage is consulted.
    """
    original = candidate_from_history(
        asset_id, bars, schedule, as_of=as_of, archive_sha256=archive_sha256
    )
    return extend_candidate(original)


def extend_candidate(original):
    """Enrich an actual frozen original candidate without changing its readiness."""
    if not isinstance(original, FeatureCandidate):
        raise ValueError("Frozen original source candidate required")
    if not original.eligible:
        return WiderCandidate(
            original,
            tuple(
                FeatureValue(
                    f, None, "ORIGINAL_REQUIRED_READINESS_FAILED", "unavailable", "original"
                )
                for f in FEATURE_IDS
            ),
        )
    if len(original.source_rows) != 253 or original.reasons:
        raise ValueError("Original exact253 valid scheduled source window required")
    rows = original.source_rows
    closes = tuple(b.close for b in rows)
    logs = tuple(math.log(b / a) for a, b in zip(closes[-64:-1], closes[-63:], strict=True))
    values = dict(original.features)
    reasons = {}
    values.update(
        {
            "F02": closes[-22] / closes[-127] - 1,
            "F03": closes[-6] / closes[-64] - 1,
            "F04": closes[-1] / closes[-64] - 1,
            "F05": (closes[-1] / closes[-64] - 1) - (closes[-64] / closes[-127] - 1),
            "F06": -(closes[-1] / closes[-6] - 1),
            "F08": math.sqrt(252) * stdev(logs[-21:]),
            "F10": math.sqrt(252 * math.fsum(min(x, 0.0) ** 2 for x in logs) / 63),
            "F14": closes[-1] / (math.fsum(closes[-252:]) / 252) - 1,
            "F16": closes[-1] / max(closes[-63:]) - 1,
        }
    )

    def ratio(feature, numerator, denominator):
        if denominator <= ZERO_DENOMINATOR_TOLERANCE:
            values[feature] = None
            reasons[feature] = "ZERO_DENOMINATOR"
        else:
            values[feature] = numerator / denominator - 1

    ratio("F11", values["F08"], values["F09"])
    average = math.fsum(logs) / 63
    centered = tuple(x - average for x in logs)
    m2 = math.fsum(x * x for x in centered) / 63
    if m2 <= ZERO_DENOMINATOR_TOLERANCE:
        values["F12"], reasons["F12"] = None, "ZERO_VARIANCE"
    else:
        values["F12"] = (math.fsum(x**3 for x in centered) / 63) / (m2**1.5)
    y = tuple(math.log(x) for x in closes[-63:])
    y_mean = math.fsum(y) / 63
    deviations = tuple(x - y_mean for x in y)
    sst = math.fsum(x * x for x in deviations)
    if sst <= ZERO_DENOMINATOR_TOLERANCE:
        values["F17"], reasons["F17"] = None, "CONSTANT_LOG_PRICE"
    else:
        x = tuple(j - 31 for j in range(63))
        slope = math.fsum(a * b for a, b in zip(x, deviations, strict=True)) / math.fsum(
            a * a for a in x
        )
        sse = math.fsum((b - slope * a) ** 2 for a, b in zip(x, deviations, strict=True))
        sign = 1 if slope > 0 else -1 if slope < 0 else 0
        values["F17"] = sign * (1 - sse / sst)
    volumes = tuple(b.volume for b in rows[-64:])
    if any(not math.isfinite(v) or v < 0 for v in volumes):
        for feature in ("F18", "F20", "F21"):
            values[feature], reasons[feature] = None, "UNUSABLE_OBSERVED_VOLUME"
    else:
        ratio("F18", float(median(volumes[-5:])), float(median(volumes[-63:])))
        ratio("F20", volumes[-1], float(median(volumes[:-1])))
        values["F21"] = sum(v == 0 for v in volumes[-63:]) / 63
    for feature in BASIS_GATED:
        values[feature], reasons[feature] = None, "VERIFIED_SOURCE_BASIS_PROOF_REQUIRED"
    output = []
    for feature in FEATURE_IDS:
        value = values[feature]
        if value is not None and not math.isfinite(value):
            value, reasons[feature] = None, "NONFINITE_FORMULA_RESULT"
        units = (
            "annualised_log_return_standard_deviation_252"
            if feature in ("F08", "F09")
            else "annualised_log_downside_semideviation_252"
            if feature == "F10"
            else "uncorrected_log_return_moment_skewness"
            if feature == "F12"
            else "signed_descriptive_log_price_OLS_R_squared"
            if feature == "F17"
            else "observed_zero_volume_fraction"
            if feature == "F21"
            else "log1p_source_value_proxy_divided_by_one_source_value_unit"
            if feature == "F19"
            else "unavailable_pending_basis_proof"
            if feature in BASIS_GATED
            else "dimensionless_observed_source_ratio"
        )
        convention = (
            VOLUME_CONVENTION
            if feature in ("F18", "F20", "F21")
            else (
                original.activity_convention if feature in ("F19", "F22") else original.price_basis
            )
        )
        output.append(FeatureValue(feature, value, reasons.get(feature), units, convention))
    return WiderCandidate(original, tuple(output))
