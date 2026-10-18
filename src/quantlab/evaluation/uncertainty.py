"""Paired time-block uncertainty using observed returns only.

Resampling shares every date index across the candidate and reference. It does
not generate prices, refit a model, change holdings, or replay execution costs.
Intervals describe the supplied return history and do not correct for searching
many strategies. Inference assumes the sampled history is representative and
that a chosen block length captures the important serial dependence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np


def paired_ic_intervals(
    sessions, candidate_ic, *, reference, block_months=6, draws=2000, seed=20261017, confidence=0.95
):
    """Exploratory paired blocks of whole monthly cross-sectional rank ICs.

    All candidates share the same sampled month indices. Stocks within a month
    are never resampled as independent histories. This describes the supplied
    development results, including selected candidates; it does not adjust for
    the candidate search or establish out-of-sample superiority.
    """
    dates = tuple(sessions)
    names = tuple(sorted(candidate_ic))
    if (
        len(dates) < 3
        or any(type(day) is not date for day in dates)
        or any(a >= b for a, b in zip(dates, dates[1:], strict=False))
        or reference not in names
        or any(not isinstance(name, str) or not name for name in names)
        or type(block_months) is not int
        or not 1 <= block_months <= len(dates)
        or type(draws) is not int
        or draws < 100
        or type(seed) is not int
        or seed < 0
        or isinstance(confidence, bool)
        or not 0 < confidence < 1
    ):
        raise ValueError("Explicit aligned monthly IC cohort and resampling policy required")
    arrays = [np.asarray(candidate_ic[name]) for name in names]
    if any(values.shape != (len(dates),) or values.dtype.kind not in "fiu" for values in arrays):
        raise ValueError("Each candidate requires one actual IC per identical month")
    values = np.column_stack(arrays).astype(np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError("Only finite observed rank correlations in [-1,1] are accepted")
    baseline = names.index(reference)
    means = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty((draws, len(names)))
    offsets = np.arange(block_months)
    for draw in range(draws):
        starts = rng.integers(0, len(dates), size=(len(dates) + block_months - 1) // block_months)
        indices = ((starts[:, None] + offsets) % len(dates)).ravel()[: len(dates)]
        sampled_means[draw] = values[indices].mean(axis=0)
    differences = sampled_means - sampled_means[:, [baseline]]
    tail = (1 - confidence) / 2
    bounds = np.quantile(differences, [tail, 1 - tail], axis=0, method="linear")
    return {
        "schema": "paired_monthly_rank_IC_intervals_v1",
        "months": len(dates),
        "first_month": dates[0].isoformat(),
        "last_month": dates[-1].isoformat(),
        "reference": reference,
        "method": "paired_circular_contiguous_observed_month_blocks",
        "block_months": block_months,
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "selection_adjusted": False,
        "cross_section_resampled": False,
        "numpy_version": np.__version__,
        "input_sha256": _hash(
            {
                "dates": [day.isoformat() for day in dates],
                "candidates": names,
                "observed_ICs": values.tolist(),
            }
        ),
        "candidates": [
            {
                "candidate_id": name,
                "mean_monthly_IC": float(means[i]),
                "mean_IC_difference": float(means[i] - means[baseline]),
                "paired_difference_interval": [float(bounds[0, i]), float(bounds[1, i])],
            }
            for i, name in enumerate(names)
        ],
    }


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _aligned(sessions, candidate, reference):
    dates = tuple(sessions)
    if (
        len(dates) < 3
        or any(type(day) is not date for day in dates)
        or any(a >= b for a, b in zip(dates, dates[1:], strict=False))
    ):
        raise ValueError("Strictly increasing unique observed session dates required")
    left, right = np.asarray(candidate), np.asarray(reference)
    if (
        left.dtype.kind not in "fiu"
        or right.dtype.kind not in "fiu"
        or left.shape != (len(dates),)
        or right.shape != left.shape
    ):
        raise ValueError("One numeric observed return per matched session required")
    values = np.column_stack((left, right)).astype(np.float64)
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Finite simple returns greater than -1 required; no gap filling")
    return dates, values


def paired_block_intervals(
    sessions,
    candidate,
    reference,
    *,
    block_lengths=(21, 63, 126),
    draws=2000,
    seed=20261017,
    annual_sessions=252,
    confidence=0.95,
):
    """Return percentile intervals for annual mean-return and Sharpe differences.

    Each draw samples circular contiguous date blocks with replacement, trims
    to the original length, and uses the same sampled indices for both arms.
    Circular wrapping joins the end to the beginning as a bootstrap convention,
    not as a newly observed market interval. Returns already include the original
    ledger's fees. Annual mean return is arithmetic, distinct from CAGR.

    No dates are silently dropped. Undefined Sharpe draws are counted explicitly;
    a Sharpe interval is withheld if even one draw has zero sample volatility.
    Block-length sensitivity is reported in full, without selecting an interval.
    """
    dates, values = _aligned(sessions, candidate, reference)
    lengths = tuple(block_lengths)
    if (
        not lengths
        or len(set(lengths)) != len(lengths)
        or any(type(length) is not int or not 1 <= length <= len(dates) for length in lengths)
        or type(draws) is not int
        or draws < 100
        or type(seed) is not int
        or seed < 0
        or type(annual_sessions) is not int
        or annual_sessions <= 0
        or isinstance(confidence, bool)
        or not 0 < confidence < 1
    ):
        raise ValueError("Explicit valid block lengths, draws, seed and interval policy required")
    root_seed = np.random.SeedSequence(seed)
    # One independent stream per block length, unchanged when order is permuted.
    streams = {
        length: np.random.default_rng(
            np.random.SeedSequence(entropy=root_seed.entropy, spawn_key=(length,))
        )
        for length in lengths
    }
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    observed_sharpe = (
        float(
            np.sqrt(annual_sessions)
            * (mean[0] / standard_deviation[0] - mean[1] / standard_deviation[1])
        )
        if np.all(standard_deviation > 0)
        else None
    )
    tail = (1 - confidence) / 2

    def interval(samples):
        return [float(value) for value in np.quantile(samples, [tail, 1 - tail], method="linear")]

    output = {
        "schema": "paired_observed_return_block_intervals_v1",
        "method": "paired_circular_moving_block_percentile",
        "sessions": len(dates),
        "first_session": dates[0].isoformat(),
        "last_session": dates[-1].isoformat(),
        "input_sha256": _hash(
            {
                "dates": [day.isoformat() for day in dates],
                "candidate_reference_returns": values.tolist(),
            }
        ),
        "seed": seed,
        "draws": draws,
        "confidence": confidence,
        "annual_sessions": annual_sessions,
        "numpy_version": np.__version__,
        "observed_annual_arithmetic_mean_difference": float((mean[0] - mean[1]) * annual_sessions),
        "observed_annual_sharpe_difference": observed_sharpe,
        "selection_adjusted": False,
        "resampling_unit": "shared_observed_session_blocks_no_generated_market_data",
        "block_sensitivity": [],
    }
    offsets_by_length = {length: np.arange(length) for length in lengths}
    for length in lengths:
        rng = streams[length]
        mean_differences, sharpe_differences = [], []
        undefined = 0
        for _ in range(draws):
            starts = rng.integers(0, len(dates), size=(len(dates) + length - 1) // length)
            indices = ((starts[:, None] + offsets_by_length[length]) % len(dates)).ravel()
            sample = values[indices[: len(dates)]]
            sample_mean, sample_sd = sample.mean(axis=0), sample.std(axis=0, ddof=1)
            mean_differences.append(float((sample_mean[0] - sample_mean[1]) * annual_sessions))
            if np.all(sample_sd > 0):
                sharpe_differences.append(
                    float(
                        np.sqrt(annual_sessions)
                        * (sample_mean[0] / sample_sd[0] - sample_mean[1] / sample_sd[1])
                    )
                )
            else:
                undefined += 1
        output["block_sensitivity"].append(
            {
                "block_sessions": length,
                "annual_arithmetic_mean_difference_interval": interval(mean_differences),
                "annual_sharpe_difference_interval": interval(sharpe_differences)
                if not undefined
                else None,
                "undefined_sharpe_draws": undefined,
            }
        )
    return output
