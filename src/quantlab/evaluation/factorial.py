"""Paired exposure-component contrasts on four complete observed return paths."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np

ARMS = ("strategic_equal", "trend_252", "volatility_63", "trend_252_volatility_63")
CONTRASTS = {
    "trend_without_scaler": (-1, 1, 0, 0),
    "scaler_without_trend": (-1, 0, 1, 0),
    "trend_with_scaler": (0, 0, -1, 1),
    "scaler_with_trend": (0, -1, 0, 1),
    "interaction": (1, -1, -1, 1),
}


def component_intervals(
    sessions, observed_returns, *, block_sessions=63, draws=2000, seed=20261017
):
    """Preserve the four-arm vector in every resampled observed date block.

    Contrasts of nonlinear Sharpe ratios are computed after each arm's ratio,
    never from the Sharpe ratio of a difference-return series. No new market
    prices, economic ledgers, model fits or execution events are generated.
    """
    dates = tuple(sessions)
    if (
        len(dates) < 3
        or set(observed_returns) != set(ARMS)
        or any(type(day) is not date for day in dates)
        or any(a >= b for a, b in zip(dates, dates[1:], strict=False))
        or type(block_sessions) is not int
        or not 1 <= block_sessions <= len(dates)
        or type(draws) is not int
        or draws < 100
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Explicit complete paired four-arm observed cohort required")
    arrays = [np.asarray(observed_returns[name]) for name in ARMS]
    if any(row.shape != (len(dates),) or row.dtype.kind not in "fiu" for row in arrays):
        raise ValueError("One observed return per arm per identical date required")
    values = np.column_stack(arrays).astype(float)
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Finite observed simple returns greater than -1 required")
    matrix = np.asarray(list(CONTRASTS.values()), dtype=float).T

    def statistics(sample):
        means, deviations = sample.mean(axis=0), sample.std(axis=0, ddof=1)
        if np.any(deviations == 0):
            return means * 252 @ matrix, None
        return means * 252 @ matrix, (means / deviations * np.sqrt(252)) @ matrix

    actual_mean, actual_sharpe = statistics(values)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(block_sessions,)))
    offsets = np.arange(block_sessions)
    means, sharpes, undefined = [], [], 0
    for _ in range(draws):
        starts = rng.integers(
            0, len(dates), size=(len(dates) + block_sessions - 1) // block_sessions
        )
        indices = ((starts[:, None] + offsets) % len(dates)).ravel()[: len(dates)]
        mean, sharpe = statistics(values[indices])
        means.append(mean)
        if sharpe is None:
            undefined += 1
        else:
            sharpes.append(sharpe)
    mean_intervals = np.quantile(means, [0.025, 0.975], axis=0)
    sharpe_intervals = None if undefined else np.quantile(sharpes, [0.025, 0.975], axis=0)
    payload = {
        "dates": [day.isoformat() for day in dates],
        "arms": ARMS,
        "observed_returns": values.tolist(),
    }
    return {
        "schema": "paired_four_policy_component_intervals_v1",
        "sessions": len(dates),
        "block_sessions": block_sessions,
        "draws": draws,
        "seed": seed,
        "selection_adjusted": False,
        "undefined_sharpe_draws": undefined,
        "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "contrasts": [
            {
                "name": name,
                "coefficients": dict(zip(ARMS, weights, strict=True)),
                "annual_arithmetic_mean_difference": float(actual_mean[index]),
                "annual_arithmetic_mean_95_interval": mean_intervals[:, index].tolist(),
                "annual_sharpe_difference": None
                if actual_sharpe is None
                else float(actual_sharpe[index]),
                "annual_sharpe_95_interval": None
                if sharpe_intervals is None
                else sharpe_intervals[:, index].tolist(),
            }
            for index, (name, weights) in enumerate(CONTRASTS.items())
        ],
    }
