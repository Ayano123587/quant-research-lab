"""Training-only transforms of already mature observed stock returns."""

from __future__ import annotations

import math
from collections import defaultdict

RAW = "none_no_clipping_or_synthetic_replacement"
RANK = "within_mature_signal_date_average_rank_centered_v1"


def training_targets(batch, policy):
    """Keep raw outcomes immutable; rank only the rows admitted by the fit clock.

    For each training date use (average_rank - 0.5) / n - 0.5.
    Tied observed returns receive the same target. This changes the training
    objective; evaluation continues to use the original observed returns.
    """
    if policy not in (RAW, RANK):
        raise ValueError("Explicit registered training target policy required")
    groups = defaultdict(list)
    values = []
    keys = set()
    for index, (feature, outcome) in enumerate(
        zip(batch.feature_rows, batch.outcomes, strict=True)
    ):
        key = (feature.signal_session, feature.asset_id)
        if (
            key != (outcome.signal_session, outcome.asset_id)
            or key in keys
            or outcome.available_at > batch.information_cutoff
            or feature.signal_at > batch.information_cutoff
            or not math.isfinite(outcome.value)
        ):
            raise ValueError("Unique aligned mature observed training outcomes required")
        keys.add(key)
        values.append(outcome.value)
        groups[feature.signal_session].append(index)
    if not values:
        raise ValueError("Observed training rows required")
    if policy == RAW:
        return tuple(values)
    result = [None] * len(values)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: values[index])
        left = 0
        while left < len(ordered):
            right = left + 1
            while right < len(ordered) and values[ordered[right]] == values[ordered[left]]:
                right += 1
            # Zero-based tie positions [left,right) have average one-based rank
            # (left + right + 1) / 2.
            target = (left + right) / (2 * len(ordered)) - 0.5
            for index in ordered[left:right]:
                result[index] = target
            left = right
    return tuple(result)
