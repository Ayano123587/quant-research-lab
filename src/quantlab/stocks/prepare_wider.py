"""In-memory wider-feature preparation prototype, never a published panel builder.

The caller supplies an authenticated original CrossSection and its upstream
prepared-manifest SHA (a source pin, never an asset count). This function neither
reselects the cohort nor reads labels, writes data or fits a model.
"""

import re
import struct
from dataclasses import dataclass

from quantlab.stocks.features import FEATURE_IDS as ORIGINAL_FEATURES
from quantlab.stocks.features import CrossSection, select_cross_section
from quantlab.stocks.wider_features import (
    BASIS_GATED,
    FEATURE_IDS,
    WiderCandidate,
    extend_candidate,
)

ARMS = (
    ("fresh_six_reference", ORIGINAL_FEATURES),
    (
        "add_five_union_eleven",
        ("F01", "F02", "F06", "F07", "F08", "F09", "F13", "F15", "F18", "F19", "F20"),
    ),
    (
        "documented_compact_ten",
        ("F01", "F02", "F06", "F07", "F08", "F09", "F13", "F15", "F18", "F20"),
    ),
)
ARMS = ARMS + (("all_supported_twenty_one", FEATURE_IDS[:21]),)
OPTIONAL_ADDITIONS = tuple(f for f in FEATURE_IDS[:21] if f not in ORIGINAL_FEATURES)
REGISTRY_STATUS = "prospective_source_only_requires_protocol_and_source_freeze_before_training"
BLUEPRINT_THREE_CONTROL = (("F01", 1 / 3), ("F07", 1 / 3), ("F09", -1 / 3))
ORIGINAL_FIVE_CONTROL = (("F01", 0.2), ("F07", 0.2), ("F09", -0.2), ("F13", 0.2), ("F15", 0.2))


@dataclass(frozen=True)
class WiderPreparedRow:
    asset_id: str
    liquidity_rank: int
    candidate: WiderCandidate
    normalized: tuple[tuple[str, float | None], ...]
    missing_indicators: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class ModelProjection:
    arm_id: str
    feature_ids: tuple[str, ...]
    column_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    missing_encoding: str = (
        "optional_model_rank_neutral_zero_plus_explicit_indicator_not_raw_imputation"
    )


@dataclass(frozen=True)
class WiderCrossSectionPrototype:
    original: CrossSection
    rows: tuple[WiderPreparedRow, ...]
    declared_parent_prepared_manifest_sha256: str | None
    original_six_rank_float64_identity: bool
    software_test_scope: bool
    fresh_reference_score_reproduction_verified: bool = False
    training_authorized: bool = False
    scope: str = "trusted_original_cross_section_in_memory_only_no_label_join_or_published_subset"
    original_five_factor_control: tuple = ORIGINAL_FIVE_CONTROL
    separate_blueprint_three_factor_control: tuple = BLUEPRINT_THREE_CONTROL

    def project(self, arm_id):
        arms = dict(ARMS)
        if arm_id not in arms:
            raise ValueError("Exactly one of four prospectively declared feature arms required")
        ids = arms[arm_id]
        optional = tuple(f for f in ids if f in OPTIONAL_ADDITIONS)
        columns = ids + tuple("missing_" + f for f in optional)
        values = []
        for row in self.rows:
            ranked, missing = dict(row.normalized), dict(row.missing_indicators)
            features = []
            for feature in ids:
                value = ranked[feature]
                if value is None:
                    if feature not in optional or not missing[feature]:
                        raise ValueError("Required/gated feature cannot be silently encoded")
                    value = 0.0
                features.append(value)
            features.extend(float(missing[f]) for f in optional)
            values.append(tuple(features))
        return ModelProjection(
            arm_id, ids, columns, tuple(r.asset_id for r in self.rows), tuple(values)
        )


def extend_cross_section(
    original, *, declared_parent_prepared_manifest_sha256=None, software_test_scope=False
):
    """Preserve the original full decision cohort and six exact rank values.

    Authentication of the supplied CrossSection and upstream manifest is the
    caller's boundary. A SHA-looking string is not represented as file validation.
    Real scoped software tests are not a released population or a research study.
    """
    if not isinstance(original, CrossSection) or not isinstance(software_test_scope, bool):
        raise ValueError("Frozen original CrossSection and explicit scope required")
    if software_test_scope:
        if declared_parent_prepared_manifest_sha256 is not None:
            raise ValueError("Scoped software check cannot claim a prepared-panel parent")
    elif not isinstance(declared_parent_prepared_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", declared_parent_prepared_manifest_sha256
    ):
        raise ValueError("Declared upstream prepared manifest SHA required")
    reconstructed = select_cross_section(original.candidates, top_n=original.top_n)
    if reconstructed != original or original.feature_ids != ORIGINAL_FEATURES:
        raise ValueError("Original selected cohort/ranks must reproduce exactly")
    ids = original.selected_asset_ids
    original_rows = {r.asset_id: r for r in original.rows}
    candidates = {c.asset_id: c for c in original.candidates}
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(original_rows)
        or any(a not in candidates or not candidates[a].eligible for a in ids)
    ):
        raise ValueError("Exact unchanged original selected cohort required")
    enriched = {a: extend_candidate(candidates[a]) for a in ids}
    normalized = {a: {} for a in ids}
    for feature in FEATURE_IDS:
        if feature in ORIGINAL_FEATURES:
            for a in ids:
                normalized[a][feature] = dict(original_rows[a].normalized)[feature]
            continue
        available = []
        for a in ids:
            value = next(v.value for v in enriched[a].values if v.feature_id == feature)
            normalized[a][feature] = None
            if value is not None:
                available.append((value, a))
        available.sort()
        n, left = len(available), 0
        while left < n:
            right = left + 1
            while right < n and available[right][0] == available[left][0]:
                right += 1
            average_rank = (left + 1 + right) / 2
            rank = (average_rank - 0.5) / n - 0.5 if n > 1 else 0.0
            for _, asset in available[left:right]:
                normalized[asset][feature] = rank
            left = right
    result = []
    for asset in sorted(ids):
        row = original_rows[asset]
        base = dict(row.normalized)
        for feature in ORIGINAL_FEATURES:
            assert struct.pack("<d", base[feature]) == struct.pack("<d", normalized[asset][feature])
        missing = tuple((v.feature_id, v.value is None) for v in enriched[asset].values)
        assert all(normalized[asset][feature] is None for feature in BASIS_GATED)
        result.append(
            WiderPreparedRow(
                asset,
                row.liquidity_rank,
                enriched[asset],
                tuple((f, normalized[asset][f]) for f in FEATURE_IDS),
                missing,
            )
        )
    return WiderCrossSectionPrototype(
        original, tuple(result), declared_parent_prepared_manifest_sha256, True, software_test_scope
    )
