"""Opt-in, audited feature representation portability; never changes market data.

Exact comparison is the default. This prospective policy retains authenticated
parent raw values and recognizes only exact audited F09/F19 float64 pairs when
reproducing the frozen Windows preparation on Linux. Source rows, eligibility,
clocks, ranks, model inputs and saved-score gates remain caller-enforced EXACT.
"""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

SIX = ("F01", "F07", "F09", "F13", "F15", "F19")
PARENT_MANIFEST_SHA256 = "6273a9cff15b751d271859cc2e289640177bae57c43b12208bf6997ee65bf917"
INITIAL_DIAGNOSTIC_SHA256 = "9bda32a7c2237c4a3a09e4e195a7cc5c1052ce85d42015dfd46e9581da94694a"
DIAGNOSTIC_SHA256 = "b7a8bd63c4f0941e4fb8f05b9d54368941bfae9bfd505039e6bf4cb0fdf1a9a0"
WHITELIST_SHA256 = "0fc604514d7d725ef6e0d439cb0561023ad02a6bf31e9973a39e2f9b65681743"
WHITELIST_PATH = (
    Path(__file__).resolve().parents[3] / "configs/stock-feature-portability-whitelist.json"
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@lru_cache(maxsize=1)
def _whitelist():
    raw = WHITELIST_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WHITELIST_SHA256:
        raise ValueError("Exact audited whitelist bytes required")
    data = json.loads(raw)
    if (data["summary_sha256"], data["cohort_rows"], data["cohort_dates"]) != (
        DIAGNOSTIC_SHA256,
        71309,
        180,
    ):
        raise ValueError("Full audited original cohort required")
    result = {}
    for row in data["rows"]:
        key = (row["asset_id"], row["signal_session"])
        if key in result:
            raise ValueError("Duplicate whitelist identity")
        result[key] = row
    if len(result) != 894:
        raise ValueError("Full audited mismatch inventory required")
    return result


def require_full_acceptance_inventory(records):
    """No successful full preparation with missing or extra accepted drift."""
    expected = sorted(
        (asset, day, f, d["parent_float64_hex"], d["computed_float64_hex"])
        for (asset, day), row in _whitelist().items()
        for f, d in row["differences"].items()
    )
    actual = sorted(
        (
            r["asset_id"],
            str(r["signal_session"]),
            r["feature_id"],
            r["parent_float64_hex"],
            r["computed_float64_hex"],
        )
        for r in records
    )
    if actual != expected:
        raise ValueError("Accepted drift must equal all 896 audited field differences exactly")


POLICY = MappingProxyType(
    {
        "schema": "quant_feature_reproduction_policy_v1",
        "name": "exact_audited_f09_f19_windows_linux_parent_retained_v2",
        "parent_prepared_manifest_sha256": PARENT_MANIFEST_SHA256,
        "observed_cloud_diagnostic_sha256": DIAGNOSTIC_SHA256,
        "parent_platform": "Windows",
        "computed_platform": "Linux",
        "python": "3.12.15",
        "whitelist_sha256": WHITELIST_SHA256,
        "cohort_rows": 71309,
        "cohort_dates": 180,
        "acceptance": "exact_audited_row_source_clocks_and_float64_pair_f09_max2_f19_max1",
        "retained_values": "authenticated_parent_raw_six_and_rank_six_unchanged",
        "other_fields": "exact_no_tolerance",
        "fresh_six_prediction_gate": "original_b46_float64_bytes_and_IC_exact",
    }
)

POLICY_SHA256 = hashlib.sha256(
    json.dumps(dict(POLICY), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True)
class AcceptedRepresentationDifference:
    asset_id: str
    signal_session: date
    feature_id: str
    parent_float64_hex: str
    computed_float64_hex: str
    adjacent_float64_steps: int
    retained_basis: str
    policy_sha256: str
    diagnostic_sha256: str
    whitelist_sha256: str
    source_context_sha256: str


@dataclass(frozen=True)
class ReproducedFeatures:
    retained_parent_features: tuple[tuple[str, float], ...]
    accepted_differences: tuple[AcceptedRepresentationDifference, ...]


def reproduce_raw_six(
    parent_features,
    computed_features,
    *,
    asset_id,
    signal_session,
    parent_prepared_manifest_sha256,
    policy=None,
    parent_platform=None,
    computed_platform=None,
    parent_python=None,
    computed_python=None,
    source_context=None,
):
    """Compare observed feature values; retain parent bytes, never computed drift.

    The trusted caller must already authenticate the parent, actual source-row
    digest, eligibility and all three availability clocks. This helper neither
    authorizes a cloud run nor replaces those independent exact source gates.
    Platform/runtime declarations must come from the frozen run's provenance.
    """
    parent_features, computed_features = tuple(parent_features), tuple(computed_features)
    if tuple(k for k, _ in parent_features) != SIX or tuple(k for k, _ in computed_features) != SIX:
        raise ValueError("Exact ordered original six feature identities required")
    if not isinstance(asset_id, str) or not asset_id.startswith("stock:"):
        raise ValueError("Original source-scoped stock identity required")
    if not isinstance(signal_session, date):
        raise ValueError("Actual signal session date required")
    if parent_prepared_manifest_sha256 != PARENT_MANIFEST_SHA256:
        raise ValueError("Authenticated original parent manifest required")
    if policy is not None:
        if policy != POLICY:
            raise ValueError(
                "Only the explicitly frozen prospective exact-whitelist policy is supported"
            )
        if (parent_platform, computed_platform, parent_python, computed_python) != (
            "Windows",
            "Linux",
            "3.12.15",
            "3.12.15",
        ):
            raise ValueError("Frozen Windows-to-Linux runtime provenance required")
    differences = []
    for (feature, parent), (_, computed) in zip(parent_features, computed_features, strict=True):
        if not isinstance(parent, float) or not isinstance(computed, float):
            raise ValueError("Observed float64 feature values required")
        if not math.isfinite(parent) or not math.isfinite(computed):
            raise ValueError("Finite observed features required")
        parent_bits, computed_bits = struct.pack("<d", parent), struct.pack("<d", computed)
        if parent_bits == computed_bits:
            continue
        entry = _whitelist().get((asset_id, str(signal_session))) if policy is not None else None
        observed_context = (
            None
            if source_context is None
            else {
                "source_sha256": source_context["source_sha256"],
                "input_rows_sha256": source_context["input_rows_sha256"],
                "clocks": source_context["clocks"],
            }
        )
        expected_context = (
            None
            if entry is None
            else {k: entry[k] for k in ("source_sha256", "input_rows_sha256", "clocks")}
        )
        audited = None if entry is None else entry["differences"].get(feature)
        steps = None if audited is None else audited["ulp_distance"]
        if not (
            audited is not None
            and _canonical(observed_context) == _canonical(expected_context)
            and feature in ("F09", "F19")
            and parent > 0
            and computed > 0
            and 1 <= steps <= (2 if feature == "F09" else 1)
            and parent_bits.hex() == audited["parent_float64_hex"]
            and computed_bits.hex() == audited["computed_float64_hex"]
            and math.nextafter(parent, computed, steps=steps) == computed
        ):
            raise ValueError(f"Original {feature} failed exact audited reproduction")
        differences.append(
            AcceptedRepresentationDifference(
                asset_id,
                signal_session,
                feature,
                parent_bits.hex(),
                computed_bits.hex(),
                steps,
                "authenticated_parent_raw_value_unchanged",
                POLICY_SHA256,
                DIAGNOSTIC_SHA256,
                WHITELIST_SHA256,
                hashlib.sha256(_canonical(expected_context).encode()).hexdigest(),
            )
        )
    return ReproducedFeatures(parent_features, tuple(differences))
