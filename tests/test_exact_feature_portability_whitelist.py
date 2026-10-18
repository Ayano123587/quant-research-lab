"""Exact public derived feature-bit audit; no market prices or model fitting."""

import struct
from dataclasses import asdict
from datetime import date

import pytest

from quantlab.stocks.feature_reproduction import (
    PARENT_MANIFEST_SHA256,
    POLICY,
    SIX,
    _whitelist,
    reproduce_raw_six,
    require_full_acceptance_inventory,
)


def compare(row, **overrides):
    kwargs = dict(
        asset_id=row["asset_id"],
        signal_session=date.fromisoformat(row["signal_session"]),
        parent_prepared_manifest_sha256=PARENT_MANIFEST_SHA256,
        policy=POLICY,
        parent_platform="Windows",
        computed_platform="Linux",
        parent_python="3.12.15",
        computed_python="3.12.15",
        source_context=row,
    )
    kwargs.update(overrides)
    return reproduce_raw_six(
        tuple(
            (f, struct.unpack("<d", bytes.fromhex(row["observed_six"][f]["parent_float64_hex"]))[0])
            for f in SIX
        ),
        tuple(
            (
                f,
                struct.unpack("<d", bytes.fromhex(row["observed_six"][f]["computed_float64_hex"]))[
                    0
                ],
            )
            for f in SIX
        ),
        **kwargs,
    )


def test_all_actual_894_rows_896_fields_exact_whitelist_and_parent_retention():
    accepted = []
    counts = {}
    for row in _whitelist().values():
        with pytest.raises(ValueError, match="failed exact audited"):
            compare(row, policy=None)
        result = compare(row)
        for feature, value in result.retained_parent_features:
            assert (
                struct.pack("<d", value).hex() == row["observed_six"][feature]["parent_float64_hex"]
            )
        for record in result.accepted_differences:
            key = (record.feature_id, record.adjacent_float64_steps)
            counts[key] = counts.get(key, 0) + 1
            accepted.append(asdict(record))
    assert counts == {("F09", 1): 545, ("F09", 2): 4, ("F19", 1): 347}
    require_full_acceptance_inventory(accepted)
    with pytest.raises(ValueError, match="896"):
        require_full_acceptance_inventory(accepted[:-1])
    with pytest.raises(ValueError, match="896"):
        require_full_acceptance_inventory(accepted + accepted[:1])


def test_observed_bits_cannot_be_reused_for_other_actual_identity_or_context():
    rows = list(_whitelist().values())
    first, other = rows[:2]
    with pytest.raises(ValueError, match="failed exact audited"):
        compare(first, source_context=other)
    with pytest.raises(ValueError, match="failed exact audited"):
        compare(
            first,
            asset_id=other["asset_id"],
            signal_session=date.fromisoformat(other["signal_session"]),
        )
    with pytest.raises(ValueError, match="runtime provenance"):
        compare(first, computed_platform="Windows")
