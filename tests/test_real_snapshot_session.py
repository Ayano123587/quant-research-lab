"""Read-session controls using only the completed actual normalized archive."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from quantlab.data.snapshot import MarketSnapshot, SnapshotError

pytestmark = pytest.mark.real_data
MANIFEST_SHA256 = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
CONTRACT_SHA256 = "690e78739f0be396b61c1097e03795ce4d1b0217639e8fb2b70cf136b90c9f1e"


@pytest.fixture(scope="module")
def snapshot_path():
    root = Path(
        os.environ.get(
            "QUANTLAB_SNAPSHOT",
            str(Path(__file__).resolve().parents[1] / "data/processed/market-v1"),
        )
    )
    if not (root / "COMPLETE").is_file():
        pytest.skip("Completed authenticated supplied snapshot required")
    return root


@pytest.fixture(scope="module")
def snapshot(snapshot_path):
    return MarketSnapshot(snapshot_path, MANIFEST_SHA256)


def test_actual_histories_equal_before_inside_after_with_fewer_manifest_reads(
    snapshot, monkeypatch
):
    requested = [("etf", s) for s in ("SPY", "EFA", "EEM", "TLT", "LQD", "GLD", "DBC", "VNQ")]
    requested += [("stock", "IBM"), ("stock", "GE")]
    manifest_reads = []
    original = snapshot._read

    def observe(name):
        if name == "manifest.json":
            manifest_reads.append(name)
        return original(name)

    monkeypatch.setattr(snapshot, "_read", observe)
    before = {key: snapshot.bars(*key) for key in requested}
    assert len(manifest_reads) == len(requested)
    with snapshot.verified_read_session() as verified:
        assert verified is snapshot
        count_before_loop = len(manifest_reads)
        inside = {key: verified.bars(*key) for key in requested}
        assert len(manifest_reads) == count_before_loop
    assert inside == before
    after = {key: snapshot.bars(*key) for key in requested}
    assert after == before
    assert len(manifest_reads) == 2 * len(requested)
    assert all(before.values())


def test_actual_session_runs_full_entry_and_exit_even_on_invalid_request(snapshot, monkeypatch):
    checks = []
    original = snapshot._full_session_check

    def observe():
        original()
        checks.append("authenticated")

    monkeypatch.setattr(snapshot, "_full_session_check", observe)
    with pytest.raises(SnapshotError, match="Source asset type"):
        with snapshot.verified_read_session():
            snapshot.bars("etf", None)
    assert checks == ["authenticated", "authenticated"]
    assert snapshot._active_session is None
    assert snapshot.bars("etf", "SPY")


def test_nested_concurrent_and_reused_actual_session_are_rejected(snapshot):
    session = snapshot.verified_read_session()
    with session:
        with pytest.raises(SnapshotError, match="Nested or concurrent"):
            with snapshot.verified_read_session():
                pass

        def concurrent_entry():
            with snapshot.verified_read_session():
                pass

        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(SnapshotError, match="Nested or concurrent"):
                executor.submit(concurrent_entry).result()
            with pytest.raises(SnapshotError, match="Concurrent read"):
                executor.submit(snapshot.bars, "etf", "SPY").result()
        assert snapshot.bars("stock", "IBM")
    with pytest.raises(SnapshotError, match="cannot be reentered"):
        with session:
            pass


def test_wrong_actual_external_identity_refused(snapshot_path):
    # This is the actual contract SHA, not the externally reviewed manifest SHA.
    # No alternative data or fabricated digest is produced for this refusal.
    with pytest.raises(SnapshotError, match="COMPLETE does not match"):
        MarketSnapshot(snapshot_path, CONTRACT_SHA256)


def test_actual_calendar_columns_unchanged_in_session(snapshot):
    before = snapshot.rows("etf:spy.us")
    with snapshot.verified_read_session():
        inside = snapshot.rows("etf:spy.us")
    assert inside == before
    assert all(row["source_archive_sha256"] == snapshot.archive_sha256 for row in inside)
    assert all(
        row["xnys_open_at"] < row["xnys_close_at"] < row["assumed_available_at"]
        for row in inside
        if row["is_xnys_session"]
    )
