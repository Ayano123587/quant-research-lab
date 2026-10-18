"""Progress-file sharing failures with unchanged actual preparation receipts."""

import sys
from pathlib import Path

import pytest

from quantlab.stocks.prepare import _replace_progress_file


@pytest.fixture
def actual_progress(tmp_path):
    source = Path("data/processed/stock-monthly-v1/status.json")
    if not source.is_file():
        pytest.skip("Actual interrupted preparation receipt is unavailable")
    raw = source.read_bytes()
    pending, destination = tmp_path / "status.pending", tmp_path / "status.json"
    pending.write_bytes(raw)
    destination.write_bytes(raw)
    return pending, destination, raw


def test_windows_reader_lock_retries_then_preserves_exact_receipt(actual_progress, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows sharing-error recovery")
    pending, destination, raw = actual_progress
    original, attempts, sleeps = Path.replace, [], []

    def temporarily_locked(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            error = PermissionError("Injected Windows reader lock; actual receipt unchanged")
            error.winerror = 5
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr("quantlab.stocks.prepare.time.sleep", sleeps.append)
    _replace_progress_file(pending, destination)
    assert len(attempts) == 3 and sleeps == [0.05, 0.1]
    assert not pending.exists() and destination.read_bytes() == raw


def test_persistent_windows_lock_fails_bounded_and_preserves_both_files(
    actual_progress, monkeypatch
):
    if sys.platform != "win32":
        pytest.skip("Windows sharing-error recovery")
    pending, destination, raw = actual_progress
    attempts, sleeps = [], []

    def locked(path, target):
        attempts.append(path)
        error = PermissionError("Injected persistent software lock")
        error.winerror = 32
        raise error

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("quantlab.stocks.prepare.time.sleep", sleeps.append)
    with pytest.raises(PermissionError):
        _replace_progress_file(pending, destination)
    assert len(attempts) == 20 and len(sleeps) == 19 and sum(sleeps) < 5
    assert pending.read_bytes() == raw and destination.read_bytes() == raw


def test_other_permissions_fail_immediately(actual_progress, monkeypatch):
    pending, destination, raw = actual_progress
    attempts = []

    def denied(path, target):
        attempts.append(path)
        raise PermissionError("Injected permanent permission denial")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError):
        _replace_progress_file(pending, destination)
    assert len(attempts) == 1
    assert pending.read_bytes() == raw and destination.read_bytes() == raw
