"""Real interpreter-file links; no market records or numerical computation."""

import os
import shutil
import subprocess
import sys

import pytest

from cloud.etf_cost_runner import bootstrap_interpreter


def link_directory(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # NTFS directory junctions exercise real target resolution without the
        # administrator privilege needed for file symlinks on this host.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
    else:
        os.symlink(target, link, target_is_directory=True)


@pytest.fixture
def linked_interpreter(tmp_path):
    private = tmp_path / "private"
    managed = private / "python/managed/bin/python"
    managed.parent.mkdir(parents=True)
    shutil.copyfile(sys.executable, managed)
    virtual = private / "runtime/.venv/bin/python"
    link_directory(virtual.parent, managed.parent)
    return private, managed, virtual


def test_private_venv_link_to_actual_private_managed_binary(linked_interpreter):
    private, managed, virtual = linked_interpreter
    assert virtual.is_file() and virtual.resolve() == managed.resolve()
    assert (
        bootstrap_interpreter(
            private, {"python": str(virtual), "evidence": {"managed_interpreter": str(managed)}}
        )
        == virtual
    )


def test_changed_or_outside_target_refused(linked_interpreter, tmp_path):
    private, managed, virtual = linked_interpreter
    outside = tmp_path / "outside-python"
    shutil.copyfile(sys.executable, outside)
    with pytest.raises(ValueError, match="Exact private"):
        bootstrap_interpreter(
            private, {"python": str(virtual), "evidence": {"managed_interpreter": str(outside)}}
        )
    other_private = tmp_path / "other-private"
    outside_virtual = other_private / "runtime/.venv/bin/python"
    link_directory(outside_virtual.parent, managed.parent)
    with pytest.raises(ValueError, match="Exact private"):
        bootstrap_interpreter(
            other_private,
            {"python": str(outside_virtual), "evidence": {"managed_interpreter": str(managed)}},
        )
