"""Validate saved notebook bytes and their index without executing cells."""

import hashlib
import json
from pathlib import Path

import nbformat


def validate(root):
    root = root.resolve()
    indexed = json.loads((root / "reports/notebook-execution.json").read_bytes())["notebooks"]
    names = [row["notebook"] for row in indexed]
    actual = {p.relative_to(root).as_posix() for p in (root / "notebooks").glob("*.ipynb")}
    if not actual or len(names) != len(set(names)) or set(names) != actual:
        raise ValueError("Notebook index must cover every actual notebook exactly once")
    for row in indexed:
        path = root / row["notebook"]
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe notebook path")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Stale notebook index: {path.name}")
        notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
        outputs = [o for c in code for o in c.get("outputs", [])]
        counts = [c.execution_count for c in code]
        expected = {
            "executed_code_cells": len(code),
            "execution_counts": counts,
            "inline_plots": sum("image/png" in o.get("data", {}) for o in outputs),
            "html_table_outputs": sum(
                "<table" in str(o.get("data", {}).get("text/html", "")) for o in outputs
            ),
            "saved_outputs_valid": True,
        }
        if (
            not code or not outputs
            or any(type(n) is not int or n < 1 for n in counts)
            or any(o.output_type == "error" for o in outputs)
            or any(row.get(k) != value for k, value in expected.items())
        ):
            raise ValueError(f"Invalid saved notebook evidence: {path.name}")
    return len(actual)


if __name__ == "__main__":
    print(f"Validated {validate(Path(__file__).resolve().parents[1])} saved notebooks")
