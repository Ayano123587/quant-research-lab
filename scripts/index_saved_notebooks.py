"""Index actual saved notebook outputs without executing research again."""

import hashlib
import json
from pathlib import Path

import nbformat


def index(root):
    target = root / "reports/notebook-execution.json"
    previous = json.loads(target.read_bytes()) if target.exists() else {}
    evidence = {row["notebook"]: row for row in previous.get("notebooks", [])}
    rows = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
        if not code or any(
            not isinstance(c.execution_count, int) or c.execution_count < 1 for c in code
        ):
            raise ValueError(f"Notebook has unexecuted code: {path.name}")
        outputs = [output for cell in code for output in cell.get("outputs", [])]
        if not outputs or any(output.output_type == "error" for output in outputs):
            raise ValueError(f"Notebook lacks successful saved outputs: {path.name}")
        name = path.relative_to(root).as_posix()
        old = evidence.get(name, {})
        # Preserve recorded execution evidence only while the notebook bytes match.
        row = dict(old) if old.get("sha256") == digest else {}
        row.update(
            notebook=name,
            sha256=digest,
            executed_code_cells=len(code),
            inline_plots=sum("image/png" in output.get("data", {}) for output in outputs),
            html_table_outputs=sum(
                "<table" in str(output.get("data", {}).get("text/html", ""))
                for output in outputs
            ),
            execution_counts=[cell.execution_count for cell in code],
            saved_outputs_valid=True,
        )
        rows.append(row)
    previous["notebooks"] = rows
    previous["index_scope"] = (
        "Validated saved notebook bytes, execution counts and outputs. "
        "Indexing does not rerun cells or independently prove their research results. "
        "Execution durations are included only where already recorded for these exact bytes."
    )
    target.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    print(f"Indexed {index(Path(__file__).resolve().parents[1])} saved notebooks")
