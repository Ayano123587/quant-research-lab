"""Cloud-only pre-2015 engineering check of the unchanged concrete learner adapter."""

from __future__ import annotations

import json
import platform
from datetime import date
from importlib.metadata import version
from pathlib import Path

from quantlab.evaluation.holdout import _bound, file_sha, mature_training_keys, source_closure


def run_smoke(plan_path, expected_sha256, *, project, inputs, output):
    """Fit exactly one registered Ridge and LightGBM, without evaluating outcomes."""
    if platform.system() != "Linux" or not str(Path.cwd()).startswith("/kaggle/working"):
        raise ValueError("Actual learner integration check is Kaggle-only")
    if file_sha(plan_path) != expected_sha256:
        raise ValueError("Externally pinned engineering plan required")
    plan = json.loads(Path(plan_path).read_bytes())
    days = tuple(date.fromisoformat(d) for d in plan["prediction_sessions"])
    if (
        plan["schema"] != "pre2015_stock_adapter_training_engineering_v1"
        or not days
        or len(days) > 2
        or tuple(sorted(set(days))) != days
        or any(d.year != 2010 for d in days)
        or plan["contains_final_evaluation"] is not False
        or plan["scores_prediction_outcomes"] is not False
    ):
        raise ValueError("Only explicit pre-2015 engineering predictions allowed")
    own = Path(__file__).resolve().relative_to(Path(project).resolve()).as_posix()
    closure = source_closure(project, [own])
    if closure != plan["source_sha256"]:
        raise ValueError("Complete actual engineering source closure differs")
    runtime = plan["runtime"]
    if runtime != {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {p: version(p) for p in runtime["packages"]},
    }:
        raise ValueError("Actual engineering runtime differs")
    if not {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "exchange-calendars",
    } <= set(runtime["packages"]):
        raise ValueError("All actual learner/data/calendar dependencies must be pinned")
    config = _bound(project, plan["config_path"])
    if file_sha(config) != plan["config_sha256"]:
        raise ValueError("Actual unchanged stock configuration differs")
    panel = _bound(inputs, plan["panel_manifest_path"])
    if panel.name != "manifest.json" or file_sha(panel) != plan["panel_manifest_sha256"]:
        raise ValueError("Actual completed preparation manifest differs")
    # Only after source/config/input/runtime authentication import numerical code.
    from quantlab.backtest.engine import HistoricalView
    from quantlab.data.calendar import XNYSSchedule
    from quantlab.evaluation.holdout_adapters import StockPanelProvider
    from quantlab.evaluation.holdout_pipeline import _write
    from quantlab.stocks.development import FEATURE_IDS, load_spec, registered_candidates
    from quantlab.stocks.prepare import verify_prepared_panel

    manifest = verify_prepared_panel(panel.parent, plan["panel_manifest_sha256"])
    calendar = manifest["config"]["calendar"]
    schedule = XNYSSchedule(
        date.fromisoformat(calendar["start"]), date.fromisoformat(calendar["end"])
    )
    if schedule.provenance != calendar or any(
        schedule.month_end(d.year, d.month).session != d for d in days
    ):
        raise ValueError("Actual month-end/calendar identity differs")
    spec = load_spec(config)
    grid = {c.candidate_id: c for c in registered_candidates(spec)}
    candidates = tuple(grid[c] for c in plan["candidate_ids"])
    if len(candidates) != 2 or {c.kind for c in candidates} != {"ridge", "lightgbm"}:
        raise ValueError("Exactly one existing Ridge and LightGBM configuration required")
    partitions = [
        {**r, "session": Path(r["path"]).stem}
        for r in manifest["outputs"]
        if r.get("kind") in {"features", "labels"}
        and date.fromisoformat(Path(r["path"]).stem) <= days[-1]
    ]
    if any(date.fromisoformat(r["session"]).year >= 2015 for r in partitions):
        raise ValueError("Final partitions cannot enter engineering providers")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts, receipts = {}, []
    try:
        for candidate in candidates:
            provider = StockPanelProvider(
                panel.parent,
                partitions,
                schedule,
                days,
                candidate=candidate,
                feature_columns=tuple("rank_" + f for f in FEATURE_IDS),
                raw_columns=tuple("raw_" + f for f in FEATURE_IDS),
                target_policy=spec["target_transform"],
                threads=spec["threads"],
                target_spec=spec,
                minimum_rows=spec["minimum_training_rows"],
                minimum_dates=spec["minimum_training_signal_dates"],
            )
            for day in days:
                view = HistoricalView(
                    (), day, schedule.session(day).decision_at, schedule.next_session(day).open_at
                )
                selected = mature_training_keys(
                    provider.clocks(view), signal_session=day, decision_at=view.cutoff
                )
                fit = provider.fit(view, selected)
                if fit.used_keys != selected.keys or fit.cutoff != view.cutoff:
                    raise ValueError("Actual fit receipt differs from approved mature keys")
                decision = provider.decide(view, None)
                receipts.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "session": day,
                        "fit": fit,
                        "training_rows": len(selected.keys),
                        "training_dates": len({d for d, _ in selected.keys}),
                        "decision": decision,
                        "predictions": provider.predictions[-1],
                    }
                )
            path = output / (candidate.candidate_id + ".json")
            artifacts[path.name] = _write(
                path,
                {
                    "engineering_only": True,
                    "records": [r for r in receipts if r["candidate_id"] == candidate.candidate_id],
                },
            )
        if (
            source_closure(project, [own]) != closure
            or file_sha(panel) != plan["panel_manifest_sha256"]
            or file_sha(config) != plan["config_sha256"]
        ):
            raise ValueError("Engineering source/input/config changed during fitting")
        if runtime != {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "packages": {p: version(p) for p in runtime["packages"]},
        }:
            raise ValueError("Engineering runtime changed during fitting")
        for item in (*partitions, *artifacts.values()):
            path = (
                Path(item["path"])
                if Path(item["path"]).is_absolute()
                else _bound(panel.parent, item["path"])
            )
            if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
                raise ValueError("Actual training input or saved forecast bytes changed")
        result = {
            "complete": True,
            "scope": "pre2015_adapter_integration_not_final_evaluation",
            "plan_sha256": expected_sha256,
            "source_sha256": closure,
            "runtime": runtime,
            "actual_fits": len(receipts),
            "artifacts": artifacts,
            "outcome_scores_computed": False,
        }
        _write(output / "COMPLETE.json", result)
        return result
    except Exception as error:
        _write(
            output / "FAILURE.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "reason": str(error),
                "artifacts": artifacts,
                "automatic_retry": False,
            },
        )
        raise
