"""Development-only prequential candidates; full fitting is a Kaggle operation.

Authentication may inspect structural Parquet footers across the whole prepared
panel. Outcome VALUE columns are read only after a clock-only projection proves
the entire partition matures by the 2014 selection cutoff. No 2015+ signal
partition or 2014 label spilling into 2015 supplies outcome values to this code.
"""

import hashlib
import json
import math
import os
import platform
import shutil
import time
import tomllib
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from quantlab.backtest import engine as engine_module
from quantlab.backtest.engine import Decision
from quantlab.data.archive import file_sha256
from quantlab.data.calendar import XNYSSchedule
from quantlab.stocks.features import FEATURE_IDS
from quantlab.stocks.prepare import _source_code as prepared_source_closure
from quantlab.stocks.prepare import verify_prepared_panel

HOLDOUT_START = date(2015, 1, 1)


def development_source_closure():
    from . import training_targets as target_module

    return {
        **prepared_source_closure(),
        "development": file_sha256(Path(__file__)),
        "engine": file_sha256(Path(engine_module.__file__)),
        "training_targets": file_sha256(Path(target_module.__file__)),
    }


def prediction_score_kind(candidate, spec):
    from .training_targets import RANK, RAW

    if candidate.kind == "control":
        return "ordinal_score"
    if spec["target_transform"] == RANK:
        return "centered_cross_section_rank_forecast"
    if spec["target_transform"] == RAW:
        return "decimal_return_forecast"
    raise ValueError("Registered score units required")


def _raw(value):
    return (
        json.dumps(
            value, sort_keys=True, indent=2, allow_nan=False, default=lambda x: x.isoformat()
        )
        + "\n"
    ).encode()


def _digest(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(_raw(value))
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def restore_checkpoint(path, expected_sha256, destination, pins):
    """Copy only externally pinned, complete cells into a fresh output directory.

    A checkpoint is an interruption receipt, never a final research manifest.
    Native model payloads are hashed/copied as opaque bytes, never unpickled.
    """
    path, destination = Path(path).resolve(), Path(destination).resolve()
    origin = path.parent.parent
    cells, chain, visited = [], [], set()
    cursor, expected = path, expected_sha256
    while cursor is not None:
        if cursor in visited or origin not in cursor.parents:
            raise ValueError("Invalid checkpoint chain")
        visited.add(cursor)
        if file_sha256(cursor) != expected:
            raise ValueError("Checkpoint SHA mismatch")
        receipt = json.loads(cursor.read_text(encoding="utf-8"))
        if receipt["pins"] != json.loads(_raw(pins)) or receipt["complete"] is not False:
            raise ValueError("Checkpoint source/config/runtime pins mismatch")
        cells[0:0] = receipt["cells"]
        chain.append(cursor)
        parent = receipt.get("parent_checkpoint")
        cursor = (origin / parent["path"]).resolve() if parent else None
        expected = parent["sha256"] if parent else None
    seen = set()
    for cell in cells:
        key = (cell["signal_session"], cell["candidate_id"])
        if key in seen:
            raise ValueError("Duplicate checkpoint cell")
        seen.add(key)
        for artifact in cell["artifacts"]:
            relative = Path(artifact["path"])
            source, target = (origin / relative).resolve(), (destination / relative).resolve()
            if origin not in source.parents or destination not in target.parents:
                raise ValueError("Checkpoint artifact escapes output")
            if not source.is_file() or file_sha256(source) != artifact["sha256"]:
                raise ValueError("Checkpoint artifact SHA mismatch")
            if target.exists():
                raise ValueError("Resume destination must not contain checkpoint artifacts")
    # Validate every reference before copying any payload.
    for cell in cells:
        for artifact in cell["artifacts"]:
            target = destination / artifact["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / artifact["path"], target)
    for checkpoint in chain:
        target = destination / checkpoint.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint, target)
    return cells


def require_completed_grid(cells, decision_dates, candidates):
    """A complete cell is required even when real support prevents fitting."""
    expected = {(day.isoformat(), c.candidate_id) for day in decision_dates for c in candidates}
    actual = {(cell["signal_session"], cell["candidate_id"]) for cell in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError("Incomplete registered date/candidate grid; comparison refused")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    parameters: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class FeatureRow:
    asset_id: str
    signal_session: date
    signal_at: datetime
    intended_execution_at: datetime
    values: tuple[float, ...]
    raw_values: tuple[float, ...]
    input_rows_sha256: str


@dataclass(frozen=True)
class Outcome:
    asset_id: str
    signal_session: date
    available_at: datetime
    value: float
    input_rows_sha256: str


@dataclass(frozen=True)
class LabelClock:
    asset_id: str
    signal_session: date
    available_at: datetime | None
    status: str


@dataclass(frozen=True)
class DevelopmentData:
    prepared_manifest_sha256: str
    full_research_scope: bool
    selection_cutoff: datetime
    features: tuple[FeatureRow, ...]
    outcomes: tuple[Outcome, ...]
    clocks: tuple[LabelClock, ...]
    decision_times: tuple[tuple[date, datetime], ...]
    outcome_value_partitions_read: tuple[str, ...]
    outcome_value_partitions_refused: tuple[str, ...]


@dataclass(frozen=True)
class TrainingBatch:
    feature_rows: tuple[FeatureRow, ...]
    outcomes: tuple[Outcome, ...]
    weights: tuple[float, ...]
    information_cutoff: datetime
    date_counts: tuple[tuple[date, int], ...]
    max_label_available_at: datetime | None
    input_sha256: str
    ready: bool
    reasons: tuple[str, ...]


def load_spec(path):
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    from .training_targets import RANK, RAW

    target_policy = spec.get("target_transform")
    expected_status = {
        RAW: "initial_development_family_not_final_holdout_recipe",
        RANK: "rank_target_development_family_not_final_holdout_recipe",
    }.get(target_policy)
    if (
        tuple(spec["feature_ids"]) != FEATURE_IDS
        or expected_status is None
        or spec["status"] != expected_status
        or spec["selection_information_cutoff_session"] != "2014-12-31"
        or spec["minimum_training_signal_dates"] != 36
        or spec["minimum_training_rows"] != 10000
        or spec["minimum_scored_cross_section"] != 30
        or spec["threads"] != 4
        or spec["random_seed"] != 20261017
        or spec["baselines"]["reversal_feature"] != "F07"
        or spec["selection_tie_rule"] != "higher_mean_IC_then_ascending_fixed_candidate_id"
        or spec["training_sample_weight"] != "inverse_rows_per_signal_date_normalized_to_mean_one"
        or spec["ridge"]["objective"]
        != (
            "weighted_sum_squared_error_plus_alpha_times_squared_coefficient_norm_unpenalized_intercept"
        )
        or spec["final_period"] != "refused_by_development_runner"
    ):
        raise ValueError("Registered development feature/clock/objective/support contract required")
    windows = spec["validation_windows"]
    previous = None
    for window in windows:
        start, end = (
            date.fromisoformat(window["signal_start"]),
            date.fromisoformat(window["signal_end"]),
        )
        if start > end or end >= HOLDOUT_START or (previous is not None and start <= previous):
            raise ValueError("Nonoverlapping forward development date groups required")
        previous = end
    registered_candidates(spec)
    return spec


def registered_candidates(spec):
    ridge, light = spec["ridge"], spec["lightgbm"]
    if (
        ridge["alphas"] != [0.1, 1.0, 10.0, 100.0]
        or ridge["solver"] != "svd"
        or ridge["fit_intercept"] is not True
        or light["objectives"] != ["regression_l1", "regression_l2"]
        or light["num_leaves"] != [7, 15]
        or light["n_estimators"] != [100, 300]
        or light["early_stopping"] != "none_fixed_iterations_no_future_date_early_stopping"
    ):
        raise ValueError("Exactly four Ridge and eight registered boosting candidates required")
    candidates = [
        Candidate("control_momentum_F01", "control", (("F01", 1.0),)),
        Candidate("control_reversal_F07", "control", (("F07", 1.0),)),
        Candidate("control_low_volatility_F09", "control", (("F09", -1.0),)),
        Candidate(
            "control_fixed_composite",
            "control",
            tuple((f, float(spec["fixed_composite"][f])) for f in FEATURE_IDS),
        ),
    ]
    for alpha in ridge["alphas"]:
        candidates.append(
            Candidate(
                "ridge_alpha_" + format(alpha, "g").replace(".", "p"),
                "ridge",
                (("alpha", alpha), ("solver", "svd"), ("fit_intercept", True)),
            )
        )
    fixed = {
        k: light[k]
        for k in (
            "max_depth",
            "learning_rate",
            "min_child_samples",
            "reg_lambda",
            "colsample_bytree",
            "subsample",
            "deterministic",
            "force_col_wise",
        )
    }
    fixed.update(
        random_state=spec["random_seed"], n_jobs=spec["threads"], device_type="cpu", verbosity=-1
    )
    for objective in light["objectives"]:
        for leaves in light["num_leaves"]:
            for iterations in light["n_estimators"]:
                parameters = {
                    **fixed,
                    "objective": objective,
                    "num_leaves": leaves,
                    "n_estimators": iterations,
                }
                candidates.append(
                    Candidate(
                        f"lgbm_{objective}_leaves{leaves}_iterations{iterations}",
                        "lightgbm",
                        tuple(sorted(parameters.items())),
                    )
                )
    return tuple(sorted(candidates, key=lambda c: c.candidate_id))


def _read_columns(path, columns):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).read(columns=columns).to_pylist()


def read_development_inputs(
    prepared_root, expected_manifest_sha256, spec, schedule, *, allow_scoped_test=False
):
    """Clock-first column projection, never read then discard a future return."""
    if spec["selection_information_cutoff_session"] != "2014-12-31":
        raise ValueError("Development refuses a selection cutoff exposing 2015 outcomes")
    manifest = verify_prepared_panel(
        prepared_root, expected_manifest_sha256, allow_scoped_test=allow_scoped_test
    )
    if manifest["config"]["source_code_sha256"] != prepared_source_closure():
        raise ValueError("Preparation source differs from the completed feature/label panel")
    if (
        not isinstance(schedule, XNYSSchedule)
        or schedule.provenance != manifest["config"]["calendar"]
    ):
        raise ValueError("Exact actual preparation calendar required")
    cutoff = schedule.session(
        date.fromisoformat(spec["selection_information_cutoff_session"])
    ).decision_at
    root = Path(prepared_root)
    tables = {
        (item["kind"], date.fromisoformat(Path(item["path"]).stem)): item["path"]
        for item in manifest["outputs"]
        if "kind" in item
    }
    features, outcomes, clocks, times, read, refused = [], [], [], [], [], []
    for day in sorted(d for kind, d in tables if kind == "features"):
        if day >= HOLDOUT_START or schedule.session(day).decision_at > cutoff:
            refused.append(tables["labels", day])
            continue
        signal = schedule.session(day).decision_at
        times.append((day, signal))
        current = _read_columns(
            root / tables["features", day],
            [
                "asset_id",
                "signal_session",
                "signal_at",
                "intended_execution_at",
                "max_input_available_at",
                "input_rows_sha256",
                *("rank_" + f for f in FEATURE_IDS),
                *("raw_" + f for f in FEATURE_IDS),
            ],
        )
        ids = set()
        for row in current:
            values = tuple(float(row["rank_" + f]) for f in FEATURE_IDS)
            raw_values = tuple(float(row["raw_" + f]) for f in FEATURE_IDS)
            if (
                row["asset_id"] in ids
                or row["signal_session"] != day
                or row["signal_at"] != signal
                or row["max_input_available_at"] > signal
                or row["intended_execution_at"] != schedule.next_session(day).open_at
                or not all(math.isfinite(v) for v in (*values, *raw_values))
            ):
                raise ValueError("Feature key/availability/finite-value contract failed")
            ids.add(row["asset_id"])
            features.append(
                FeatureRow(
                    row["asset_id"],
                    day,
                    signal,
                    row["intended_execution_at"],
                    values,
                    raw_values,
                    row["input_rows_sha256"],
                )
            )
        relative = tables["labels", day]
        # First projection deliberately excludes prices and return values.
        metadata = _read_columns(
            root / relative,
            [
                "asset_id",
                "signal_session",
                "signal_at",
                "planned_available_at",
                "label_available_at",
                "status",
                "quality_flags",
                "input_rows_sha256",
            ],
        )
        expected_next_year, expected_next_month = (
            (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        )
        planned = schedule.next_session(
            schedule.month_end(expected_next_year, expected_next_month).session
        ).decision_at
        for row in metadata:
            if (
                row["signal_session"] != day
                or row["signal_at"] != signal
                or row["planned_available_at"] != planned
                or (
                    row["status"] == "valid"
                    and (row["label_available_at"] != planned or row["quality_flags"])
                )
            ):
                raise ValueError("Label clock contract failed before outcome values were read")
            if row["asset_id"] in ids:
                clocks.append(
                    LabelClock(row["asset_id"], day, row["label_available_at"], row["status"])
                )
        if planned > cutoff:
            refused.append(relative)
            continue
        values = _read_columns(root / relative, ["asset_id", "return_value"])
        read.append(relative)
        target = {row["asset_id"]: row["return_value"] for row in values}
        if len(target) != len(values) or len({r["asset_id"] for r in metadata}) != len(metadata):
            raise ValueError("Duplicate label join key")
        for row in metadata:
            if row["asset_id"] in ids and row["status"] == "valid":
                value = target[row["asset_id"]]
                if value is None or not math.isfinite(value) or value <= -1:
                    raise ValueError("Resolved positive-price outcome required; no replacement")
                outcomes.append(
                    Outcome(
                        row["asset_id"],
                        day,
                        row["label_available_at"],
                        float(value),
                        row["input_rows_sha256"],
                    )
                )
    return DevelopmentData(
        expected_manifest_sha256,
        manifest["research_scope_complete"],
        cutoff,
        tuple(sorted(features, key=lambda r: (r.signal_session, r.asset_id))),
        tuple(sorted(outcomes, key=lambda r: (r.signal_session, r.asset_id))),
        tuple(clocks),
        tuple(times),
        tuple(read),
        tuple(refused),
    )


def training_batch(data, signal_session, information_cutoff, spec):
    """Expand across all prior prepared months; maturity gates every individual fit."""
    if (
        signal_session >= HOLDOUT_START
        or information_cutoff > data.selection_cutoff
        or dict(data.decision_times).get(signal_session) != information_cutoff
    ):
        raise ValueError("Exact development decision cutoff required")
    labels = {
        (r.signal_session, r.asset_id): r
        for r in data.outcomes
        if r.available_at <= information_cutoff and r.signal_session < signal_session
    }
    selected = tuple(
        r
        for r in data.features
        if r.signal_session < signal_session and (r.signal_session, r.asset_id) in labels
    )
    outcomes = tuple(labels[r.signal_session, r.asset_id] for r in selected)
    counts = Counter(r.signal_session for r in selected)
    n, months = len(selected), len(counts)
    weights = tuple(n / (months * counts[r.signal_session]) for r in selected)
    reasons = []
    if months < spec["minimum_training_signal_dates"]:
        reasons.append("INSUFFICIENT_TRAINING_DATES")
    if n < spec["minimum_training_rows"]:
        reasons.append("INSUFFICIENT_TRAINING_ROWS")
    payload = {
        "features": FEATURE_IDS,
        "weight_convention": spec["training_sample_weight"],
        "fit_cutoff": information_cutoff,
        "rows": [
            (
                r.signal_session,
                r.asset_id,
                r.input_rows_sha256,
                r.values,
                label.input_rows_sha256,
                label.available_at,
                label.value,
                weight,
            )
            for r, label, weight in zip(selected, outcomes, weights, strict=True)
        ],
    }
    return TrainingBatch(
        selected,
        outcomes,
        weights,
        information_cutoff,
        tuple(sorted(counts.items())),
        max((r.available_at for r in outcomes), default=None),
        _digest(payload),
        not reasons,
        tuple(reasons),
    )


def control_scores(candidate, rows):
    if candidate.kind != "control":
        raise ValueError("Only fixed controls may score without fitting")
    weights = dict(candidate.parameters)
    return tuple(
        (row.asset_id, math.fsum(w * row.values[FEATURE_IDS.index(f)] for f, w in weights.items()))
        for row in rows
    )


def target_from_scores(scores, signal_at, intended_execution_at, spec):
    scores = tuple(scores)
    if len({a for a, _ in scores}) != len(scores) or not all(math.isfinite(v) for _, v in scores):
        raise ValueError("Unique source IDs and finite actual prediction scores required")
    n = spec["baselines"]["positions"]
    if len(scores) < n:
        return Decision(
            "HOLD",
            None,
            signal_at,
            intended_execution_at,
            "insufficient_scored_cross_section_preserve_previous_positions_and_cash",
        )
    weight = 1 / n
    if weight > spec["baselines"]["single_asset_cap"]:
        raise ValueError("Registered equal top-N target violates cap")
    selected = sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:n]
    return Decision(
        "REBALANCE",
        tuple((asset, weight) for asset, _ in selected),
        signal_at,
        intended_execution_at,
        "fixed_top_N_descending_score_source_id_ties",
    )


def date_ic(scores, outcomes, *, minimum_support=30):
    """Per-date observed support only; undefined correlations never become zero."""
    if minimum_support < 30:
        raise ValueError("Registered IC requires at least thirty resolved scored names")
    scores = tuple(scores)
    if len({asset for asset, _ in scores}) != len(scores):
        raise ValueError("Duplicate prediction key before outcome join")
    labels = {r.asset_id: r.value for r in outcomes}
    if len(labels) != len(outcomes):
        raise ValueError("Duplicate outcome key before evaluation join")
    paired = [(score, labels[asset]) for asset, score in scores if asset in labels]
    base = {"n_predictions": len(scores), "n_resolved": len(paired), "ic": None}
    if len(paired) < minimum_support:
        return {**base, "status": "insufficient_resolved_cross_section"}
    x, y = zip(*paired, strict=True)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {**base, "status": "undefined_constant_cross_section"}
    from scipy.stats import spearmanr

    value = float(spearmanr(x, y).statistic)
    if not math.isfinite(value):
        return {**base, "status": "undefined_nonfinite_correlation"}
    return {**base, "status": "valid", "ic": value}


def common_paired_summary(metrics, candidates):
    ids = tuple(c.candidate_id for c in candidates)
    indexed = {(row["candidate_id"], row["signal_session"]): row for row in metrics}
    if len(indexed) != len(metrics):
        raise ValueError("Duplicate candidate-date metric")
    dates = sorted({row["signal_session"] for row in metrics})
    common = tuple(
        day
        for day in dates
        if all(
            (candidate, day) in indexed and indexed[candidate, day]["status"] == "valid"
            for candidate in ids
        )
    )
    results = [
        {
            "candidate_id": candidate,
            "common_date_count": len(common),
            "mean_monthly_ic": math.fsum(indexed[candidate, day]["ic"] for day in common)
            / len(common)
            if common
            else None,
        }
        for candidate in ids
    ]
    winner = (
        min(results, key=lambda row: (-row["mean_monthly_ic"], row["candidate_id"]))["candidate_id"]
        if common
        else None
    )
    return {
        "common_dates": common,
        "candidates": results,
        "selected_candidate_id": winner,
        "status": "development_selected_not_final_recipe"
        if common
        else "no_common_resolved_support",
    }


def _fit(candidate, batch, threads, target_policy):
    import numpy as np

    from .training_targets import training_targets

    if not batch.ready:
        raise ValueError("Refuse fitting insufficient observed support")
    if candidate.kind == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(**dict(candidate.parameters))
    elif candidate.kind == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(**dict(candidate.parameters))
    else:
        raise ValueError("Fixed control has no fitted estimator")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        model.fit(
            np.asarray([r.values for r in batch.feature_rows], dtype=float),
            np.asarray(training_targets(batch, target_policy), dtype=float),
            sample_weight=np.asarray(batch.weights, dtype=float),
        )
    return model


def run_stock_development(
    prepared_root,
    expected_manifest_sha256,
    spec_path,
    schedule,
    output,
    *,
    resume_checkpoint=None,
    expected_resume_sha256=None,
    service_deadline_monotonic=None,
):
    """Execute the registered full family only in Kaggle; never a local toy fit.

    Scores and model/fit receipts are sealed before current outcomes are joined.
    Financial execution is a later shared-engine stage; these are target intents.
    """
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise ValueError("Full model fits are restricted to the reviewed Kaggle runtime")

    def budget_check():
        if service_deadline_monotonic is not None:
            if not math.isfinite(service_deadline_monotonic):
                raise ValueError("Service deadline must be finite")
            if time.monotonic() >= service_deadline_monotonic:
                raise TimeoutError("Service budget exhausted; complete cells remain resumable")

    budget_check()
    spec = load_spec(spec_path)
    spec_sha = file_sha256(Path(spec_path))
    data = read_development_inputs(prepared_root, expected_manifest_sha256, spec, schedule)
    if not data.full_research_scope:
        raise ValueError("Full observed prepared population required before fitting")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    candidates = registered_candidates(spec)
    from threadpoolctl import threadpool_limits

    runtime = {
        name: version(name)
        for name in ("numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow", "threadpoolctl")
    }
    runtime["python"] = platform.python_version()
    runtime["platform"] = platform.platform()
    runtime["processor"] = platform.processor()
    closure = development_source_closure()
    code_hash = _digest(closure)
    root = Path(output).resolve()
    if root == Path(prepared_root).resolve() or Path(prepared_root).resolve() in root.parents:
        raise ValueError("Development output must be outside immutable preparation")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "stock_prequential_development_v1",
        "status": "staging",
        "complete": False,
        "source_prepared_manifest_sha256": expected_manifest_sha256,
        "spec_sha256": spec_sha,
        "spec": spec,
        "source_code_sha256": code_hash,
        "source_closure_sha256": closure,
        "runtime": runtime,
        "candidate_ids": [c.candidate_id for c in candidates],
        "selection_cutoff": data.selection_cutoff,
        "outcome_value_partitions_read": data.outcome_value_partitions_read,
        "outcome_value_partitions_refused": data.outcome_value_partitions_refused,
        "economic_evaluation": "not_run_saved_intentions_are_not_fills",
        "uncertainty": "not_run_by_this_prediction_runner",
        "final_holdout_evaluation": False,
        "resume_checkpoint_sha256": expected_resume_sha256,
        "resume_checkpoint_path": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "artifacts": [],
        "errors": [],
    }
    metrics, fit_records, targets = [], [], []
    pins = {
        "prepared_sha256": expected_manifest_sha256,
        "spec_sha256": spec_sha,
        "source_closure": closure,
        "runtime": runtime,
        "candidate_ids": manifest["candidate_ids"],
    }
    completed_cells = []
    current_day, current_candidate = None, None
    try:
        _write(root / "config.json", manifest)
        if (resume_checkpoint is None) != (expected_resume_sha256 is None):
            raise ValueError("Resume requires checkpoint path and external SHA together")
        if resume_checkpoint is not None:
            completed_cells = restore_checkpoint(
                resume_checkpoint, expected_resume_sha256, root, pins
            )
        restored = {
            (cell["signal_session"], cell["candidate_id"]): cell for cell in completed_cells
        }
        by_date = {
            day: tuple(r for r in data.features if r.signal_session == day)
            for day, _ in data.decision_times
        }
        labels = {
            day: tuple(r for r in data.outcomes if r.signal_session == day)
            for day, _ in data.decision_times
        }
        validation = [
            (date.fromisoformat(w["signal_start"]), date.fromisoformat(w["signal_end"]))
            for w in spec["validation_windows"]
        ]
        for day, cutoff in data.decision_times:
            if not any(start <= day <= end for start, end in validation):
                continue
            current_day = day
            batch = training_batch(data, day, cutoff, spec)
            current = by_date[day]
            predictions, statuses = [], {}
            for candidate in candidates:
                budget_check()
                current_candidate = candidate.candidate_id
                resumed = restored.get((day.isoformat(), candidate.candidate_id))
                if resumed is not None:
                    roles = {
                        resumed[role] for role in ("fit_path", "prediction_path", "target_path")
                    }
                    if not roles <= {item["path"] for item in resumed["artifacts"]}:
                        raise ValueError("Resumed cell semantic role lacks bound artifact")
                    record = json.loads((root / resumed["fit_path"]).read_text())
                    expected_fit_id = _digest(
                        {
                            "candidate": asdict(candidate),
                            "training": batch.input_sha256,
                            "signal_session": day,
                            "spec_sha256": spec_sha,
                            "source_code_sha256": code_hash,
                            "runtime": runtime,
                        }
                    )
                    expected_status = (
                        "no_current_feature_rows"
                        if not current
                        else "fixed_control_no_fit"
                        if candidate.kind == "control"
                        else "training_support_unavailable"
                        if not batch.ready
                        else "fitted"
                    )
                    if (
                        record["training_input_sha256"] != batch.input_sha256
                        or record["candidate_id"] != candidate.candidate_id
                        or record["signal_session"] != day.isoformat()
                        or record["information_cutoff"] != cutoff.isoformat()
                        or record["fit_id"] != expected_fit_id
                        or record["status"] != expected_status
                    ):
                        raise ValueError("Resumed fit semantics or training input changed")
                    if expected_status == "fitted":
                        expected_model_path = f"models/{day}/{candidate.candidate_id}." + (
                            "json" if candidate.kind == "ridge" else "txt"
                        )
                        if (
                            record.get("model_path") != expected_model_path
                            or expected_model_path not in {a["path"] for a in resumed["artifacts"]}
                            or record.get("model_sha256") != file_sha256(root / expected_model_path)
                        ):
                            raise ValueError("Resumed native model semantic role/hash mismatch")
                    saved_predictions = _read_columns(root / resumed["prediction_path"], None)
                    indexed_current = {r.asset_id: r for r in current}
                    prediction_ids = [r["asset_id"] for r in saved_predictions]
                    expected_ids = (
                        set(indexed_current)
                        if expected_status in ("fixed_control_no_fit", "fitted")
                        else set()
                    )
                    if (
                        len(set(prediction_ids)) != len(prediction_ids)
                        or set(prediction_ids) != expected_ids
                    ):
                        raise ValueError("Resumed prediction population mismatch")
                    for prediction in saved_predictions:
                        feature = indexed_current[prediction["asset_id"]]
                        if (
                            prediction["candidate_id"] != candidate.candidate_id
                            or prediction["fit_id"] != expected_fit_id
                            or prediction["signal_session"] != day
                            or prediction["signal_at"] != cutoff
                            or prediction["score_kind"]
                            != prediction_score_kind(candidate, spec)
                            or prediction["feature_input_sha256"] != feature.input_rows_sha256
                            or not math.isfinite(prediction["score"])
                        ):
                            raise ValueError("Resumed prediction fit/key/provenance mismatch")
                    saved_targets = json.loads((root / resumed["target_path"]).read_text())
                    expected_targets = []
                    if saved_predictions:
                        expected_targets = [
                            {
                                "candidate_id": candidate.candidate_id,
                                "signal_session": day,
                                **asdict(
                                    target_from_scores(
                                        tuple(
                                            (r["asset_id"], r["score"]) for r in saved_predictions
                                        ),
                                        cutoff,
                                        current[0].intended_execution_at,
                                        spec,
                                    )
                                ),
                            }
                        ]
                    if _raw(saved_targets) != _raw(expected_targets):
                        raise ValueError("Resumed target does not match saved predictions")
                    fit_records.append(record)
                    statuses[candidate.candidate_id] = record["status"]
                    predictions.extend(saved_predictions)
                    targets.extend(saved_targets)
                    continue
                candidate_predictions, candidate_targets = [], []
                fit_id = _digest(
                    {
                        "candidate": asdict(candidate),
                        "training": batch.input_sha256,
                        "signal_session": day,
                        "spec_sha256": spec_sha,
                        "source_code_sha256": code_hash,
                        "runtime": runtime,
                    }
                )
                record = {
                    "candidate_id": candidate.candidate_id,
                    "signal_session": day,
                    "information_cutoff": cutoff,
                    "fit_id": fit_id,
                    "training_input_sha256": batch.input_sha256,
                    "training_target_transform": spec["target_transform"],
                    "parameters": candidate.parameters,
                    "training_rows": len(batch.feature_rows),
                    "training_dates": len(batch.date_counts),
                    "date_counts": batch.date_counts,
                    "max_label_available_at": batch.max_label_available_at,
                    "weight_sum": math.fsum(batch.weights),
                    "weight_mean": 1.0 if batch.weights else None,
                    "objective": spec["ridge"]["objective"]
                    if candidate.kind == "ridge"
                    else candidate.kind,
                    "actual_fit_started_at": datetime.now(UTC),
                }
                if not current:
                    record["status"] = "no_current_feature_rows"
                    scores = ()
                elif candidate.kind == "control":
                    scores = control_scores(candidate, current)
                    record["status"] = "fixed_control_no_fit"
                elif not batch.ready:
                    scores = ()
                    record.update(status="training_support_unavailable", reasons=batch.reasons)
                else:
                    model = _fit(candidate, batch, spec["threads"], spec["target_transform"])
                    with threadpool_limits(limits=spec["threads"]):
                        values = model.predict(np.asarray([r.values for r in current], dtype=float))
                    scores = tuple(
                        (r.asset_id, float(v)) for r, v in zip(current, values, strict=True)
                    )
                    model_path = (
                        root / "models" / day.isoformat() / (candidate.candidate_id + ".json")
                    )
                    if candidate.kind == "ridge":
                        _write(
                            model_path,
                            {
                                "coefficients": model.coef_.tolist(),
                                "intercept": float(model.intercept_),
                                "feature_ids": FEATURE_IDS,
                            },
                        )
                    else:
                        model_path = model_path.with_suffix(".txt")
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        model.booster_.save_model(str(model_path))
                    record.update(
                        status="fitted",
                        model_path=model_path.relative_to(root).as_posix(),
                        model_sha256=file_sha256(model_path),
                    )
                record["actual_fit_finished_at"] = datetime.now(UTC)
                fit_records.append(record)
                _write(root / "fits" / day.isoformat() / (candidate.candidate_id + ".json"), record)
                statuses[candidate.candidate_id] = record["status"]
                if any(not math.isfinite(value) for _, value in scores):
                    raise ValueError("Nonfinite model score; no substitute or candidate omission")
                for asset, score in scores:
                    feature = next(r for r in current if r.asset_id == asset)
                    candidate_predictions.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "asset_id": asset,
                            "signal_session": day,
                            "signal_at": cutoff,
                            "score": score,
                            "score_kind": prediction_score_kind(candidate, spec),
                            "fit_id": fit_id,
                            "feature_input_sha256": feature.input_rows_sha256,
                        }
                    )
                if scores:
                    intent = target_from_scores(
                        scores, cutoff, current[0].intended_execution_at, spec
                    )
                    candidate_targets.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "signal_session": day,
                            **asdict(intent),
                        }
                    )
                # Seal this complete candidate/date before any outcome join.
                cell_path = root / "cells" / day.isoformat() / (candidate.candidate_id + ".parquet")
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                cell_schema = pa.schema(
                    [
                        ("candidate_id", pa.string()),
                        ("asset_id", pa.string()),
                        ("signal_session", pa.date32()),
                        ("signal_at", pa.timestamp("us", tz="UTC")),
                        ("score", pa.float64()),
                        ("score_kind", pa.string()),
                        ("fit_id", pa.string()),
                        ("feature_input_sha256", pa.string()),
                    ]
                )
                pending = cell_path.with_suffix(".pending")
                pq.write_table(
                    pa.Table.from_pylist(candidate_predictions, schema=cell_schema),
                    pending,
                    compression="zstd",
                )
                pending.rename(cell_path)
                fit_path = root / "fits" / day.isoformat() / (candidate.candidate_id + ".json")
                target_path = (
                    root / "intents" / day.isoformat() / (candidate.candidate_id + ".json")
                )
                _write(target_path, candidate_targets)
                cell_files = [cell_path, fit_path, target_path]
                if record.get("model_path"):
                    cell_files.append(root / record["model_path"])
                completed_cells.append(
                    {
                        "signal_session": day.isoformat(),
                        "candidate_id": candidate.candidate_id,
                        "fit_path": fit_path.relative_to(root).as_posix(),
                        "prediction_path": cell_path.relative_to(root).as_posix(),
                        "target_path": target_path.relative_to(root).as_posix(),
                        "artifacts": [
                            {"path": p.relative_to(root).as_posix(), "sha256": file_sha256(p)}
                            for p in cell_files
                        ],
                    }
                )
                _write(
                    root / "checkpoints" / (f"{len(completed_cells):06d}.json"),
                    {
                        "schema": "stock_development_checkpoint_v1",
                        "complete": False,
                        "pins": pins,
                        "cells": [completed_cells[-1]],
                        "parent_checkpoint": {
                            "path": f"checkpoints/{len(completed_cells) - 1:06d}.json",
                            "sha256": file_sha256(
                                root / "checkpoints" / f"{len(completed_cells) - 1:06d}.json"
                            ),
                        }
                        if len(completed_cells) > 1
                        else None,
                    },
                )
                predictions.extend(candidate_predictions)
                targets.extend(candidate_targets)
            path = root / "predictions" / (day.isoformat() + ".parquet")
            path.parent.mkdir(exist_ok=True)
            schema = pa.schema(
                [
                    ("candidate_id", pa.string()),
                    ("asset_id", pa.string()),
                    ("signal_session", pa.date32()),
                    ("signal_at", pa.timestamp("us", tz="UTC")),
                    ("score", pa.float64()),
                    ("score_kind", pa.string()),
                    ("fit_id", pa.string()),
                    ("feature_input_sha256", pa.string()),
                ]
            )
            pq.write_table(
                pa.Table.from_pylist(predictions, schema=schema), path, compression="zstd"
            )
            prediction_sha = file_sha256(path)
            # Only the now-sealed prediction artifact is passed to evaluation.
            saved = _read_columns(path, schema.names)
            for candidate in candidates:
                scores = tuple(
                    (r["asset_id"], r["score"])
                    for r in saved
                    if r["candidate_id"] == candidate.candidate_id
                )
                measured = date_ic(
                    scores, labels[day], minimum_support=spec["minimum_scored_cross_section"]
                )
                metrics.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "signal_session": day,
                        "prediction_sha256": prediction_sha,
                        "fit_status": statuses[candidate.candidate_id],
                        **measured,
                    }
                )
        require_completed_grid(
            completed_cells,
            [
                day
                for day, _ in data.decision_times
                if any(start <= day <= end for start, end in validation)
            ],
            candidates,
        )
        budget_check()
        _write(root / "metrics.json", metrics)
        _write(root / "target-intentions.json", targets)
        _write(root / "fit-index.json", fit_records)
        summary = common_paired_summary(metrics, candidates)
        final_closure = development_source_closure()
        if file_sha256(Path(spec_path)) != spec_sha or final_closure != closure:
            raise ValueError("Development source/spec changed during run")
        verify_prepared_panel(prepared_root, expected_manifest_sha256)
        budget_check()
        _write(root / "development-selection.json", summary)
        manifest.update(
            status="complete",
            complete=True,
            selected_candidate_id=summary["selected_candidate_id"],
            comparison_status=summary["status"],
            finished_at=datetime.now(UTC),
            artifacts=[
                {
                    "path": p.relative_to(root).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ],
        )
        _write(root / "manifest.json", manifest)
        digest = file_sha256(root / "manifest.json")
        with (root / "COMPLETE").open("xb") as handle:
            handle.write((digest + "\n").encode())
        return {**manifest, "manifest_sha256": digest}
    except BaseException as error:
        _write(
            root / "FAILURE.json",
            {
                "stage": "prequential_development",
                "signal_session": current_day,
                "candidate_id": current_candidate,
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_fit_records": fit_records,
                "complete": False,
            },
        )
        raise
