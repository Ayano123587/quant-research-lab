"""Concrete providers for an externally frozen shared evaluation.

Construction neither selects a recipe nor opens outcomes. The release gate must
authenticate these sources, recipe bytes and provider input files before use.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

from quantlab.allocation.policies import allocate
from quantlab.backtest.engine import Decision
from quantlab.evaluation.holdout import FitReceipt, LabelClock, LoadedHistory, file_sha
from quantlab.stocks.development import (
    FeatureRow,
    Outcome,
    TrainingBatch,
    _fit,
    control_scores,
    target_from_scores,
)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class ArchiveProvider:
    """Read authenticated real histories after the external release gate."""

    def __init__(self, path, expected_sha256, assets, source_identity):
        self.path, self.sha = path, expected_sha256
        self.assets, self.identity = tuple(assets), source_identity
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError("Explicit unique (asset_type, symbol) source collection required")

    def load(self):
        import hashlib

        from quantlab.data.archive import MarketArchive
        from quantlab.evaluation.holdout import EmptySourceMember

        histories, empty = {}, []
        with MarketArchive(self.path, self.sha) as source:
            for kind, symbol in self.assets:
                bars = source.bars(kind, symbol)
                asset = kind + ":" + symbol.lower().removesuffix(".us") + ".us"
                if asset in histories or any(b.asset_id != asset for b in bars):
                    raise ValueError("Repeated or mismatched authentic source identity")
                # A genuinely empty source member remains part of the population.
                histories[asset] = bars
                if not bars:
                    member, raw = source.source_bytes(kind, symbol)
                    empty.append(EmptySourceMember(asset, member, hashlib.sha256(raw).hexdigest()))
        return LoadedHistory(
            histories, self.identity, tuple(sorted(r.asset_id for r in empty)), tuple(empty)
        )


class SnapshotProvider:
    """Actual authenticated BIN snapshot, including genuine empty stock members."""

    def __init__(self, root, expected_manifest_sha256, assets, schedule, expected_identity):
        self.root, self.sha, self.assets = root, expected_manifest_sha256, tuple(assets)
        self.schedule, self.identity = schedule, expected_identity

    def load(self):
        from quantlab.data.provenance import identity_from_source
        from quantlab.data.snapshot import MarketSnapshot
        from quantlab.evaluation.holdout import EmptySourceMember

        source = MarketSnapshot(self.root, self.sha)
        identity = identity_from_source(source, self.schedule)
        if identity != self.identity:
            raise ValueError("Actual BIN source identity differs")
        self.source = source
        assets = {
            kind + ":" + symbol.lower().removesuffix(".us") + ".us" for kind, symbol in self.assets
        }
        if len(assets) != len(self.assets):
            raise ValueError("Unique BIN source identities required")
        if all(kind == "stock" for kind, _ in self.assets):
            if assets != {a for a in source.asset_ids if a.startswith("stock:")}:
                raise ValueError(
                    "All7195 original stock members, including empty members, required"
                )
        histories, empty = {}, []
        with source.verified_read_session():
            for kind, symbol in self.assets:
                asset = kind + ":" + symbol.lower().removesuffix(".us") + ".us"
                histories[asset] = source.bars(kind, symbol)
                entry = source._assets[asset]
                if not histories[asset]:
                    if entry["rows"] != 0:
                        raise ValueError("BIN source empty declaration differs from actual data")
                    empty.append(
                        EmptySourceMember(asset, entry["source_member"], entry["source_sha256"])
                    )
        return LoadedHistory(
            histories, identity, tuple(sorted(r.asset_id for r in empty)), tuple(empty)
        )


class YahooETFProvider:
    """Preserve the separate authenticated provider price-index source identity."""

    def __init__(self, root, expected_manifest_sha256, schedule):
        self.root, self.sha, self.schedule = root, expected_manifest_sha256, schedule

    def load(self):
        from quantlab.allocation.policies import SYMBOLS
        from quantlab.data.provenance import identity_from_source
        from quantlab.data.yahoo import YahooETFRelease

        source = YahooETFRelease(self.root, self.sha, self.schedule)
        histories = {}
        for symbol in SYMBOLS:
            bars = source.bars("etf", symbol)
            if not bars:
                raise ValueError("Required authentic ETF sleeve missing")
            histories[bars[0].asset_id] = bars
        return LoadedHistory(histories, identity_from_source(source, self.schedule))


class CovarianceForecastProvider:
    """Past-only daily forecasts for a prescribed panel and shared ex-ante q.

    The caller supplies the actual dated asset order/reference policy, never an
    outcome-ranked universe. No future proxy is constructed by this provider.
    """

    def __init__(self, schedule, source_identity, *, observations, estimator_parameters):
        if observations not in (63, 126, 252) or not estimator_parameters:
            raise ValueError("Explicit registered window and estimators required")
        self.schedule, self.identity = schedule, source_identity
        self.count, self.parameters = observations, dict(estimator_parameters)
        self.forecasts, self.references = {}, {}

    def forecast(self, view, reference):
        import numpy as np

        from quantlab.risk.covariance import ReturnWindow, estimate_covariance

        if (
            reference.known_at > view.cutoff
            or reference.max_input_available_at > view.cutoff
            or view.session in self.forecasts
        ):
            raise ValueError("One immutable past-only reference/forecast per signal required")
        end = self.schedule.sessions.index(view.session)
        days = self.schedule.sessions[end - self.count : end + 1]
        if len(days) != self.count + 1:
            raise ValueError("Insufficient exact scheduled training window")
        histories = dict(view.histories)
        prices, row_ids = [], []
        for asset in reference.asset_ids:
            indexed = {b.session: b for b in histories[asset]}
            bars = tuple(indexed.get(d) for d in days)
            if any(
                b is None
                or b.quality_flags
                or b.session > view.session
                or not math.isfinite(b.close)
                or b.close <= 0
                for b in bars
            ):
                raise ValueError("Missing/unusable scheduled covariance mark; no gap repair")
            prices.append([b.close for b in bars])
            row_ids.append([(b.source_sha256, b.row_sha256) for b in bars])
        values = np.asarray(prices).T
        provenance = {
            **self.identity.as_provenance(),
            "return_convention": "source_close_to_close_adjustment_unverified",
            "source_snapshot_id": self.identity.source_id,
            "source_rows_sha256": _identity(row_ids),
        }
        window = ReturnWindow(
            values[1:] / values[:-1] - 1,
            reference.asset_ids,
            tuple(self.schedule.session(d).close_at for d in days[:-1]),
            tuple(self.schedule.session(d).close_at for d in days[1:]),
            tuple(self.schedule.session(d).decision_at for d in days[1:]),
            view.cutoff,
            self.identity.source_id,
            provenance["return_convention"],
            "frozen_reference_panel_no_future_selection",
            _identity(row_ids),
            provenance,
        )
        required = {
            "method",
            "min_observations",
            "ewma_lambda",
            "n_factors",
            "residual_floor",
            "constant_correlation_intensity",
            "relative_tolerance",
            "absolute_tolerance",
        }
        estimates = {}
        for name, parameters in self.parameters.items():
            if set(parameters) != required or parameters["min_observations"] != self.count:
                raise ValueError("Every numerical estimator parameter must be explicitly frozen")
            estimates[name] = estimate_covariance(window, **parameters)
        self.forecasts[view.session] = estimates
        self.references[view.session] = reference
        return estimates


class ETFProvider:
    """The unchanged eight-sleeve controls, on scheduled signals only."""

    def __init__(self, schedule, source_identity, signal_sessions, *, policy, covariance_method):
        self.schedule = schedule
        self.identity = source_identity
        self.signals = frozenset(signal_sessions)
        self.policy = policy
        self.method = covariance_method
        self.targets = []

    def decide(self, view, state):
        if view.session not in self.signals:
            return Decision("HOLD", None, view.cutoff, view.next_open_at, "between_frozen_signals")
        target = allocate(
            view,
            self.schedule,
            policy=self.policy,
            source_identity=self.identity,
            covariance_method=self.method,
        )
        self.targets.append(target)
        return target.decision


class RiskPortfolioProvider:
    """Use unchanged solver/projection/accounting targets on a frozen dated panel.

    Dated references must come from the authenticated past-only universe stage;
    no future coverage or realized return is consulted to replace a panel.
    """

    def __init__(
        self,
        forecast_provider,
        references_by_session,
        *,
        economic_arm,
        cap,
        solver_policy,
        projection_path,
        projection_sha256,
        sample_reference_id,
    ):
        self.provider = forecast_provider
        self.references = dict(references_by_session)
        self.arm, self.cap, self.solver_policy = economic_arm, cap, solver_policy
        self.projection_path, self.projection_sha = projection_path, projection_sha256
        self.sample_id = sample_reference_id
        self.decisions = []

    def decide(self, view, state):
        from quantlab.risk.economic_study import decide

        if view.session not in self.references:
            return Decision("HOLD", None, view.cutoff, view.next_open_at, "between_frozen_signals")
        estimates = self.provider.forecast(view, self.references[view.session])
        if self.arm.recipe_id not in estimates or self.sample_id not in estimates:
            raise ValueError("Frozen selected and sample reference estimators both required")
        value, audit = decide(
            view,
            state,
            self.arm,
            estimates[self.arm.recipe_id],
            estimates[self.sample_id],
            self.cap,
            self.solver_policy,
            self.projection_path,
            self.projection_sha,
        )
        self.decisions.append({"session": view.session, "decision": value, "audit": audit})
        return value


class SensitivityProvider:
    """Concrete adapter for every unchanged registered one-change ETF scenario."""

    def __init__(self, schedule, source_identity, signal_sessions, *, scenario):
        from quantlab.allocation.sensitivity import SCENARIOS

        if scenario not in SCENARIOS:
            raise ValueError("An explicit existing registered scenario is required")
        self.schedule, self.identity = schedule, source_identity
        self.signals, self.scenario = frozenset(signal_sessions), scenario
        self.targets = []

    def decide(self, view, state):
        from quantlab.allocation.sensitivity import sensitivity_target

        if view.session not in self.signals:
            return Decision("HOLD", None, view.cutoff, view.next_open_at, "between_frozen_signals")
        target = sensitivity_target(view, self.schedule, self.identity, self.scenario)
        self.targets.append(target)
        return target.decision


class StockPanelProvider:
    """Clock-first prepared Parquet reads and unchanged registered learners.

    Each partition is externally pinned before construction. Value columns are
    opened only for the exact keys approved by the gate's maturity selector.
    Feature names are explicit: this also supports a frozen wider projection,
    provided its prepared columns were authenticated by the release publisher.
    """

    def __init__(
        self,
        root,
        partitions,
        schedule,
        signal_sessions,
        *,
        candidate,
        feature_columns,
        raw_columns,
        target_policy,
        threads,
        target_spec,
        minimum_rows,
        minimum_dates,
    ):
        self.root = Path(root).resolve()
        self.partitions = {}
        self.partition_identity = {}
        for item in partitions:
            key = (item["kind"], date.fromisoformat(item["session"]))
            path = self.root / item["path"]
            if (
                key in self.partitions
                or path.is_symlink()
                or not path.resolve().is_relative_to(self.root)
                or path.stat().st_size != item["bytes"]
                or file_sha(path) != item["sha256"]
            ):
                raise ValueError("Prepared partition identity differs")
            self.partitions[key] = path
            self.partition_identity[path] = (item["bytes"], item["sha256"])
        if not feature_columns or not raw_columns:
            raise ValueError("Explicit model projection and original raw columns required")
        if minimum_rows < 1 or minimum_dates < 1 or threads < 1:
            raise ValueError("Explicit positive support and thread policy required")
        self.schedule, self.signals = schedule, frozenset(signal_sessions)
        self.candidate, self.columns, self.raw_columns = (
            candidate,
            tuple(feature_columns),
            tuple(raw_columns),
        )
        self.target_policy, self.threads, self.spec = target_policy, threads, target_spec
        self.minimum_rows, self.minimum_dates = minimum_rows, minimum_dates
        self.model = None
        self.predictions = []

    def _read(self, path, columns):
        import pyarrow.parquet as pq

        size, digest = self.partition_identity[path]
        if path.is_symlink() or path.stat().st_size != size or file_sha(path) != digest:
            raise ValueError("Prepared partition changed before projected read")
        return pq.ParquetFile(path).read(columns=columns).to_pylist()

    def _features(self, day):
        signal = self.schedule.session(day).decision_at
        records = self._read(
            self.partitions["features", day],
            [
                "asset_id",
                "signal_session",
                "signal_at",
                "intended_execution_at",
                "max_input_available_at",
                "input_rows_sha256",
                *self.columns,
                *self.raw_columns,
            ],
        )
        rows = []
        for row in records:
            values = tuple(float(row[c]) for c in self.columns)
            raw = tuple(float(row[c]) for c in self.raw_columns)
            if (
                row["signal_session"] != day
                or row["signal_at"] != signal
                or row["max_input_available_at"] > signal
                or row["intended_execution_at"] != self.schedule.next_session(day).open_at
                or not all(math.isfinite(v) for v in (*values, *raw))
            ):
                raise ValueError("Prepared feature timing/value contract differs")
            rows.append(
                FeatureRow(
                    row["asset_id"],
                    day,
                    signal,
                    row["intended_execution_at"],
                    values,
                    raw,
                    row["input_rows_sha256"],
                )
            )
        if len({r.asset_id for r in rows}) != len(rows):
            raise ValueError("Duplicate prepared feature key")
        return tuple(sorted(rows, key=lambda r: r.asset_id))

    def _feature_keys(self, day):
        rows = self._read(
            self.partitions["features", day],
            [
                "asset_id",
                "signal_session",
                "signal_at",
                "max_input_available_at",
                "intended_execution_at",
            ],
        )
        signal = self.schedule.session(day).decision_at
        ids = set()
        for row in rows:
            if (
                row["asset_id"] in ids
                or row["signal_session"] != day
                or row["signal_at"] != signal
                or row["max_input_available_at"] > signal
                or row["intended_execution_at"] != self.schedule.next_session(day).open_at
            ):
                raise ValueError("Original feature membership/timing differs")
            ids.add(row["asset_id"])
        return ids

    def clocks(self, view):
        result = []
        for (kind, day), path in sorted(self.partitions.items()):
            if kind != "labels" or day >= view.session:
                continue
            # Membership was fixed by past-only feature eligibility/activity.
            # The source label table also contains names outside that cohort.
            feature_ids = self._feature_keys(day)
            # Deliberately no return_value or price columns in this projection.
            for row in self._read(
                path,
                [
                    "asset_id",
                    "signal_session",
                    "label_available_at",
                    "planned_available_at",
                    "status",
                    "input_rows_sha256",
                ],
            ):
                if row["signal_session"] != day:
                    raise ValueError("Label partition date differs")
                if row["asset_id"] not in feature_ids:
                    continue
                available = row["label_available_at"] or row["planned_available_at"]
                if available is None or (
                    row["status"] == "valid" and row["label_available_at"] is None
                ):
                    raise ValueError("Explicit label maturity clock required")
                result.append(
                    LabelClock(
                        row["asset_id"], day, available, row["status"], row["input_rows_sha256"]
                    )
                )
        return tuple(result)

    def fit(self, view, selected):
        if not str(Path.cwd()).replace("\\", "/").startswith("/kaggle/working"):
            raise ValueError("Actual learner fitting is cloud-only")
        keys = set(selected.keys)
        if len(keys) != len(selected.keys):
            raise ValueError("Unique approved training keys required")
        features, outcomes = {}, {}
        for day in sorted({d for d, _ in keys}):
            if day >= view.session:
                raise ValueError("Training signal must precede prediction")
            features.update(
                {(day, r.asset_id): r for r in self._features(day) if (day, r.asset_id) in keys}
            )
            # Parquet projection reads a complete column. Authenticate the whole
            # partition's maturity before opening any outcome values, including
            # unselected rows in that column.
            metadata = self._read(
                self.partitions["labels", day],
                [
                    "planned_available_at",
                    "label_available_at",
                ],
            )
            if any(
                (
                    r["planned_available_at"] is None
                    or r["planned_available_at"] > view.cutoff
                    or (
                        r["label_available_at"] is not None
                        and r["label_available_at"] > view.cutoff
                    )
                )
                for r in metadata
            ):
                raise ValueError("Entire value partition must be mature before projection")
            rows = self._read(
                self.partitions["labels", day],
                [
                    "asset_id",
                    "signal_session",
                    "label_available_at",
                    "status",
                    "return_value",
                    "input_rows_sha256",
                ],
            )
            for row in rows:
                key = (day, row["asset_id"])
                if key not in keys:
                    continue
                if (
                    key in outcomes
                    or row["signal_session"] != day
                    or row["status"] != "valid"
                    or row["label_available_at"] > view.cutoff
                    or not math.isfinite(row["return_value"])
                ):
                    raise ValueError("Approved mature outcome changed")
                outcomes[key] = Outcome(
                    row["asset_id"],
                    day,
                    row["label_available_at"],
                    row["return_value"],
                    row["input_rows_sha256"],
                )
        if set(features) != keys or set(outcomes) != keys:
            raise ValueError("Missing approved training keys cannot be silently dropped")
        counts = Counter(d for d, _ in selected.keys)
        if len(keys) < self.minimum_rows or len(counts) < self.minimum_dates:
            raise ValueError("Frozen minimum training support unavailable")
        maximum = max(outcomes[k].available_at for k in selected.keys)
        identity = _identity([(k, features[k], outcomes[k]) for k in selected.keys])
        batch = TrainingBatch(
            tuple(features[k] for k in selected.keys),
            tuple(outcomes[k] for k in selected.keys),
            selected.sample_weights,
            view.cutoff,
            tuple(sorted(counts.items())),
            maximum,
            identity,
            True,
            (),
        )
        self.model = _fit(self.candidate, batch, self.threads, self.target_policy)
        self.fit_id = identity
        return FitReceipt(
            identity, view.cutoff, selected.keys, max(r.signal_at for r in batch.feature_rows)
        )

    def decide(self, view, state):
        if view.session not in self.signals:
            return Decision("HOLD", None, view.cutoff, view.next_open_at, "between_frozen_signals")
        if self.model is None and self.candidate.kind != "control":
            raise ValueError("No authenticated maturity-gated fitted model")
        rows = self._features(view.session)
        scores = (
            control_scores(self.candidate, rows)
            if self.candidate.kind == "control"
            else tuple(
                zip(
                    (r.asset_id for r in rows),
                    map(float, self.model.predict([r.values for r in rows])),
                    strict=True,
                )
            )
        )
        if not all(math.isfinite(v) for _, v in scores):
            raise ValueError("Nonfinite predictions")
        self.predictions.append(
            {
                "session": view.session,
                "signal_at": view.cutoff,
                "fit_id": getattr(self, "fit_id", "registered_unfitted_control"),
                "scores": scores,
            }
        )
        return target_from_scores(scores, view.cutoff, view.next_open_at, self.spec)

    def export_predictions(self, path):
        """Exclusive durable forecast snapshot, before outcome consumers run."""
        payload = {
            "schema": "shared_holdout_stock_predictions_v1",
            "candidate_id": self.candidate.candidate_id,
            "feature_columns": self.columns,
            "predictions": self.predictions,
            "contains_outcomes": False,
        }
        raw = json.dumps(payload, sort_keys=True, default=str, allow_nan=False).encode()
        with Path(path).open("xb") as stream:
            stream.write(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
