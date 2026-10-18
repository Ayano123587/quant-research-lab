"""Released final features and strictly mature labels using original formulas.

No public entrypoint prepares final data before the actual single-dispatch gate.
Low-level record helpers also support unchanged development-record comparisons.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quantlab.evaluation.holdout import START, LabelClock, file_sha


def validate_parent_partitions(spec):
    """Original container authentication permits future bytes, never future values."""
    for collection in ("partitions", "wider_partitions"):
        for row in spec.get(collection, ()):
            day = date.fromisoformat(row["session"])
            if day > START or Path(row["path"]).stem != day.isoformat():
                raise ValueError(
                    "Base feature/label/wider partitions must be explicitly filtered <=START"
                )


def label_timing(schedule, day):
    year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
    return schedule.next_session(day), schedule.next_session(
        schedule.month_end(year, month).session
    )


def feature_records(view, schedule, identity, inventory, snapshot_id, *, top_n):
    """Original candidate formulas and exact original activity/rank transform."""
    from quantlab.stocks.features import FEATURE_IDS, candidate_from_history
    from quantlab.stocks.prepare import _digest, _rank_rows

    if top_n != 500:
        raise ValueError("Original full eligible top500 rule required")
    histories = dict(view.histories)
    if set(histories) != set(inventory):
        raise ValueError("Every supplied original source identity must be preserved")
    end = schedule.sessions.index(view.session)
    if end < 252:
        raise ValueError("Actual253 scheduled mark window required")
    first = schedule.sessions[end - 252]
    candidates, audit = [], []
    for asset, history in sorted(histories.items()):
        if any(b.session > view.session for b in history):
            raise ValueError("Final feature builder refuses future source rows")
        bars = tuple(b for b in history if first <= b.session <= view.session)
        source = inventory[asset]
        candidate = candidate_from_history(
            asset, bars, schedule, as_of=view.session, archive_sha256=identity.archive_sha256
        )
        raw = dict(candidate.features)
        row = {
            "asset_id": asset,
            "signal_session": view.session,
            "signal_at": candidate.signal_at,
            "intended_execution_at": candidate.intended_execution_at,
            "eligible": candidate.eligible,
            "reasons": list(candidate.reasons),
            "activity_proxy": candidate.activity_proxy,
            "max_input_available_at": candidate.max_input_available_at,
            "observed_scheduled_marks": candidate.observed_scheduled_marks,
            "required_scheduled_marks": 253,
            "scheduled_window_start": first,
            "scheduled_window_end": view.session,
            "source_member": source["source_member"],
            "source_member_sha256": source["source_sha256"],
            "source_snapshot_id": snapshot_id,
            "input_rows_sha256": _digest(candidate.source_rows, source["source_sha256"]),
            **{"raw_" + feature: raw.get(feature) for feature in FEATURE_IDS},
        }
        candidates.append(row)
        audit.append(
            {
                "asset_id": asset,
                "eligible": candidate.eligible,
                "reasons": candidate.reasons,
                "input_rows_sha256": row["input_rows_sha256"],
            }
        )
    return _rank_rows(candidates, top_n), audit


def label_records(histories, schedule, identity, inventory, snapshot_id, *, day, endpoint):
    """Unchanged exact-open label formula; caller authenticates export/maturity first."""
    from quantlab.stocks.labels import label_from_history
    from quantlab.stocks.prepare import _digest

    entry, exit_time = label_timing(schedule, day)
    records = []
    for asset, history in sorted(histories.items()):
        source = inventory[asset]
        indexed = {b.session: b for b in history}
        marks = tuple(
            indexed[d] for d in (entry.session, exit_time.session) if d <= endpoint and d in indexed
        )
        label = label_from_history(
            asset,
            marks,
            schedule,
            signal_session=day,
            archive_endpoint=endpoint,
            archive_sha256=identity.archive_sha256,
        )
        row = {
            name: getattr(label, name)
            for name in (
                "asset_id",
                "signal_session",
                "signal_at",
                "entry_session",
                "entry_at",
                "exit_session",
                "exit_at",
                "planned_available_at",
                "label_available_at",
                "entry_mark",
                "exit_mark",
                "return_value",
                "status",
                "source_status",
            )
        }
        row.update(
            reasons=list(label.reasons),
            quality_flags=list(label.quality_flags),
            source_member=source["source_member"],
            source_member_sha256=source["source_sha256"],
            source_snapshot_id=snapshot_id,
            input_rows_sha256=_digest(marks, source["source_sha256"]),
        )
        for side in ("entry", "exit"):
            mark = getattr(label, side + "_row")
            row[side + "_source_row_number"] = mark.row_number if mark else None
            row[side + "_source_row_sha256"] = mark.row_sha256 if mark else None
        records.append(row)
    return records


def wider_records(original, view, schedule, identity):
    """Extend only the fixed original cohort with unchanged optional formulas."""
    import struct

    from quantlab.stocks.features import FEATURE_IDS as SIX
    from quantlab.stocks.features import candidate_from_history
    from quantlab.stocks.prepare import _digest
    from quantlab.stocks.wider_features import extend_candidate
    from quantlab.stocks.wider_panel import rank_added

    histories, result = dict(view.histories), []
    first = schedule.sessions[schedule.sessions.index(view.session) - 252]
    for original_row in original:
        row = dict(original_row)
        candidate = candidate_from_history(
            row["asset_id"],
            tuple(b for b in histories[row["asset_id"]] if first <= b.session <= view.session),
            schedule,
            as_of=view.session,
            archive_sha256=identity.archive_sha256,
        )
        if (
            not candidate.eligible
            or candidate.signal_at != row["signal_at"]
            or candidate.max_input_available_at != row["max_input_available_at"]
            or candidate.intended_execution_at != row["intended_execution_at"]
            or _digest(candidate.source_rows, row["source_member_sha256"])
            != row["input_rows_sha256"]
            or any(
                struct.pack("<d", dict(candidate.features)[f]) != struct.pack("<d", row["raw_" + f])
                for f in SIX
            )
        ):
            raise ValueError("Selected wider bridge changed original raw cohort/clocks/source")
        for value in extend_candidate(candidate).values:
            if value.feature_id not in SIX:
                row["raw_" + value.feature_id] = value.value
            row["missing_" + value.feature_id] = value.value is None
            row["reason_" + value.feature_id] = value.missing_reason
        result.append(row)
    rank_added(result)
    return result


class DeferredStockProvider:
    """Bind actual BIN source after release; derive hashes only after generation."""

    def __init__(self, spec, *, inputs, schedule, signals, release, release_sha256):
        validate_parent_partitions(spec)
        self.spec, self.inputs, self.schedule = spec, Path(inputs), schedule
        self.signals, self.release, self.release_sha = tuple(signals), release, release_sha256
        self.actual, self.generated, self.export_bindings = None, {}, None
        self.eligibility_artifacts = {}

    def __getattr__(self, name):
        actual = self.__dict__.get("actual")
        if actual is None:
            raise AttributeError(name)
        return getattr(actual, name)

    def bind_history(self, loaded, window, *, source_loader, export_root):
        import platform

        from quantlab.evaluation.holdout_registry import construct_stock_provider
        from quantlab.stocks.prepare import verify_prepared_panel

        validate_parent_partitions(self.spec)

        marker = (
            Path("/kaggle/working/shared-holdout") / self.release["release_id"] / "started.json"
        )
        if (
            platform.system() != "Linux"
            or not marker.is_file()
            or json.loads(marker.read_bytes())["release_sha256"] != self.release_sha
        ):
            raise ValueError("Actual consumed shared release required before final preparation")
        if not hasattr(source_loader, "source"):
            raise ValueError("Actual authenticated BIN source required for final stock preparation")
        self.loaded, self.window, self.source = loaded, window, source_loader.source
        if len(loaded.histories) != 7195 or set(loaded.histories) != {
            a for a in self.source.asset_ids if a.startswith("stock:")
        }:
            raise ValueError("Full original7195 identity population required")
        preparation = self.spec["final_preparation"]
        if (
            preparation["policy"]
            != "original_formulas_full7195_past_features_mature_labels_after_exports"
            or preparation["top_n"] != 500
        ):
            raise ValueError("Explicit original final-preparation policy required")
        base = self.inputs / self.spec["panel_root"]
        self.base_manifest = verify_prepared_panel(base, preparation["base_manifest_sha256"])
        if self.base_manifest["source_manifest_sha256"] != loaded.source_identity.manifest_sha256:
            raise ValueError("Original prepared/source BIN identities differ")
        if self.base_manifest["config"]["calendar"] != self.schedule.provenance:
            raise ValueError("Original and final preparation calendars differ")
        actual_spec = dict(self.spec)
        actual_spec.pop("final_preparation")
        self.actual = construct_stock_provider(
            actual_spec, inputs=self.inputs, schedule=self.schedule, signals=self.signals
        )
        self.root = Path(export_root).parent.parent / "derived-stock" / Path(export_root).name
        self.root.mkdir(parents=True, exist_ok=False)

    def bind_forecast_exports(self, bindings):
        self.export_bindings = bindings

    def _prior_exports(self, day):
        if self.export_bindings is None or not self.export_bindings:
            raise ValueError("All-arm prior forecast export register required before label values")
        for register in self.export_bindings.values():
            if day not in register:
                raise ValueError("Every arm must export this signal before its future label values")
            item = register[day]
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
                raise ValueError("Prior forecast export bytes changed before label values")

    def _table(self, kind, day, records, schema):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.root / kind / (str(day) + ".parquet")
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            raise ValueError("Generated final partitions are immutable")
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path, compression="zstd")
        item = {
            "path": str(path),
            "kind": kind,
            "session": str(day),
            "rows": len(records),
            "bytes": path.stat().st_size,
            "sha256": file_sha(path),
            "release_sha256": self.release_sha,
        }
        self.actual.partitions[kind, day] = path
        self.actual.partition_identity[path] = (item["bytes"], item["sha256"])
        self.generated[kind, day] = item

    def prepare_features(self, view):
        if ("features", view.session) in self.actual.partitions:
            return
        import pyarrow as pa

        from quantlab.stocks.prepare import _schemas

        if view.session not in self.window.rebalance_signals or view.session <= START:
            raise ValueError("Only actual released final signal features may be generated")
        inventory = {a: self.source._assets[a] for a in self.loaded.histories}
        records, audit = feature_records(
            view,
            self.schedule,
            self.loaded.source_identity,
            inventory,
            self.source.snapshot_id,
            top_n=500,
        )
        self._table("features", view.session, records, _schemas(pa)["features"])
        from quantlab.evaluation.holdout_pipeline import _write

        self.eligibility_artifacts[view.session] = _write(
            self.root / "eligibility" / (str(view.session) + ".json"),
            {"population": 7195, "candidates": audit, "selection_uses_outcomes": False},
        )
        if self.spec["projection"]["kind"] == "registered_wider":
            from quantlab.stocks.features import FEATURE_IDS as SIX
            from quantlab.stocks.wider_features import FEATURE_IDS

            fields = list(_schemas(pa)["features"])
            for feature in FEATURE_IDS:
                if feature not in SIX:
                    fields.extend(
                        (
                            pa.field("raw_" + feature, pa.float64()),
                            pa.field("rank_" + feature, pa.float64()),
                        )
                    )
                fields.extend(
                    (
                        pa.field("missing_" + feature, pa.bool_()),
                        pa.field("reason_" + feature, pa.string()),
                    )
                )
            self._table(
                "wider_features",
                view.session,
                wider_records(records, view, self.schedule, self.loaded.source_identity),
                pa.schema(fields),
            )

    def _labels(self, day, *, cutoff, purpose):
        if purpose not in {"mature_training", "outcome_only"}:
            raise ValueError("Explicit maturity or outcome-only label purpose required")
        if cutoff > self.schedule.session(self.window.terminal_close).decision_at:
            raise ValueError("Label request outside the released calendar boundary")
        if day not in self.signals and not (
            day < self.window.initial_signal
            and ("features", day) in self.actual.partitions
            and purpose == "mature_training"
        ):
            raise ValueError("Only original pre-start feature dates or released signals allowed")
        if ("labels", day) in self.generated:
            return
        _, exit_time = label_timing(self.schedule, day)
        if purpose == "mature_training" and exit_time.decision_at > cutoff:
            raise ValueError("No future label generation for expanding refits")
        if ("labels", day) in self.actual.partitions:
            metadata = self.actual._read(
                self.actual.partitions["labels", day],
                ["signal_session", "status", "planned_available_at"],
            )
            if any(
                r["signal_session"] != day or r["planned_available_at"] != exit_time.decision_at
                for r in metadata
            ):
                raise ValueError("Original label vintage timing differs from fixed calendar")
            # Mature original values and actual unresolved-source cases stay
            # immutable. Only an earlier endpoint's censoring is refreshed.
            if not any(r["status"] == "right_censored" for r in metadata):
                return
        import pyarrow as pa

        from quantlab.stocks.prepare import _schemas

        if day >= self.window.initial_signal:
            self._prior_exports(day)
        # Membership and raw feature bytes are inherited, never regenerated
        # or selected using the newly available outcome.
        self.actual._feature_keys(day)
        records = label_records(
            self.loaded.histories,
            self.schedule,
            self.loaded.source_identity,
            {a: self.source._assets[a] for a in self.loaded.histories},
            self.source.snapshot_id,
            day=day,
            endpoint=self.window.terminal_close,
        )
        self._table("labels", day, records, _schemas(pa)["labels"])

    def clocks(self, view):
        pending = []
        feature_days = {d for kind, d in self.actual.partitions if kind == "features"}
        for day in sorted(feature_days | set(self.signals)):
            if day >= view.session:
                continue
            _, exit_time = label_timing(self.schedule, day)
            if exit_time.decision_at <= view.cutoff:
                self._labels(day, cutoff=view.cutoff, purpose="mature_training")
            elif ("labels", day) not in self.actual.partitions:
                path = self.actual.partitions["features", day]
                digest = self.actual.partition_identity[path][1]
                pending.extend(
                    LabelClock(a, day, exit_time.decision_at, "not_yet_mature", digest)
                    for a in self.actual._feature_keys(day)
                )
        return (*self.actual.clocks(view), *pending)

    def fit(self, view, selected):
        for day in {d for d, _ in selected.keys if d >= self.window.initial_signal}:
            self._prior_exports(day)
        return self.actual.fit(view, selected)

    def decide(self, view, state):
        return self.actual.decide(view, state)

    def prepare_outcomes(self, *, cutoff):
        if cutoff != self.schedule.session(self.window.terminal_close).decision_at:
            raise ValueError("Outcome-only labels require the exact released terminal clock")
        for day in self.signals:
            self._prior_exports(day)
            self._labels(day, cutoff=cutoff, purpose="outcome_only")
