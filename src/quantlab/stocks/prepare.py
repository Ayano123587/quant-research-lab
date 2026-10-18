"""Stream a complete, authenticated stock population into monthly research tables.

Only explicit scoped software checks can omit source identities. They carry a
non-research scope marker and are refused by the normal consumption gate.
Labels are independent offline outcomes and never enter feature eligibility.
"""

import hashlib
import json
import operator
import platform
import re
import sys
import time
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from quantlab.data import archive as archive_module
from quantlab.data import calendar as calendar_module
from quantlab.data import ingest as ingest_module
from quantlab.data import returns as returns_module
from quantlab.data import snapshot as snapshot_module
from quantlab.data.archive import file_sha256
from quantlab.data.calendar import XNYSSchedule
from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks import features as feature_module
from quantlab.stocks import labels as label_module
from quantlab.stocks.features import FEATURE_IDS, candidate_from_history
from quantlab.stocks.labels import label_from_history

SOURCE_MANIFEST_SHA256 = "9d1a82b8f6f94d6b52fcb774b2e83a4b8af57a883f3a4a190c9f1a8288d71675"
SCHEMA = "real_stock_monthly_preparation_v1"


def _replace_progress_file(temporary, destination):
    """Allow short Windows reader locks without hiding permanent write errors."""
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError as error:
            if sys.platform != "win32" or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.25))


class _IndexedSessions(tuple):
    """Unchanged actual dates, with the same index contract as Python tuple."""

    def index(self, value, start=0, stop=sys.maxsize, /):
        start, stop, _ = slice(operator.index(start), operator.index(stop)).indices(len(self))
        if type(value) is not date:
            return super().index(value, start, stop)
        position = bisect_left(self, value, start, stop)
        if position < stop and self[position] == value:
            return position
        raise ValueError("tuple.index(x): x not in tuple")


class _ActualCalendarCache(XNYSSchedule):
    """Delegate actual timing; cache only answers returned by the real calendar."""

    def __init__(self, source):
        if not isinstance(source, XNYSSchedule):
            raise ValueError("Actual source schedule required")
        self._source = source
        self.sessions = _IndexedSessions(source.sessions)
        self._month_ends = {}

    @property
    def provenance(self):
        return self._source.provenance

    def session(self, day):
        return self._source.session(day)

    def next_session(self, day):
        return self._source.next_session(day)

    def month_end(self, year, month):
        key = year, month
        if key not in self._month_ends:
            self._month_ends[key] = self._source.month_end(year, month)
        return self._month_ends[key]


def _raw(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_new(path, value):
    with path.open("xb") as handle:
        handle.write(_raw(value))


def _source_code():
    return {
        name: file_sha256(Path(module.__file__))
        for name, module in (
            ("prepare", __import__(__name__, fromlist=["__file__"])),
            ("features", feature_module),
            ("labels", label_module),
            ("snapshot", snapshot_module),
            ("calendar", calendar_module),
            ("returns", returns_module),
            ("archive", archive_module),
            ("ingest", ingest_module),
        )
    }


def _schemas(pa):
    timestamp = pa.timestamp("us", tz="UTC")
    common = [
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("signal_session", pa.date32(), nullable=False),
        pa.field("signal_at", timestamp, nullable=False),
    ]
    provenance = [
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_member_sha256", pa.string(), nullable=False),
        pa.field("source_snapshot_id", pa.string(), nullable=False),
        pa.field("input_rows_sha256", pa.string(), nullable=False),
    ]
    feature = common + [
        pa.field("intended_execution_at", timestamp, nullable=False),
        pa.field("liquidity_rank", pa.int32(), nullable=False),
        pa.field("activity_proxy", pa.float64(), nullable=False),
        pa.field("max_input_available_at", timestamp, nullable=False),
        *provenance,
        *[
            pa.field(prefix + f, pa.float64(), nullable=False)
            for prefix in ("raw_", "rank_")
            for f in FEATURE_IDS
        ],
    ]
    candidate = common + [
        pa.field("intended_execution_at", timestamp, nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("reasons", pa.list_(pa.string()), nullable=False),
        pa.field("activity_proxy", pa.float64()),
        pa.field("max_input_available_at", timestamp),
        pa.field("observed_scheduled_marks", pa.int32(), nullable=False),
        pa.field("required_scheduled_marks", pa.int32(), nullable=False),
        pa.field("scheduled_window_start", pa.date32(), nullable=False),
        pa.field("scheduled_window_end", pa.date32(), nullable=False),
        *provenance,
        *[pa.field("raw_" + f, pa.float64()) for f in FEATURE_IDS],
    ]
    label = common + [
        pa.field("entry_session", pa.date32(), nullable=False),
        pa.field("entry_at", timestamp, nullable=False),
        pa.field("exit_session", pa.date32(), nullable=False),
        pa.field("exit_at", timestamp, nullable=False),
        pa.field("planned_available_at", timestamp, nullable=False),
        pa.field("label_available_at", timestamp),
        pa.field("entry_mark", pa.float64()),
        pa.field("exit_mark", pa.float64()),
        pa.field("return_value", pa.float64()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("source_status", pa.string(), nullable=False),
        pa.field("reasons", pa.list_(pa.string()), nullable=False),
        pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
        pa.field("entry_source_row_number", pa.int32()),
        pa.field("entry_source_row_sha256", pa.string()),
        pa.field("exit_source_row_number", pa.int32()),
        pa.field("exit_source_row_sha256", pa.string()),
        *provenance,
    ]
    return {
        "candidates": pa.schema(candidate),
        "features": pa.schema(feature),
        "labels": pa.schema(label),
    }


def _digest(rows, source_member_sha256):
    digest = hashlib.sha256(source_member_sha256.encode())
    for row in rows:
        digest.update(f"\n{row.session}:{row.row_number}:{row.row_sha256}".encode())
    return digest.hexdigest()


class _BufferedMonths:
    def __init__(self, root, signals, schemas, pa, pq):
        self.pa, self.pq, self.schemas = pa, pq, schemas
        self.writers, self.buffers, self.paths, self.counts = {}, {}, {}, Counter()
        for kind in ("candidates", "labels"):
            for signal in signals:
                key = kind, signal
                relative = f"{kind}/{signal.isoformat()}.parquet"
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                self.paths[key] = relative
                self.writers[key] = pq.ParquetWriter(path, schemas[kind], compression="zstd")
                self.buffers[key] = []

    def append(self, kind, signal, row):
        key = kind, signal
        self.buffers[key].append(row)
        self.counts[key] += 1
        if len(self.buffers[key]) >= 128:
            self.flush(key)

    def flush(self, key):
        rows = self.buffers[key]
        if rows:
            self.writers[key].write_table(
                self.pa.Table.from_pylist(rows, schema=self.schemas[key[0]])
            )
            rows.clear()

    def close(self):
        for key in tuple(self.writers):
            self.flush(key)
            self.writers.pop(key).close()


def _rank_rows(rows, top_n):
    """Compact equivalent of select_cross_section; no outcomes are accepted here."""
    selected = sorted(
        (r for r in rows if r["eligible"]), key=lambda r: (-r["activity_proxy"], r["asset_id"])
    )[:top_n]
    normalized = {r["asset_id"]: {} for r in selected}
    n = len(selected)
    for feature in FEATURE_IDS:
        ordered = sorted(selected, key=lambda r: (r["raw_" + feature], r["asset_id"]))
        left = 0
        while left < n:
            right = left + 1
            while right < n and ordered[right]["raw_" + feature] == ordered[left]["raw_" + feature]:
                right += 1
            rank = (left + 1 + right) / 2
            for row in ordered[left:right]:
                normalized[row["asset_id"]]["rank_" + feature] = (rank - 0.5) / n - 0.5
            left = right
    result = []
    for i, row in enumerate(selected):
        excluded = {
            "eligible",
            "reasons",
            "observed_scheduled_marks",
            "required_scheduled_marks",
            "scheduled_window_start",
            "scheduled_window_end",
        }
        result.append(
            {
                **{k: v for k, v in row.items() if k not in excluded},
                "liquidity_rank": i + 1,
                **normalized[row["asset_id"]],
            }
        )
    return result


def _signals(schedule, start, end):
    year, month = start.year, start.month
    result = []
    while (year, month) <= (end.year, end.month):
        signal = schedule.month_end(year, month).session
        if start <= signal <= end:
            result.append(signal)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    if not result:
        raise ValueError("At least one complete monthly decision required")
    if schedule.sessions.index(result[0]) < 252:
        raise ValueError("Calendar must supply 252 scheduled warm-up intervals")
    final = result[-1]
    year, month = (final.year + 1, 1) if final.month == 12 else (final.year, final.month + 1)
    schedule.next_session(schedule.month_end(year, month).session)
    return tuple(result)


def prepare_stock_panel(
    snapshot,
    schedule,
    output,
    *,
    start=date(2000, 1, 1),
    end=date(2017, 11, 10),
    top_n=500,
    scoped_test_assets=None,
):
    """Prepare every stock identity, unless explicit non-research test scope is set.

    No full preparation is run by importing this module. Each source history is
    authenticated/read once. Bisected feature windows and exact label marks are
    copied from that unchanged history; no missing market observations are made.
    """
    if (
        not isinstance(snapshot, MarketSnapshot)
        or snapshot.manifest_sha256 != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("Reviewed authenticated full MarketSnapshot required")
    if not isinstance(schedule, XNYSSchedule):
        raise ValueError("Actual XNYSSchedule required")
    schedule = _ActualCalendarCache(schedule)
    if type(start) is not date or type(end) is not date or start > end:
        raise ValueError("Ordered start/end dates required")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("Positive integer top_n required")
    parent_raw = (snapshot.root / "manifest.json").read_bytes()
    if _sha(parent_raw) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Source manifest changed")
    parent = json.loads(parent_raw)
    source_calendar = parent["contract"]["calendar"]
    for key in ("calendar", "library", "version", "pandas_version"):
        if schedule.provenance[key] != source_calendar[key]:
            raise ValueError("Study calendar runtime differs from authenticated normalization")
    endpoint = date.fromisoformat(parent["coverage"]["last_source_date"])
    if end > endpoint:
        raise ValueError("Decision endpoint exceeds global source coverage")
    signals = _signals(schedule, start, end)
    inventory = {
        r["asset_id"]: r for r in parent["source_inventory"] if r["asset_id"].startswith("stock:")
    }
    if len(inventory) != 7195:
        raise ValueError("Complete 7195 stock source population required")
    assets = tuple(sorted(inventory))
    scoped = scoped_test_assets is not None
    if scoped:
        requested = tuple(scoped_test_assets)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or set(requested) - inventory.keys()
        ):
            raise ValueError("Scoped software checks require unique real stock identities")
        assets = tuple(sorted(requested))
    output = Path(output).resolve()
    source_root = snapshot.root.resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("Prepared output must be outside immutable source snapshot")
    import pyarrow as pa
    import pyarrow.parquet as pq

    schemas = _schemas(pa)
    code = _source_code()
    config = {
        "schema": SCHEMA,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "top_n": top_n,
        "scope": "scoped_software_check_not_research_panel" if scoped else "all_archived_stocks",
        "scoped_test_assets": list(assets) if scoped else None,
        "archive_sha256": snapshot.archive_sha256,
        "snapshot_id": snapshot.snapshot_id,
        "source_manifest_sha256": snapshot.manifest_sha256,
        "calendar": schedule.provenance,
        "normalization_calendar": source_calendar,
        "calendar_cache": {
            "implementation": "transparent_actual_calendar_cache_v1",
            "unchanged_schedule_sha256": schedule.provenance["schedule_sha256"],
            "calendar_source_sha256": code["calendar"],
        },
        "source_code_sha256": code,
        "formula_version": feature_module.FORMULA_VERSION,
        "feature_units": feature_module.FEATURE_UNITS,
        "activity_convention": feature_module.ACTIVITY_CONVENTION,
        "required_features": FEATURE_IDS,
        "optional_imputation": "none_all_six_required",
        "target_id": label_module.TARGET_ID,
        "label_units": "decimal_simple_opening_price_return",
        "price_basis": parent["contract"]["price_basis"],
        "global_archive_endpoint": endpoint.isoformat(),
        "label_availability": "exit_session_close_plus_one_minute_assumed",
        "source_read_mode": (
            "verified_read_session_full_metadata_entry_exit_whole_partition_sha256_before_arrow"
        ),
        "selection": "past_only_top_activity_descending_then_source_asset_id",
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": version("pyarrow"),
            "exchange_calendars": version("exchange-calendars"),
        },
        "limitations": [
            "archive_conditioned_population_not_historical_index_membership",
            "adjustment_currency_and_price_volume_units_unverified",
            "opening_marks_research_proxies_not_auction_fills",
            "no_delisting_returns_or_terminal_recovery_assumed",
            "2015_and_2017_marks_previously_inspected_for_availability_checks",
        ],
    }
    config = json.loads(_raw(config))  # Detach caller-owned calendar provenance.
    config_hash = _sha(_raw(config))
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    counts, source_stats, tables, support = Counter(), {}, [], []
    writer = None
    stage, current_asset = "initialization", None

    def status(state):
        temporary = output / "status.json.pending"
        temporary.write_bytes(
            _raw(
                {
                    "schema": SCHEMA,
                    "status": state,
                    "complete": state == "complete",
                    "config_sha256": config_hash,
                    "counts": dict(counts),
                    "stage": stage,
                    "started_at": started,
                }
            )
        )
        _replace_progress_file(temporary, output / "status.json")

    try:
        _write_new(output / "config.json", config)
        status("staging")
        writer = _BufferedMonths(output, signals, schemas, pa, pq)
        stage = "source_read_session_entry"
        with (
            snapshot.verified_read_session() as verified,
            (output / "source-inputs.jsonl").open("xb") as journal,
        ):
            stage = "asset_features_and_independent_labels"
            for asset in assets:
                current_asset = asset
                entry = inventory[asset]
                relative = entry.get("output", {}).get("path")
                if relative:
                    path = snapshot.root / relative
                    stamp = path.stat()
                    source_stats[relative] = (stamp.st_size, stamp.st_mtime_ns)
                bars = verified.bars("stock", asset.split(":", 1)[1])
                dates = tuple(b.session for b in bars)
                indexed = {b.session: b for b in bars}
                source_hash = entry["source_sha256"]
                provenance = {
                    "source_member": entry["source_member"],
                    "source_member_sha256": source_hash,
                    "source_snapshot_id": snapshot.snapshot_id,
                }
                journal.write(
                    _raw(
                        {
                            "asset_id": asset,
                            **provenance,
                            "rows": len(bars),
                            "input_rows_sha256": _digest(bars, source_hash),
                            "source_partition_sha256": entry.get("output", {}).get("sha256"),
                            "empty_source_preserved": not bars,
                        }
                    ).replace(b"\n", b" ")
                    + b"\n"
                )
                for signal in signals:
                    index = schedule.sessions.index(signal)
                    first = schedule.sessions[index - 252]
                    window = bars[bisect_left(dates, first) : bisect_right(dates, signal)]
                    candidate = candidate_from_history(
                        asset,
                        window,
                        schedule,
                        as_of=signal,
                        archive_sha256=snapshot.archive_sha256,
                    )
                    raw_features = dict(candidate.features)
                    row = {
                        "asset_id": asset,
                        "signal_session": signal,
                        "signal_at": candidate.signal_at,
                        "intended_execution_at": candidate.intended_execution_at,
                        "eligible": candidate.eligible,
                        "reasons": list(candidate.reasons),
                        "activity_proxy": candidate.activity_proxy,
                        "max_input_available_at": candidate.max_input_available_at,
                        "observed_scheduled_marks": candidate.observed_scheduled_marks,
                        "required_scheduled_marks": 253,
                        "scheduled_window_start": first,
                        "scheduled_window_end": signal,
                        **provenance,
                        "input_rows_sha256": _digest(candidate.source_rows, source_hash),
                        **{"raw_" + f: raw_features.get(f) for f in FEATURE_IDS},
                    }
                    writer.append("candidates", signal, row)
                    next_year, next_month = (
                        (signal.year + 1, 1)
                        if signal.month == 12
                        else (signal.year, signal.month + 1)
                    )
                    entry_day = schedule.next_session(signal).session
                    exit_day = schedule.next_session(
                        schedule.month_end(next_year, next_month).session
                    ).session
                    marks = tuple(indexed[d] for d in (entry_day, exit_day) if d in indexed)
                    label = label_from_history(
                        asset,
                        marks,
                        schedule,
                        signal_session=signal,
                        archive_endpoint=endpoint,
                        archive_sha256=snapshot.archive_sha256,
                    )
                    label_row = {
                        k: getattr(label, k)
                        for k in (
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
                    label_row.update(
                        {
                            "reasons": list(label.reasons),
                            "quality_flags": list(label.quality_flags),
                            **provenance,
                            "input_rows_sha256": _digest(marks, source_hash),
                        }
                    )
                    for side in ("entry", "exit"):
                        mark = getattr(label, side + "_row")
                        label_row[side + "_source_row_number"] = mark.row_number if mark else None
                        label_row[side + "_source_row_sha256"] = mark.row_sha256 if mark else None
                    writer.append("labels", signal, label_row)
                counts["source_histories_read"] += 1
                counts["source_rows_read"] += len(bars)
                counts["empty_source_histories"] += not bars
                if counts["source_histories_read"] % 64 == 0:
                    status("staging")
            stage, current_asset = "source_read_session_postflight", None
        # The context's full source postflight must succeed before any finalization.
        writer.close()
        stage, current_asset = "monthly_cross_sections", None
        (output / "features").mkdir()
        for signal in signals:
            candidate_path = output / f"candidates/{signal.isoformat()}.parquet"
            rows = pq.ParquetFile(candidate_path).read().to_pylist()
            if len(rows) != len(assets) or len({r["asset_id"] for r in rows}) != len(assets):
                raise ValueError("Monthly complete candidate population mismatch")
            selected = _rank_rows(rows, top_n)
            path = output / f"features/{signal.isoformat()}.parquet"
            pq.write_table(
                pa.Table.from_pylist(selected, schema=schemas["features"]), path, compression="zstd"
            )
            labels = (
                pq.ParquetFile(output / f"labels/{signal.isoformat()}.parquet").read().to_pylist()
            )
            if len(labels) != len(assets) or {r["asset_id"] for r in labels} != set(assets):
                raise ValueError("Monthly independent label population mismatch")
            reasons = Counter(reason for r in rows for reason in r["reasons"])
            support.append(
                {
                    "signal_session": signal.isoformat(),
                    "candidates": len(rows),
                    "eligible": sum(r["eligible"] for r in rows),
                    "selected": len(selected),
                    "feature_exclusion_reasons": dict(reasons),
                    "label_status_counts": dict(Counter(r["status"] for r in labels)),
                }
            )
            for kind in schemas:
                relative = f"{kind}/{signal.isoformat()}.parquet"
                actual = output / relative
                metadata = pq.read_metadata(actual)
                expected_rows = len(selected) if kind == "features" else len(assets)
                if (
                    metadata.num_rows != expected_rows
                    or metadata.schema.to_arrow_schema() != schemas[kind]
                ):
                    raise ValueError("Prepared table schema/count mismatch")
                tables.append(
                    {
                        "path": relative,
                        "kind": kind,
                        "rows": expected_rows,
                        "sha256": file_sha256(actual),
                        "bytes": actual.stat().st_size,
                    }
                )
                counts[kind + "_rows"] += expected_rows
        if not scoped and (
            counts["source_histories_read"] != 7195
            or counts["source_rows_read"] != 14887665
            or counts["empty_source_histories"] != 32
        ):
            raise ValueError("Full source population/counts mismatch")
        for relative, expected in source_stats.items():
            stamp = (snapshot.root / relative).stat()
            if (stamp.st_size, stamp.st_mtime_ns) != expected:
                raise ValueError("Source partition changed during preparation")
        if (
            (snapshot.root / "COMPLETE").read_bytes() != (SOURCE_MANIFEST_SHA256 + "\n").encode()
            or file_sha256(snapshot.root / "manifest.json") != SOURCE_MANIFEST_SHA256
            or (snapshot.root / "FAILURE.json").exists()
            or _source_code() != code
            or schedule.provenance != config["calendar"]
        ):
            raise ValueError("Source completion/code changed during preparation")
        _write_new(output / "support.json", support)
        outputs = tables + [
            {
                "path": name,
                "sha256": file_sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("config.json", "source-inputs.jsonl", "support.json")
        ]
        stage = "complete"
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "complete": True,
            "research_scope_complete": not scoped,
            "scope": config["scope"],
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_snapshot_id": snapshot.snapshot_id,
            "config_sha256": config_hash,
            "config": config,
            "counts": dict(counts),
            "decision_count": len(signals),
            "outputs": outputs,
            "errors": [],
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        _write_new(output / "manifest.json", manifest)
        status("complete")
        digest = file_sha256(output / "manifest.json")
        with (output / "COMPLETE").open("xb") as handle:
            handle.write((digest + "\n").encode())
        return {**manifest, "manifest_sha256": digest}
    except BaseException as error:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass  # Original failure remains authoritative; partial tables stay unconsumable.
        _write_new(
            output / "FAILURE.json",
            {
                "schema": SCHEMA,
                "stage": stage,
                "asset_id": current_asset,
                "counts": dict(counts),
                "error_type": type(error).__name__,
                "message": str(error),
                "config_sha256": config_hash,
                "complete": False,
            },
        )
        status("failed")
        raise


def verify_prepared_panel(output, expected_manifest_sha256, *, allow_scoped_test=False):
    """Authenticate a completed panel; ordinary consumers refuse scoped checks."""
    output = Path(output)
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError("External prepared manifest SHA256 required")
    if (output / "FAILURE.json").exists() or not (output / "COMPLETE").is_file():
        raise ValueError("Completed error-free prepared panel required")
    if (
        file_sha256(output / "manifest.json") != expected_manifest_sha256
        or (output / "COMPLETE").read_bytes() != (expected_manifest_sha256 + "\n").encode()
    ):
        raise ValueError("External prepared manifest hash mismatch")
    manifest = json.loads((output / "manifest.json").read_bytes())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("status") != "complete"
        or manifest.get("errors") != []
        or manifest.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("Completed reviewed-source prepared panel required")
    if not manifest.get("research_scope_complete") and not allow_scoped_test:
        raise ValueError("Scoped software check is not a full research panel")
    if _sha(_raw(manifest["config"])) != manifest["config_sha256"] or (
        output / "config.json"
    ).read_bytes() != _raw(manifest["config"]):
        raise ValueError("Prepared configuration identity mismatch")
    import pyarrow as pa
    import pyarrow.parquet as pq

    schemas = _schemas(pa)
    names, table_counts, table_months = set(), Counter(), {kind: set() for kind in schemas}
    for item in manifest["outputs"]:
        relative = item["path"]
        path = output / relative
        if relative in names or path.resolve().is_relative_to(output.resolve()) is False:
            raise ValueError("Unsafe or duplicate prepared output path")
        names.add(relative)
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError("Prepared output size/hash mismatch")
        if "kind" in item:
            kind = item["kind"]
            if kind not in schemas or not relative.startswith(kind + "/"):
                raise ValueError("Unknown prepared table contract")
            metadata = pq.read_metadata(path)
            if (
                metadata.num_rows != item["rows"]
                or metadata.schema.to_arrow_schema() != schemas[kind]
            ):
                raise ValueError("Prepared table schema/count mismatch")
            table_counts[kind + "_rows"] += item["rows"]
            table_months[kind].add(Path(relative).stem)
    if (
        any(len(months) != manifest["decision_count"] for months in table_months.values())
        or table_months["candidates"] != table_months["labels"]
        or table_months["candidates"] != table_months["features"]
        or any(manifest["counts"].get(k) != v for k, v in table_counts.items())
    ):
        raise ValueError("Prepared table support/count mismatch")
    inputs = [
        json.loads(line) for line in (output / "source-inputs.jsonl").read_bytes().splitlines()
    ]
    source_counts = {
        "source_histories_read": len(inputs),
        "source_rows_read": sum(r["rows"] for r in inputs),
        "empty_source_histories": sum(r["empty_source_preserved"] for r in inputs),
    }
    if (
        len({r["asset_id"] for r in inputs}) != len(inputs)
        or any(manifest["counts"].get(k) != v for k, v in source_counts.items())
        or table_counts["candidates_rows"] != len(inputs) * manifest["decision_count"]
        or table_counts["labels_rows"] != len(inputs) * manifest["decision_count"]
    ):
        raise ValueError("Prepared source population mismatch")
    full = manifest["research_scope_complete"]
    if full and (
        source_counts
        != {
            "source_histories_read": 7195,
            "source_rows_read": 14887665,
            "empty_source_histories": 32,
        }
        or manifest["scope"] != "all_archived_stocks"
        or manifest["config"]["scoped_test_assets"] is not None
    ):
        raise ValueError("Full research scope requires every authenticated stock identity")
    if not full and (
        manifest["scope"] != "scoped_software_check_not_research_panel"
        or set(manifest["config"]["scoped_test_assets"]) != {r["asset_id"] for r in inputs}
    ):
        raise ValueError("Scoped software-check identity mismatch")
    expected = names | {"manifest.json", "COMPLETE", "status.json"}
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != expected:
        raise ValueError("Unexpected or absent prepared outputs")
    status = json.loads((output / "status.json").read_bytes())
    if (
        status.get("complete") is not True
        or status.get("status") != "complete"
        or status.get("config_sha256") != manifest["config_sha256"]
        or status.get("counts") != manifest["counts"]
    ):
        raise ValueError("Prepared completion status mismatch")
    return manifest
