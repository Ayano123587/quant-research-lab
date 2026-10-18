"""Immutable, lossless normalization of the supplied real daily-price archive.

The snapshot retains financially invalid and off-calendar observations with flags.
It never fills a price, repairs an OHLC value, or infers a delisting return.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from collections import Counter
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from . import archive as archive_module
from . import calendar as calendar_module
from .archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive, file_sha256
from .calendar import XNYSSchedule

SCHEMA_VERSION = "real_market_snapshot_v1"
PRICE_BASIS = "provider_claimed_split_dividend_adjusted_not_independently_verified"
EXPECTED_FILES = 8539
EXPECTED_ROWS = 17453243
PROVIDER_CLAIM = {
    "publisher": "borismarjanovic",
    "url": "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
    "claim": "Publisher describes prices as dividend/split adjusted and data as CC0.",
    "independent_verification": False,
    "archive_facts": (
        "No adjustment factors, action events, dividends, or delisting returns supplied."
    ),
}


class IngestionError(ValueError):
    """A snapshot failed; its partial data and manifest must remain for inspection."""


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _checked_json(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise IngestionError("An externally recorded audit SHA256 is required")
    raw = Path(path).read_bytes()
    if _sha(raw) != expected_sha256:
        raise IngestionError("SOURCE_AUDIT_HASH_MISMATCH: " + str(path))
    return raw, json.loads(raw)


def _source_identity():
    return {
        "quantlab/data/ingest.py": file_sha256(Path(__file__)),
        "quantlab/data/archive.py": file_sha256(Path(archive_module.__file__)),
        "quantlab/data/calendar.py": file_sha256(Path(calendar_module.__file__)),
    }


def _arrow_schema(pa):
    return pa.schema(
        [
            pa.field("asset_id", pa.string(), nullable=False),
            pa.field("asset_type", pa.string(), nullable=False),
            pa.field("source_symbol", pa.string(), nullable=False),
            pa.field("session", pa.date32(), nullable=False),
            *[
                pa.field(name, pa.float64(), nullable=False)
                for name in ("open", "high", "low", "close", "volume", "open_interest")
            ],
            pa.field("source_member", pa.string(), nullable=False),
            pa.field("source_row_number", pa.int32(), nullable=False),
            pa.field("source_member_sha256", pa.string(), nullable=False),
            pa.field("source_row_sha256", pa.string(), nullable=False),
            pa.field("source_archive_sha256", pa.string(), nullable=False),
            pa.field("quality_flags", pa.list_(pa.field("element", pa.string())), nullable=False),
            pa.field("is_xnys_session", pa.bool_(), nullable=False),
            pa.field("xnys_open_at", pa.timestamp("us", tz="UTC")),
            pa.field("xnys_close_at", pa.timestamp("us", tz="UTC")),
            pa.field("assumed_available_at", pa.timestamp("us", tz="UTC")),
            pa.field("price_basis", pa.string(), nullable=False),
        ]
    )


def _asset_metadata(records):
    assets, members = set(), set()
    for entry in records:
        member = entry.get("path")
        match = re.fullmatch(r"Data/(Stocks|ETFs)/([a-z0-9][a-z0-9._^-]*)\.us\.txt", member or "")
        if not match:
            raise IngestionError("UNSAFE_OR_UNRECOGNIZED_AUDITED_MEMBER: " + str(member))
        directory, symbol = match.groups()
        asset_type = "stock" if directory == "Stocks" else "etf"
        asset_id = asset_type + ":" + symbol + ".us"
        if member in members or asset_id in assets:
            raise IngestionError("DUPLICATE_AUDITED_ASSET: " + member)
        if entry.get("asset_type") != directory or entry.get("symbol") != symbol:
            raise IngestionError("AUDITED_SYMBOL_OR_ASSET_TYPE_MISMATCH: " + member)
        if type(entry.get("rows")) is not int or entry["rows"] < 0:
            raise IngestionError("INVALID_AUDITED_ROW_COUNT: " + member)
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise IngestionError("AUDITED_MEMBER_HASH_REQUIRED: " + member)
        assets.add(asset_id)
        members.add(member)
        yield entry, asset_type, symbol, asset_id


def ingest_archive(
    archive_path: str | Path,
    output: str | Path,
    *,
    archive_sha256: str,
    audit_summary_path: str | Path,
    audit_summary_sha256: str,
    file_summaries_path: str | Path,
    file_summaries_sha256: str,
    schedule: XNYSSchedule,
) -> dict:
    """Write all 8,539 real histories to a fresh, partitioned Parquet snapshot.

    ``schedule`` must be a reviewed XNYSSchedule spanning 1962 through 2018.
    The caller freezes both existing full-file audit hashes before dispatch.
    Empty files remain inventory records, not invented empty price observations.
    A failed member remains an explicit error; no incomplete snapshot is complete.
    No resume/overwrite path is provided. Partial snapshots remain untouched.
    """
    if archive_sha256 != SUPPLIED_ARCHIVE_SHA256:
        raise IngestionError("This normalizer supports the exact audited supplied archive only")
    summary_raw, summary = _checked_json(audit_summary_path, audit_summary_sha256)
    files_raw, records = _checked_json(file_summaries_path, file_summaries_sha256)
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise IngestionError("FULL_SOURCE_AUDIT_REQUIRED")
    if (
        summary.get("source_sha256") != archive_sha256
        or summary.get("primary_files") != EXPECTED_FILES
        or summary.get("primary_rows") != EXPECTED_ROWS
        or summary.get("all_mirrors_byte_identical") is not True
        or summary.get("unsafe_members") != []
        or summary.get("encrypted_members") != []
        or summary.get("repeated_member_names") != []
        or len(records) != EXPECTED_FILES
        or sum(row.get("rows", 0) for row in records) != EXPECTED_ROWS
    ):
        raise IngestionError("AUDIT_DOES_NOT_COVER_COMPLETE_SUPPLIED_ARCHIVE")
    assets = tuple(_asset_metadata(records))
    # Detach the frozen identity from the caller's mutable provenance dictionary.
    provenance = json.loads(json.dumps(schedule.provenance, allow_nan=False))
    if (
        provenance.get("calendar") != "XNYS"
        or date.fromisoformat(provenance["start"]) > date(1962, 1, 1)
        or date.fromisoformat(provenance["end"]) < date(2018, 12, 31)
        or not schedule.sessions
    ):
        raise IngestionError("REVIEWED_XNYS_1962_THROUGH_2018_SCHEDULE_REQUIRED")
    # Optional dependencies are imported only after authenticating the real input audits.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _arrow_schema(pa)
    schema_raw = schema.serialize().to_pybytes()
    sources = _source_identity()
    contract = {
        "schema": SCHEMA_VERSION,
        "archive_sha256": archive_sha256,
        "parents": {
            "full_source_audit_sha256": audit_summary_sha256,
            "full_file_audit_sha256": file_summaries_sha256,
        },
        "source_code_sha256": sources,
        "arrow_schema_sha256": _sha(schema_raw),
        "calendar": provenance,
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": version("pyarrow"),
            "exchange_calendars": version("exchange-calendars"),
        },
        "provider_claim": PROVIDER_CLAIM,
        "price_basis": PRICE_BASIS,
        "source_clock": {
            "observed": "Source Date is a daily label, not an observed publication timestamp.",
            "derived": (
                "XNYS UTC open/close times are a reference-calendar mapping, "
                "not verified source exchange metadata."
            ),
            "availability": (
                "Daily bar assumed available at XNYS close plus one minute; "
                "not measured in supplied archive."
            ),
            "off_calendar": (
                "Keep row; derived market timestamps are null. No shift, fill, or deletion."
            ),
        },
        "row_policy": (
            "Preserve every parseable actual row and original numeric values; "
            "flag quality findings without repair."
        ),
        "universe_policy": (
            "All audited source files; no performance, liquidity, survivorship, "
            "or future-coverage screen."
        ),
        "expected_primary_files": EXPECTED_FILES,
        "expected_rows": EXPECTED_ROWS,
    }
    contract_raw = _json_bytes(contract)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    outputs, errors, inventory = [], [], []
    counts = Counter()
    flags = Counter()
    first_date, last_date = None, None
    started = datetime.now(UTC).isoformat()
    status_path = output / "status.json"

    def status(state):
        value = {
            "schema": SCHEMA_VERSION,
            "status": state,
            "started_at": started,
            "contract_sha256": _sha(contract_raw),
            "counts": dict(counts),
            "errors": errors,
            "complete": state == "complete",
        }
        temporary = output / "status.json.pending"
        temporary.write_bytes(_json_bytes(value))
        os.replace(temporary, status_path)

    def write_new(relative, raw):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
        outputs.append({"path": relative, "sha256": _sha(raw), "bytes": len(raw)})

    def final_manifest(state):
        return {
            "schema": SCHEMA_VERSION,
            "status": state,
            "complete": state == "complete",
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "snapshot_id": _sha(contract_raw),
            "contract": contract,
            "counts": dict(counts),
            "quality_flag_counts": dict(flags),
            "coverage": {
                "first_source_date": str(first_date) if first_date else None,
                "last_source_date": str(last_date) if last_date else None,
            },
            "exclusions": [],
            "errors": errors,
            "outputs": outputs,
            "source_inventory": inventory,
            "incomplete_rows": EXPECTED_ROWS - counts["rows_written"],
            "row_loss_policy": (
                "Any failed member is explicit; "
                "incomplete snapshots cannot be consumed as complete."
            ),
        }

    status("staging")
    try:
        write_new("contract.json", contract_raw)
        write_new("schema.arrow", schema_raw)
        write_new("parents/source-audit-summary.json", summary_raw)
        write_new("parents/source-file-audit.json", files_raw)
        archive_path = Path(archive_path)
        before = archive_path.stat()
        source_stat = (before.st_size, before.st_mtime_ns)
        with MarketArchive(archive_path, archive_sha256) as source:
            for expected, asset_type, symbol, asset_id in assets:
                entry = {
                    "asset_id": asset_id,
                    "source_member": expected["path"][5:],
                    "audited_member": expected["path"],
                    "source_sha256": expected["sha256"],
                    "source_bytes": expected["bytes"],
                    "expected_rows": expected["rows"],
                }
                try:
                    bars = source.bars(asset_type, symbol)
                    source_sha = (
                        bars[0].source_sha256
                        if bars
                        else _sha(source.source_bytes(asset_type, symbol)[1])
                    )
                    if source_sha != expected["sha256"] or len(bars) != expected["rows"]:
                        raise IngestionError("MEMBER_HASH_OR_ROW_COUNT_MISMATCH")
                    if bars and (
                        str(bars[0].session) != expected["first_date"]
                        or str(bars[-1].session) != expected["last_date"]
                    ):
                        raise IngestionError("AUDITED_DATE_COVERAGE_MISMATCH")
                    rows = []
                    asset_flags = Counter()
                    for bar in bars:
                        if bar.asset_id != asset_id or bar.source_sha256 != source_sha:
                            raise IngestionError("CONFLICTING_ROW_IDENTITY")
                        quality = set(bar.quality_flags)
                        if bar.volume == 0:
                            quality.add("ZERO_VOLUME")
                        if not bar.volume.is_integer():
                            quality.add("FRACTIONAL_VOLUME")
                        if bar.open_interest != 0:
                            quality.add("NONZERO_OPEN_INTEREST")
                        try:
                            timing = schedule.session(bar.session)
                        except ValueError:
                            timing = None
                            quality.add("OFF_XNYS_SESSION")
                        asset_flags.update(quality)
                        rows.append(
                            {
                                "asset_id": asset_id,
                                "asset_type": asset_type,
                                "source_symbol": symbol + ".us",
                                "session": bar.session,
                                "open": bar.open,
                                "high": bar.high,
                                "low": bar.low,
                                "close": bar.close,
                                "volume": bar.volume,
                                "open_interest": bar.open_interest,
                                "source_member": bar.source_member,
                                "source_row_number": bar.row_number,
                                "source_member_sha256": bar.source_sha256,
                                "source_row_sha256": bar.row_sha256,
                                "source_archive_sha256": archive_sha256,
                                "quality_flags": sorted(quality),
                                "is_xnys_session": timing is not None,
                                "xnys_open_at": timing.open_at if timing else None,
                                "xnys_close_at": timing.close_at if timing else None,
                                "assumed_available_at": timing.decision_at if timing else None,
                                "price_basis": PRICE_BASIS,
                            }
                        )
                    entry.update(rows=len(rows), quality_flag_counts=dict(asset_flags))
                    if rows:
                        table = pa.Table.from_pylist(rows, schema=schema)
                        relative = (
                            f"bars/asset_type={asset_type}/source_symbol={symbol}.us/"
                            "part-00000.parquet"
                        )
                        path = output / relative
                        path.parent.mkdir(parents=True, exist_ok=False)
                        temporary = path.with_suffix(".parquet.pending")
                        pq.write_table(table, temporary, compression="zstd", version="2.6")
                        metadata = pq.read_metadata(temporary)
                        if (
                            metadata.num_rows != len(rows)
                            or metadata.schema.to_arrow_schema() != schema
                        ):
                            raise IngestionError("PARQUET_ROW_COUNT_OR_SCHEMA_MISMATCH")
                        # Only a newly created snapshot owns this directory and destination.
                        if path.exists():
                            raise IngestionError("PARQUET_DESTINATION_ALREADY_EXISTS")
                        os.rename(temporary, path)
                        output_item = {
                            "path": relative,
                            "sha256": file_sha256(path),
                            "bytes": path.stat().st_size,
                            "rows": len(rows),
                            "asset_id": asset_id,
                        }
                        outputs.append(output_item)
                        entry.update(status="written", output=output_item)
                        first_date = min(first_date or bars[0].session, bars[0].session)
                        last_date = max(last_date or bars[-1].session, bars[-1].session)
                        counts["nonempty_files"] += 1
                        counts["rows_written"] += len(rows)
                    else:
                        entry["status"] = "empty_source_preserved_in_inventory"
                        counts["empty_files"] += 1
                    counts["successful_files"] += 1
                    counts[asset_type + "_files"] += 1
                    flags.update(asset_flags)
                except Exception as error:
                    entry.update(status="failed", error_type=type(error).__name__, error=str(error))
                    errors.append(
                        {
                            "asset_id": asset_id,
                            "source_member": entry["source_member"],
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    counts["failed_files"] += 1
                inventory.append(entry)
                counts["processed_files"] += 1
                with (output / "inventory.jsonl").open("ab") as handle:
                    handle.write(
                        (json.dumps(entry, sort_keys=True, allow_nan=False) + "\n").encode()
                    )
                if counts["processed_files"] % 100 == 0:
                    status("staging")
            after = archive_path.stat()
            if (after.st_size, after.st_mtime_ns) != source_stat:
                raise IngestionError("ARCHIVE_CHANGED_DURING_NORMALIZATION")
        if _source_identity() != sources:
            raise IngestionError("SOURCE_CODE_CHANGED_DURING_NORMALIZATION")
        if schedule.provenance != provenance:
            raise IngestionError("CALENDAR_PROVENANCE_CHANGED_DURING_NORMALIZATION")
        if (
            errors
            or counts["processed_files"] != EXPECTED_FILES
            or counts["rows_written"] != EXPECTED_ROWS
        ):
            raise IngestionError("INCOMPLETE_FULL_ARCHIVE_SNAPSHOT")
        journal = output / "inventory.jsonl"
        outputs.append(
            {
                "path": "inventory.jsonl",
                "sha256": file_sha256(journal),
                "bytes": journal.stat().st_size,
            }
        )
        manifest = final_manifest("complete")
        raw = _json_bytes(manifest)
        with (output / "manifest.json").open("xb") as handle:
            handle.write(raw)
        with (output / "COMPLETE").open("xb") as handle:
            handle.write((_sha(raw) + "\n").encode())
        status("complete")
        return manifest
    except BaseException as error:
        errors.append(
            {"stage": "snapshot", "error_type": type(error).__name__, "message": str(error)}
        )
        with (output / "FAILURE.json").open("xb") as handle:
            handle.write(
                _json_bytes(
                    {
                        "complete": False,
                        "errors": errors,
                        "instruction": "Reject this snapshot even if a finalization marker exists.",
                    }
                )
            )
        if not (output / "manifest.json").exists():
            with (output / "manifest.json").open("xb") as handle:
                handle.write(_json_bytes(final_manifest("failed")))
        status("failed")
        raise
