"""Authenticate completed full-archive snapshots before parsing their partitions.

The caller supplies the manifest hash from an external review receipt. A
self-authenticating COMPLETE marker alone never authorizes this loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections import Counter
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from threading import Lock, get_ident

from .archive import SUPPLIED_ARCHIVE_SHA256, Bar


class SnapshotError(ValueError):
    """The snapshot is incomplete, unauthenticated, or inconsistent."""


class _VerifiedReadSession:
    """One owner thread, one entry, and an unconditional authenticated exit."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.used = False
        self.entered = False
        self.owner = None
        self.identities = None

    def __enter__(self):
        if self.used:
            raise SnapshotError("Verified read session cannot be reentered")
        self.used = True
        snapshot = self.snapshot
        if not snapshot._session_lock.acquire(blocking=False):
            raise SnapshotError("Nested or concurrent verified read session forbidden")
        try:
            snapshot._full_session_check()
            self.identities = snapshot._metadata_identities()
            self.owner = get_ident()
            snapshot._active_session = self
            self.entered = True
            return snapshot
        except BaseException:
            snapshot._session_lock.release()
            raise

    def __exit__(self, error_type, error, traceback):
        if not self.entered or self.owner != get_ident():
            raise SnapshotError("Verified read session must exit on its owner thread")
        snapshot = self.snapshot
        try:
            # Always rehash externally bound metadata and the source closure,
            # including when a read or the caller's body raised an exception.
            snapshot._full_session_check()
            if snapshot._metadata_identities() != self.identities:
                raise SnapshotError("Verified session metadata identity changed")
        finally:
            snapshot._active_session = None
            self.entered = False
            snapshot._session_lock.release()
        return False


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class MarketSnapshot:
    """Lazy authenticated Bar histories plus separate derived calendar columns.

    ``bars(asset_type, symbol)`` mirrors MarketArchive. ``rows(asset_id)`` retains
    all normalized fields. ``verify_all()`` streams every real partition without
    retaining the full panel in memory. Source financial flags are reconstructed
    separately from informational normalization flags on Bar objects.
    """

    def __init__(self, root: str | Path, expected_manifest_sha256: str):
        self.root = Path(root).absolute()
        self._safe(self.root)
        if not self.root.is_dir():
            raise SnapshotError("Snapshot directory required")
        if (self.root / "FAILURE.json").exists():
            raise SnapshotError("FAILURE snapshot forbidden")
        if not (self.root / "COMPLETE").is_file():
            raise SnapshotError("COMPLETE marker required; staging snapshot refused")
        if not _hash(expected_manifest_sha256):
            raise SnapshotError("External manifest SHA256 required")
        self._completion_guard(expected_manifest_sha256)
        raw = self._read("manifest.json")
        if _sha(raw) != expected_manifest_sha256:
            raise SnapshotError("EXTERNAL_MANIFEST_HASH_MISMATCH")
        manifest = json.loads(raw)
        if (
            manifest.get("schema") != "real_market_snapshot_v1"
            or manifest.get("status") != "complete"
            or manifest.get("complete") is not True
            or manifest.get("errors") != []
            or manifest.get("exclusions") != []
            or manifest.get("incomplete_rows") != 0
        ):
            raise SnapshotError("Completed error-free full snapshot required")
        counts = manifest.get("counts", {})
        expected = {
            "processed_files": 8539,
            "successful_files": 8539,
            "rows_written": 17453243,
            "empty_files": 32,
            "nonempty_files": 8507,
            "stock_files": 7195,
            "etf_files": 1344,
        }
        if any(type(counts.get(k)) is not int or counts[k] != v for k, v in expected.items()):
            raise SnapshotError("FULL_ARCHIVE_COUNTS_REQUIRED")
        if counts.get("failed_files", 0) != 0:
            raise SnapshotError("Failed source members cannot be consumed")
        contract = manifest.get("contract", {})
        contract_raw = self._read("contract.json")
        if (
            contract_raw != _json_bytes(contract)
            or not _hash(manifest.get("snapshot_id"))
            or _sha(contract_raw) != manifest["snapshot_id"]
            or contract.get("schema") != "real_market_snapshot_v1"
            or contract.get("archive_sha256") != SUPPLIED_ARCHIVE_SHA256
            or contract.get("expected_primary_files") != 8539
            or contract.get("expected_rows") != 17453243
        ):
            raise SnapshotError("CONTRACT_IDENTITY_MISMATCH")
        self._outputs = {}
        for item in manifest.get("outputs", []):
            name = item.get("path")
            path = self._path(name)
            if name in self._outputs or not _hash(item.get("sha256")):
                raise SnapshotError("Duplicate output or invalid output hash")
            if type(item.get("bytes")) is not int or item["bytes"] < 0:
                raise SnapshotError("Invalid output size")
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise SnapshotError("Output absent or size differs: " + name)
            self._outputs[name] = dict(item)
        metadata = {
            "contract.json",
            "schema.arrow",
            "parents/source-audit-summary.json",
            "parents/source-file-audit.json",
            "inventory.jsonl",
        }
        if not metadata <= self._outputs.keys():
            raise SnapshotError("All bound metadata outputs required")
        metadata_raw = {name: self._authenticated(name) for name in metadata}
        if _sha(metadata_raw["schema.arrow"]) != contract.get("arrow_schema_sha256"):
            raise SnapshotError("Arrow schema hash differs from contract")
        parents = contract.get("parents", {})
        if self._outputs["parents/source-audit-summary.json"]["sha256"] != parents.get(
            "full_source_audit_sha256"
        ) or self._outputs["parents/source-file-audit.json"]["sha256"] != parents.get(
            "full_file_audit_sha256"
        ):
            raise SnapshotError("Parent audit hashes differ from contract")
        summary = json.loads(metadata_raw["parents/source-audit-summary.json"])
        if (
            summary.get("source_sha256") != SUPPLIED_ARCHIVE_SHA256
            or summary.get("primary_files") != 8539
            or summary.get("primary_rows") != 17453243
            or summary.get("all_mirrors_byte_identical") is not True
        ):
            raise SnapshotError("Full authenticated parent archive audit required")
        audited = json.loads(metadata_raw["parents/source-file-audit.json"])
        audit_by_member = {entry["path"]: entry for entry in audited}
        if len(audited) != 8539 or len(audit_by_member) != 8539:
            raise SnapshotError("Full unique file audit required")
        journal = [json.loads(line) for line in metadata_raw["inventory.jsonl"].splitlines()]
        inventory = manifest.get("source_inventory", [])
        if journal != inventory or len(inventory) != 8539:
            raise SnapshotError("Manifest and authenticated inventory journal differ")
        self._assets = {}
        seen_members, paths, flags = set(), set(), Counter()
        total_rows, empty = 0, 0
        for entry in inventory:
            asset = entry.get("asset_id")
            match = re.fullmatch(r"(stock|etf):([a-z0-9][a-z0-9._^-]*)\.us", asset or "")
            if match is None or asset in self._assets:
                raise SnapshotError("Invalid or repeated source asset ID")
            kind, symbol = match.groups()
            member = ("Stocks/" if kind == "stock" else "ETFs/") + symbol + ".us.txt"
            audited_member = "Data/" + member
            source = audit_by_member.get(audited_member)
            if (
                source is None
                or member in seen_members
                or entry.get("source_member") != member
                or entry.get("audited_member") != audited_member
                or entry.get("source_sha256") != source.get("sha256")
                or entry.get("source_bytes") != source.get("bytes")
                or entry.get("expected_rows") != source.get("rows")
                or entry.get("rows") != source.get("rows")
            ):
                raise SnapshotError("Source identity/count differs from authenticated audit")
            seen_members.add(member)
            if source["rows"] == 0:
                if (
                    entry.get("status") != "empty_source_preserved_in_inventory"
                    or "output" in entry
                ):
                    raise SnapshotError("Empty source must remain inventory-only")
                empty += 1
            else:
                relative = f"bars/asset_type={kind}/source_symbol={symbol}.us/part-00000.parquet"
                output = entry.get("output", {})
                if (
                    entry.get("status") != "written"
                    or output.get("path") != relative
                    or output != self._outputs.get(relative)
                    or output.get("asset_id") != asset
                    or output.get("rows") != source["rows"]
                    or relative in paths
                ):
                    raise SnapshotError("Source/output partition identity differs")
                paths.add(relative)
            total_rows += source["rows"]
            flags.update(entry.get("quality_flag_counts", {}))
            self._assets[asset] = dict(entry)
        if (
            total_rows != 17453243
            or empty != 32
            or len(paths) != 8507
            or set(self._outputs) != metadata | paths
            or dict(flags) != manifest.get("quality_flag_counts")
        ):
            raise SnapshotError("Full source/output/quality totals do not reconcile")
        self._layout_guard()
        status = json.loads(self._read("status.json"))
        if (
            status.get("status") != "complete"
            or status.get("complete") is not True
            or status.get("contract_sha256") != manifest["snapshot_id"]
            or status.get("errors") != []
            or status.get("counts") != counts
        ):
            raise SnapshotError("Completion status differs from manifest")
        self.manifest_sha256 = expected_manifest_sha256
        self.snapshot_id = manifest["snapshot_id"]
        self.archive_sha256 = SUPPLIED_ARCHIVE_SHA256
        self._contract = contract
        self._source_audits = audit_by_member
        self._schema_raw = metadata_raw["schema.arrow"]
        self._check_source_contract()
        self._session_lock = Lock()
        self._active_session = None

    def verified_read_session(self):
        """Yield this loader with cheaper guards until a mandatory full postflight.

        Only the owner thread can read during the session. This is an optional
        metadata optimization: every partition is still hashed in its entirety
        before Arrow sees its bytes. Callers must leave this context successfully
        before sealing derived outputs or writing a COMPLETE marker.
        """
        return _VerifiedReadSession(self)

    def _full_session_check(self):
        # A fresh constructor authenticates all metadata bytes, inventories,
        # completion/source guards and layout using the external manifest pin.
        fresh = MarketSnapshot(self.root, self.manifest_sha256)
        if (
            fresh.snapshot_id != self.snapshot_id
            or fresh.archive_sha256 != self.archive_sha256
            or fresh._outputs != self._outputs
            or fresh._assets != self._assets
            or fresh._contract != self._contract
            or fresh._source_audits != self._source_audits
            or fresh._schema_raw != self._schema_raw
        ):
            raise SnapshotError("In-memory metadata differs from authenticated snapshot")

    def _metadata_identities(self):
        names = {"manifest.json", "COMPLETE", "status.json"} | {
            name for name in self._outputs if not name.endswith(".parquet")
        }
        paths = {name: self._path(name) for name in names}
        src = Path(__file__).resolve().parents[2]
        paths.update(
            {"source:" + name: src / name for name in self._contract["source_code_sha256"]}
        )
        identities = {}
        for name, path in paths.items():
            self._safe(path)
            info = path.lstat()
            identities[name] = (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                getattr(info, "st_file_attributes", 0),
            )
        return identities

    def _read_guard(self):
        session = self._active_session
        if session is None:
            self._completion_guard(self.manifest_sha256)
            self._check_source_contract()
            return
        if session.owner != get_ident():
            raise SnapshotError("Concurrent read during verified session forbidden")
        self._completion_guard(self.manifest_sha256, verify_manifest=False)
        if self._metadata_identities() != session.identities:
            raise SnapshotError("Verified session metadata/source identity changed")

    def _layout_guard(self):
        allowed = set(self._outputs) | {"manifest.json", "COMPLETE", "status.json"}
        actual = set()
        for path in self.root.rglob("*"):
            self._safe(path)
            if path.is_file():
                actual.add(path.relative_to(self.root).as_posix())
        if actual != allowed:
            raise SnapshotError("Extra or missing snapshot files")

    @staticmethod
    def _safe(path):
        for item in (path, *path.parents):
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise SnapshotError("Symlink/reparse snapshot paths forbidden")

    def _path(self, name):
        if not isinstance(name, str):
            raise SnapshotError("Relative output path required")
        posix = PurePosixPath(name)
        if (
            not name
            or posix.is_absolute()
            or any(x in ("..", ".") for x in posix.parts)
            or "\\" in name
            or ":" in name
            or posix.as_posix() != name
        ):
            raise SnapshotError("Unsafe snapshot path")
        path = self.root.joinpath(*posix.parts)
        self._safe(path)
        return path

    def _read(self, name):
        return self._path(name).read_bytes()

    def _completion_guard(self, manifest_hash, *, verify_manifest=True):
        if (self.root / "FAILURE.json").exists():
            raise SnapshotError("FAILURE snapshot forbidden")
        marker = self.root / "COMPLETE"
        if not marker.is_file():
            raise SnapshotError("COMPLETE marker required; staging snapshot refused")
        if self._read("COMPLETE") != (manifest_hash + "\n").encode():
            raise SnapshotError("COMPLETE does not match external manifest hash")
        if verify_manifest and _sha(self._read("manifest.json")) != manifest_hash:
            raise SnapshotError("External manifest changed")

    def _authenticated(self, name):
        raw = self._read(name)
        item = self._outputs[name]
        if len(raw) != item["bytes"] or _sha(raw) != item["sha256"]:
            raise SnapshotError("OUTPUT_HASH_MISMATCH: " + name)
        return raw

    def _check_source_contract(self):
        sources = self._contract.get("source_code_sha256", {})
        expected = {f"quantlab/data/{name}.py" for name in ("archive", "calendar", "ingest")}
        if set(sources) != expected:
            raise SnapshotError("Normalization source closure differs")
        src = Path(__file__).resolve().parents[2]
        for name, digest in sources.items():
            if not _hash(digest) or _sha((src / name).read_bytes()) != digest:
                raise SnapshotError("Normalization implementation differs: " + name)

    @property
    def asset_ids(self):
        return tuple(self._assets)

    def rows(self, asset_id):
        """Read one authenticated partition, retaining calendar fields separately."""
        self._read_guard()
        entry = self._assets[asset_id]
        if entry["rows"] == 0:
            return ()
        raw = self._authenticated(entry["output"]["path"])
        # Authentication is complete before the first Arrow/parquet parse.
        import pyarrow as pa
        import pyarrow.parquet as pq

        from .ingest import _arrow_schema

        if version("pyarrow") != self._contract["runtime"]["pyarrow"]:
            raise SnapshotError("Arrow runtime differs from normalization contract")
        schema = pa.ipc.read_schema(pa.BufferReader(self._schema_raw))
        if schema != _arrow_schema(pa):
            raise SnapshotError("Unsupported normalized schema")
        table = pq.ParquetFile(pa.BufferReader(raw)).read()
        if table.schema != schema or table.num_rows != entry["rows"]:
            raise SnapshotError("Partition row count or schema differs")
        rows = table.to_pylist()
        previous = None
        previous_line = 1
        flags = Counter()
        kind, source_symbol = asset_id.split(":", 1)
        for row in rows:
            if (
                row["asset_id"] != asset_id
                or row["source_member"] != entry["source_member"]
                or row["asset_type"] != kind
                or row["source_symbol"] != source_symbol
                or row["source_member_sha256"] != entry["source_sha256"]
                or row["source_archive_sha256"] != self.archive_sha256
                or row["price_basis"] != self._contract["price_basis"]
                or not _hash(row["source_row_sha256"])
                or type(row["source_row_number"]) is not int
                or row["source_row_number"] <= previous_line
                or (previous is not None and row["session"] <= previous)
                or not all(
                    math.isfinite(row[k])
                    for k in ("open", "high", "low", "close", "volume", "open_interest")
                )
            ):
                raise SnapshotError("Partition source row identity/value violation")
            previous = row["session"]
            previous_line = row["source_row_number"]
            quality = []
            if min(row[k] for k in ("open", "high", "low", "close")) <= 0:
                quality.append("NONPOSITIVE_PRICE")
            if (
                row["low"] > min(row["open"], row["close"])
                or row["high"] < max(row["open"], row["close"])
                or row["low"] > row["high"]
            ):
                quality.append("INCONSISTENT_OHLC")
            if row["volume"] < 0:
                quality.append("NEGATIVE_VOLUME")
            if row["volume"] == 0:
                quality.append("ZERO_VOLUME")
            if not row["volume"].is_integer():
                quality.append("FRACTIONAL_VOLUME")
            if row["open_interest"] != 0:
                quality.append("NONZERO_OPEN_INTEREST")
            if not row["is_xnys_session"]:
                quality.append("OFF_XNYS_SESSION")
            if sorted(quality) != row["quality_flags"]:
                raise SnapshotError("Normalized findings differ from unchanged source values")
            flags.update(row["quality_flags"])
            times = (row["xnys_open_at"], row["xnys_close_at"], row["assumed_available_at"])
            if row["is_xnys_session"]:
                if (
                    any(t is None or t.utcoffset() is None for t in times)
                    or not times[0] < times[1] < times[2]
                    or "OFF_XNYS_SESSION" in row["quality_flags"]
                ):
                    raise SnapshotError("Invalid derived session timing")
            elif (
                any(t is not None for t in times) or "OFF_XNYS_SESSION" not in row["quality_flags"]
            ):
                raise SnapshotError("Off-calendar source row was shifted or assigned timing")
        if dict(flags) != entry["quality_flag_counts"]:
            raise SnapshotError("Partition quality counts differ")
        audited = self._source_audits[entry["audited_member"]]
        if (
            str(rows[0]["session"]) != audited["first_date"]
            or str(rows[-1]["session"]) != audited["last_date"]
        ):
            raise SnapshotError("Partition date coverage differs from source audit")
        return tuple(rows)

    def bars(self, asset_type: str, symbol: str):
        if asset_type not in ("stock", "etf") or not isinstance(symbol, str):
            raise SnapshotError("Source asset type and symbol required")
        symbol = symbol.lower().removesuffix(".us")
        asset = asset_type + ":" + symbol + ".us"
        result = []
        for row in self.rows(asset):
            # Bar carries the original parser's financial flags. The complete
            # normalized findings remain in rows(), including volume/calendar metadata.
            raw_flags = tuple(
                flag
                for flag in ("NONPOSITIVE_PRICE", "INCONSISTENT_OHLC", "NEGATIVE_VOLUME")
                if flag in row["quality_flags"]
            )
            result.append(
                Bar(
                    asset,
                    row["session"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["open_interest"],
                    row["source_member"],
                    row["source_row_number"],
                    row["source_member_sha256"],
                    row["source_row_sha256"],
                    raw_flags,
                )
            )
        return tuple(result)

    def verify_all(self):
        """Hash/parse each partition once; return counts, not a performance claim."""
        counts = Counter()
        for asset in self.asset_ids:
            rows = self.rows(asset)
            counts["source_files"] += 1
            counts["rows"] += len(rows)
            counts["nonempty_files" if rows else "empty_files"] += 1
        self._completion_guard(self.manifest_sha256)
        self._check_source_contract()
        self._layout_guard()
        return {
            "manifest_sha256": self.manifest_sha256,
            "snapshot_id": self.snapshot_id,
            "archive_sha256": self.archive_sha256,
            "counts": dict(counts),
            "complete": counts["source_files"] == 8539
            and counts["rows"] == 17453243
            and counts["empty_files"] == 32
            and counts["nonempty_files"] == 8507,
        }
