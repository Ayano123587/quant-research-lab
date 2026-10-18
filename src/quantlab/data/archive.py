"""Read the supplied market archive without extracting or inventing observations."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import stat
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath

SUPPLIED_ARCHIVE_SHA256 = "d9317c8fb2d63b9b00db5f933b6c9639d2bf7ea3b918169bb5cec5903dce85a1"
COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume", "OpenInt")


class ArchiveDataError(ValueError):
    """A source violation that must remain visible to the caller."""


@dataclass(frozen=True)
class Bar:
    asset_id: str
    session: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float
    source_member: str
    row_number: int
    source_sha256: str
    row_sha256: str
    quality_flags: tuple[str, ...] = ()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class MarketArchive:
    """Authenticate once, then read individual real histories from that same handle.

    Source IDs identify files in this snapshot, not permanent securities. The two
    directory trees must agree byte-for-byte; they are not separate populations.
    Invalid financial values are flagged rather than repaired or silently dropped.
    """

    def __init__(self, path: str | Path, expected_archive_sha256: str):
        self.path = Path(path)
        if not isinstance(expected_archive_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_archive_sha256
        ):
            raise ArchiveDataError("An externally recorded archive SHA256 is required")
        self._handle = self.path.open("rb")
        try:
            self.sha256 = hashlib.file_digest(self._handle, "sha256").hexdigest()
            if self.sha256 != expected_archive_sha256:
                raise ArchiveDataError("ARCHIVE_HASH_MISMATCH")
            self._handle.seek(0)
            self._zip = zipfile.ZipFile(self._handle)
            seen = set()
            for info in self._zip.infolist():
                name = PurePosixPath(info.filename)
                if (
                    info.filename in seen
                    or name.is_absolute()
                    or ".." in name.parts
                    or "\\" in info.filename
                    or ":" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)
                ):
                    raise ArchiveDataError("UNSAFE_OR_DUPLICATE_ARCHIVE_MEMBER")
                seen.add(info.filename)
            self._names = seen
        except BaseException:
            self._handle.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._zip.close()
        self._handle.close()

    def source_bytes(self, asset_type: str, symbol: str) -> tuple[str, bytes]:
        if asset_type not in ("stock", "etf"):
            raise ArchiveDataError("asset_type must be stock or etf")
        if not isinstance(symbol, str):
            raise ArchiveDataError("A source symbol is required")
        symbol = symbol.lower()
        if symbol.endswith(".us"):
            symbol = symbol[:-3]
        if not re.fullmatch(r"[a-z0-9][a-z0-9._^-]*", symbol):
            raise ArchiveDataError("Invalid source symbol")
        directory = "Stocks" if asset_type == "stock" else "ETFs"
        member = directory + "/" + symbol + ".us.txt"
        mirror = "Data/" + member
        if member not in self._names or mirror not in self._names:
            raise ArchiveDataError("SOURCE_OR_MIRROR_MISSING: " + member)
        raw = self._zip.read(member)
        if raw != self._zip.read(mirror):
            raise ArchiveDataError("CONFLICTING_ARCHIVE_MIRRORS: " + member)
        return member, raw

    def bars(self, asset_type: str, symbol: str) -> tuple[Bar, ...]:
        member, raw = self.source_bytes(asset_type, symbol)
        if not raw:
            return ()
        source_sha = hashlib.sha256(raw).hexdigest()
        lines = raw.splitlines(keepends=True)
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if tuple(next(reader, ())) != COLUMNS:
            raise ArchiveDataError("UNEXPECTED_COLUMNS: " + member)
        asset_id = asset_type + ":" + PurePosixPath(member).stem
        bars = []
        previous = None
        previous_line = reader.line_num
        for record in reader:
            line = previous_line + 1
            span = b"".join(lines[previous_line : reader.line_num])
            previous_line = reader.line_num
            try:
                if len(record) != len(COLUMNS):
                    raise ValueError("column count")
                session = date.fromisoformat(record[0])
                values = tuple(float(value) for value in record[1:])
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("nonfinite numeric value")
                if previous is not None and session <= previous:
                    raise ValueError("duplicate or unordered session")
            except (ValueError, OverflowError) as error:
                raise ArchiveDataError(f"INVALID_SOURCE_ROW: {member}:{line}: {error}") from error
            opening, high, low, closing, volume, interest = values
            flags = []
            if min(opening, high, low, closing) <= 0:
                flags.append("NONPOSITIVE_PRICE")
            if low > min(opening, closing) or high < max(opening, closing) or low > high:
                flags.append("INCONSISTENT_OHLC")
            if volume < 0:
                flags.append("NEGATIVE_VOLUME")
            bars.append(
                Bar(
                    asset_id,
                    session,
                    opening,
                    high,
                    low,
                    closing,
                    volume,
                    interest,
                    member,
                    line,
                    source_sha,
                    hashlib.sha256(span).hexdigest(),
                    tuple(flags),
                )
            )
            previous = session
        return tuple(bars)


def load_bars(archive, asset_type, symbol, expected_archive_sha256):
    with MarketArchive(archive, expected_archive_sha256) as source:
        return source.bars(asset_type, symbol)
