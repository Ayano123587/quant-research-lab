"""Stream and audit the supplied real-price ZIP without extracting raw data.

Outputs are private data artifacts. Audit completion does not certify price
adjustments, exchange membership, tradeability, or economic validity.
"""

from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import json
import math
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]


def _save(output, name, value):
    (output / name).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _audit(source, output, expected_archive_sha256):
    SOURCE, OUT = source, output
    with SOURCE.open("rb") as handle:
        archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if archive_sha != expected_archive_sha256:
        raise ValueError("ARCHIVE_HASH_MISMATCH")
    inventory, summaries, content_hashes = [], [], {}
    good_dates = {}
    headers = collections.Counter()
    with zipfile.ZipFile(SOURCE) as archive:
        members = archive.infolist()
        name_counts = collections.Counter(x.filename for x in members)
        for number, member in enumerate(members):
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            safe = not (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or ":" in member.filename
                or stat.S_ISLNK(mode)
            )
            item = dict(
                path=member.filename,
                bytes=member.file_size,
                compressed_bytes=member.compress_size,
                crc32=f"{member.CRC:08x}",
                encrypted=bool(member.flag_bits & 1),
                safe_relative_regular_path=safe,
            )
            inventory.append(item)
            if member.is_dir() or not safe or item["encrypted"]:
                continue
            digest = hashlib.sha256()
            parse = member.filename.startswith("Data/") and member.filename.endswith(".txt")
            report = dict(
                path=member.filename,
                symbol=path.name.removesuffix(".us.txt"),
                asset_type=path.parent.name,
                rows=0,
                first_date=None,
                last_date=None,
                issues={},
                examples={},
                zero_volume_rows=0,
                zero_price_rows=0,
                weekend_rows=0,
                nonzero_open_interest_rows=0,
            )
            dates, previous = set(), None

            def issue(kind, row_number, raw, report=report):
                report["issues"][kind] = report["issues"].get(kind, 0) + 1
                if len(report["examples"].setdefault(kind, [])) < 3:
                    report["examples"][kind].append(dict(line=row_number, raw=raw[:180]))

            with archive.open(member) as handle:
                for line_number, raw in enumerate(handle, 1):
                    digest.update(raw)
                    if not parse:
                        continue
                    try:
                        text = raw.decode("utf-8-sig" if line_number == 1 else "utf-8").rstrip(
                            "\r\n"
                        )
                    except UnicodeDecodeError:
                        issue("invalid_utf8", line_number, repr(raw))
                        continue
                    try:
                        row = next(csv.reader([text], strict=True))
                    except csv.Error:
                        issue("invalid_csv", line_number, text)
                        continue
                    if line_number == 1:
                        report["header"] = row
                        headers[tuple(row)] += 1
                        if row != EXPECTED:
                            issue("unexpected_header", line_number, text)
                        continue
                    if not row:
                        issue("blank_line", line_number, text)
                        continue
                    report["rows"] += 1
                    if len(row) != 7:
                        issue("wrong_column_count", line_number, text)
                        continue
                    date = row[0]
                    if date not in good_dates:
                        try:
                            parsed = datetime.date.fromisoformat(date)
                            good_dates[date] = (
                                parsed.weekday()
                                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
                                else None
                            )
                        except ValueError:
                            good_dates[date] = None
                    weekday = good_dates[date]
                    if weekday is None:
                        issue("invalid_date", line_number, text)
                    else:
                        if date in dates:
                            issue("duplicate_date", line_number, text)
                        dates.add(date)
                        if previous is not None and date < previous:
                            issue("descending_date", line_number, text)
                        previous = date
                        report["first_date"] = min(report["first_date"] or date, date)
                        report["last_date"] = max(report["last_date"] or date, date)
                        report["weekend_rows"] += weekday >= 5
                    try:
                        op, hi, lo, cl, volume, oi = map(float, row[1:])
                    except ValueError:
                        issue("nonnumeric", line_number, text)
                        continue
                    values = [op, hi, lo, cl, volume, oi]
                    if not all(math.isfinite(x) for x in values):
                        issue("nonfinite", line_number, text)
                        continue
                    if min(op, hi, lo, cl) < 0:
                        issue("negative_price", line_number, text)
                    if min(op, hi, lo, cl) == 0:
                        report["zero_price_rows"] += 1
                    tolerance = 1e-9 * max(1, abs(hi), abs(lo), abs(op), abs(cl))
                    if hi + tolerance < max(op, cl, lo) or lo - tolerance > min(op, cl, hi):
                        issue("ohlc_inconsistent", line_number, text)
                    if volume < 0:
                        issue("negative_volume", line_number, text)
                    if not volume.is_integer():
                        issue("fractional_volume", line_number, text)
                    report["zero_volume_rows"] += volume == 0
                    report["nonzero_open_interest_rows"] += oi != 0
            item["sha256"] = digest.hexdigest()
            content_hashes[member.filename] = item["sha256"]
            if parse:
                report["sha256"] = item["sha256"]
                report["bytes"] = member.file_size
                report["unique_dates"] = len(dates)
                summaries.append(report)
            if (number + 1) % 1000 == 0:
                print("Audited members", number + 1, "of", len(members), flush=True)
                _save(
                    OUT,
                    "progress.json",
                    dict(
                        members_audited=number + 1,
                        total_members=len(members),
                        parsed_primary_files=len(summaries),
                    ),
                )
    mirrors = []
    for name, digest in content_hashes.items():
        if name.startswith("Data/"):
            mirror = name[5:]
            mirrors.append(
                dict(
                    primary=name,
                    mirror=mirror,
                    identical_sha256=content_hashes.get(mirror) == digest,
                )
            )
    groups = collections.defaultdict(list)
    for row in summaries:
        if row["rows"]:
            groups[row["sha256"]].append(row["path"])
    nonempty = [x for x in summaries if x["rows"]]
    aggregate = dict(
        schema="supplied_real_daily_price_archive_audit_v1",
        source=str(SOURCE),
        source_bytes=SOURCE.stat().st_size,
        source_sha256=archive_sha,
        user_override=(
            "Real supplied data only. No synthetic data, generated fixtures, "
            "simulation or model fitting used."
        ),
        total_members=len(inventory),
        total_uncompressed_bytes=sum(x["bytes"] for x in inventory),
        unsafe_members=[x["path"] for x in inventory if not x["safe_relative_regular_path"]],
        encrypted_members=[x["path"] for x in inventory if x["encrypted"]],
        repeated_member_names=[k for k, v in name_counts.items() if v > 1],
        primary_files=len(summaries),
        primary_rows=sum(x["rows"] for x in summaries),
        asset_file_counts=dict(collections.Counter(x["asset_type"] for x in summaries)),
        asset_row_counts={
            k: sum(x["rows"] for x in summaries if x["asset_type"] == k) for k in ["Stocks", "ETFs"]
        },
        headers=[dict(columns=list(k), files=v) for k, v in headers.items()],
        empty_primary_files=[x["path"] for x in summaries if not x["rows"]],
        earliest_date=min((x["first_date"] for x in nonempty), default=None),
        latest_date=max((x["last_date"] for x in nonempty), default=None),
        terminal_date_distribution=dict(
            sorted(collections.Counter(x["last_date"] for x in nonempty).items())
        ),
        issue_totals=dict(
            sum((collections.Counter(x["issues"]) for x in summaries), collections.Counter())
        ),
        issue_file_counts=dict(collections.Counter(k for x in summaries for k in x["issues"])),
        zero_volume_rows=sum(x["zero_volume_rows"] for x in summaries),
        zero_price_rows=sum(x["zero_price_rows"] for x in summaries),
        weekend_rows=sum(x["weekend_rows"] for x in summaries),
        nonzero_open_interest_rows=sum(x["nonzero_open_interest_rows"] for x in summaries),
        mirror_pairs=len(mirrors),
        all_mirrors_byte_identical=all(x["identical_sha256"] for x in mirrors),
        duplicate_nonempty_primary_contents=[v for v in groups.values() if len(v) > 1],
        limitations=[
            (
                "OHLCV headers provide no adjusted-close, corporate-action, dividend, split or "
                "delisting-return fields. Adjustment provenance is unverified; filenames and "
                "historical prices cannot prove it."
            ),
            (
                "Terminal dates differ. Early termination does not prove delisting: archive "
                "has no delisting/status table or point-in-time security master."
            ),
            (
                "No historical constituent membership or identifier-change mapping is supplied. "
                "A present-day full-symbol universe may introduce survivorship/selection bias."
            ),
            (
                "Historical prices may have been retrospectively adjusted; absent source metadata "
                "and point-in-time corporate actions prevent verifying "
                "when adjustments were available."
            ),
            (
                "Daily OHLCV cannot validate intraday fills, market impact, bid-ask spread or "
                "high-frequency execution. Volume is not executable liquidity."
            ),
            (
                "No exchange/calendar/currency metadata is established solely by .us filename. "
                "Weekend rows require inspection, not automatic assumption."
            ),
            (
                "Duplicate archive trees must be deduplicated by actual content hashes before "
                "ingestion. No raw data is to be committed publicly."
            ),
        ],
    )
    _save(OUT, "inventory.json", inventory)
    _save(OUT, "file-summaries.json", summaries)
    _save(OUT, "mirror-comparison.json", mirrors)
    _save(OUT, "audit-summary.json", aggregate)
    return aggregate


def audit_archive(source: str | Path, output: str | Path, expected_archive_sha256: str) -> dict:
    """Audit every member and primary history, preserving a fresh output directory.

    Expected hashes come from an external source receipt. This method does not
    fit models or generate observations. It examines real rows, flags anomalies,
    and compares the two stock/ETF directory trees using actual content SHA256.
    Numeric quality findings remain in the report; no data is repaired or dropped.
    """
    if not isinstance(expected_archive_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_archive_sha256
    ):
        raise ValueError("An externally recorded archive SHA256 is required")
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.UTC).isoformat()
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def status(state, **extra):
        target = output / "status.json.pending"
        target.write_text(
            json.dumps(
                {
                    "schema": "market_archive_audit_status_v1",
                    "status": state,
                    "complete": state == "complete",
                    "started_at": started,
                    "source_archive_sha256": expected_archive_sha256,
                    "source_code_sha256": code_sha256,
                    **extra,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(target, output / "status.json")

    status("staging")
    try:
        source_before = source.stat()
        result = _audit(source, output, expected_archive_sha256)
        source_after = source.stat()
        if (source_before.st_size, source_before.st_mtime_ns) != (
            source_after.st_size,
            source_after.st_mtime_ns,
        ):
            raise ValueError("ARCHIVE_CHANGED_DURING_AUDIT")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code_sha256:
            raise ValueError("AUDIT_SOURCE_CHANGED_DURING_EXECUTION")
        result["audit_source_code_sha256"] = code_sha256
        result["audit_started_at"] = started
        result["audit_finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        result["audit_complete"] = True
        result["passed_structural_checks"] = not any(
            [
                result["unsafe_members"],
                result["encrypted_members"],
                result["repeated_member_names"],
                not result["primary_files"],
                not result["all_mirrors_byte_identical"],
                any(
                    k in result["issue_totals"]
                    for k in (
                        "unexpected_header",
                        "invalid_utf8",
                        "invalid_csv",
                        "wrong_column_count",
                        "invalid_date",
                        "duplicate_date",
                        "descending_date",
                        "nonnumeric",
                        "nonfinite",
                    )
                ),
            ]
        )
        _save(output, "audit-summary.json", result)
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file() and p.name not in ["status.json", "progress.json"]
        }
        status(
            "complete",
            files_sha256=hashes,
            passed_structural_checks=result["passed_structural_checks"],
        )
        return result
    except BaseException as error:
        failure = {
            "error_type": type(error).__name__,
            "message": str(error),
            "complete": False,
            "source_archive_sha256": expected_archive_sha256,
        }
        _save(output, "failure.json", failure)
        status("failed", failure=failure)
        raise
