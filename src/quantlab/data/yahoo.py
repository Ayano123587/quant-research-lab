"""Authenticate a distinct whole-vintage Yahoo ETF price-index research release.

Derived OHLC are a declared client transformation of observed provider cells,
not vendor-published adjusted opening prints or actual dividend cash settlement.
The supplied stock archive is neither edited nor claimed as this data's parent.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from . import archive as archive_module
from . import calendar as calendar_module
from .archive import Bar, file_sha256
from .calendar import XNYSSchedule

SYMBOLS = ("SPY", "EFA", "EEM", "TLT", "LQD", "GLD", "DBC", "VNQ")
DOCUMENTS = (
    "yahoo-adjusted-close-primary.html",
    "yfinance-utils-primary-source.py",
    "yahoo-history-footnote-primary-web-evidence.json",
)
POLICY = {
    "id": "yahoo_published_same_row_adjclose_ratio_uniform_ohlc_v1",
    "factor": "published_adjclose / observed_close",
    "all_four_prices": "observed_field * factor_including_close",
    "volume": "observed_provider_volume_unchanged",
    "open_interest": "not_provided_Bar_field_is_NaN_missing_metadata_not_an_observation",
    "published_adjclose": "preserved_separately_not_substituted_for_computed_close",
    "units": "retrospectively_adjusted_price_index_research_units_not_actual_shares",
    "not_claimed": [
        "vendor_published_adjusted_open",
        "auction_fill",
        "actual_dividend_cash_settlement",
    ],
    "double_crediting": "do_not_add_dividend_cash_to_adjusted_price_index_returns",
    "yahoo_basis": "https://ca.finance.yahoo.com/quote/SPY/history/",
    "yahoo_adjusted_close": "https://help.yahoo.com/kb/SLN28256.html",
    "client_algorithm": "https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py#L461-L477",
    "availability": (
        "actual_XNYS_close_plus_one_minute_assumption_not_historical_publication_evidence"
    ),
    "vintage": "whole_observed_download_vintage_no_cross_provider_or_2017_archive_splice",
}


class YahooSourceError(ValueError):
    pass


def _raw(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _bound(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise YahooSourceError("Regular authenticated source file required")
    raw = path.read_bytes()
    if _sha(raw) != expected:
        raise YahooSourceError("External source SHA mismatch")
    return raw


def _source_code():
    return {
        name: file_sha256(Path(module.__file__))
        for name, module in (
            ("yahoo", __import__(__name__, fromlist=["__file__"])),
            ("archive_Bar_definition", archive_module),
            ("calendar", calendar_module),
        )
    }


@dataclass(frozen=True)
class YahooObservation:
    asset_id: str
    session: str
    timestamp: int
    raw_array_index: int
    raw_response_sha256: str
    raw_row_sha256: str
    observed_open: float
    observed_high: float
    observed_low: float
    observed_close: float
    observed_volume: float
    published_adjclose: float
    factor: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_roundoff: float
    derived_row_sha256: str


def _observations(raw, symbol, schedule):
    payload = json.loads(raw)
    chart = payload["chart"]
    if chart.get("error") is not None or len(chart.get("result") or ()) != 1:
        raise YahooSourceError("Exactly one successful actual provider result required")
    data = chart["result"][0]
    meta = data["meta"]
    if (
        meta.get("symbol") != symbol
        or meta.get("currency") != "USD"
        or meta.get("exchangeTimezoneName") != "America/New_York"
    ):
        raise YahooSourceError("Provider identity/currency/timezone mismatch")
    times = data["timestamp"]
    quote = data["indicators"]["quote"][0]
    adjusted = data["indicators"]["adjclose"][0]["adjclose"]
    if any(
        len(quote[field]) != len(times) for field in ("open", "high", "low", "close", "volume")
    ) or len(adjusted) != len(times):
        raise YahooSourceError("Provider cell-array cardinality mismatch")
    response_sha, result, seen = _sha(raw), [], set()
    for index, timestamp in enumerate(times):
        if type(timestamp) is not int:
            raise YahooSourceError("Actual integer provider timestamp required")
        session = datetime.fromtimestamp(timestamp, UTC).date()
        if not date(2005, 2, 25) <= session <= date(2017, 11, 10):
            raise YahooSourceError("Observed provider session outside declared whole-capture range")
        if session in seen or (result and session.isoformat() <= result[-1].session):
            raise YahooSourceError("Unique chronological source sessions required")
        seen.add(session)
        schedule.session(session)
        cells = {field: quote[field][index] for field in ("open", "high", "low", "close", "volume")}
        cells["adjclose"] = adjusted[index]
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in cells.values()
        ):
            raise YahooSourceError("Missing/nonfinite provider cell; no imputation")
        if (
            any(cells[k] <= 0 for k in ("open", "high", "low", "close", "adjclose"))
            or cells["volume"] < 0
        ):
            raise YahooSourceError("Positive observed prices/nonnegative observed volume required")
        if cells["low"] > min(cells[k] for k in ("open", "high", "close")) or cells["high"] < max(
            cells[k] for k in ("open", "low", "close")
        ):
            raise YahooSourceError("Inconsistent observed OHLC")
        factor = cells["adjclose"] / cells["close"]
        prices = {field: cells[field] * factor for field in ("open", "high", "low", "close")}
        if (
            not math.isfinite(factor)
            or factor <= 0
            or any(not math.isfinite(v) or v <= 0 for v in prices.values())
        ):
            raise YahooSourceError("Invalid derived price/factor; no replacement")
        residual = prices["close"] - cells["adjclose"]
        if abs(residual) > 8 * math.ulp(max(prices["close"], cells["adjclose"])):
            raise YahooSourceError("Computed close exceeds explicit eight-ULP roundoff bound")
        raw_identity = {
            "response_sha256": response_sha,
            "array_index": index,
            "timestamp": timestamp,
            "observed_cells": cells,
        }
        row = {
            "asset_id": "etf:" + symbol.lower() + ".us",
            "session": session.isoformat(),
            "timestamp": timestamp,
            "raw_array_index": index,
            "raw_response_sha256": response_sha,
            "raw_row_sha256": _sha(_raw(raw_identity)),
            **{
                "observed_" + field: cells[field]
                for field in ("open", "high", "low", "close", "volume")
            },
            "published_adjclose": cells["adjclose"],
            "factor": factor,
            **prices,
            "volume": cells["volume"],
            "close_roundoff": residual,
        }
        row["derived_row_sha256"] = _sha(_raw({"policy_sha256": _sha(_raw(POLICY)), **row}))
        result.append(YahooObservation(**row))
    return tuple(result), meta, data.get("events", {})


def _raw_bundle(root, expected_sha):
    raw = _bound(root / "raw-manifest.json", expected_sha)
    manifest = json.loads(raw)
    if manifest.get("schema") != "yahoo_actual_eight_etf_raw_capture_v1" or set(
        manifest.get("files", {})
    ) != set(SYMBOLS):
        raise YahooSourceError("Externally pinned exact eight-ETF raw capture required")
    if set(manifest.get("primary_documents", {})) != set(DOCUMENTS):
        raise YahooSourceError("Preserved bound primary basis documents required")
    bound = {}
    for symbol in SYMBOLS:
        item = manifest["files"][symbol]
        if item["path"] != symbol + "-chart-raw.json" or not item["url"].startswith(
            "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?"
        ):
            raise YahooSourceError("Exact provider raw member/URL required")
        body = _bound(root / item["path"], item["sha256"])
        if len(body) != item["bytes"]:
            raise YahooSourceError("Raw member byte count mismatch")
        bound[symbol] = body
    return raw, manifest, bound


def build_yahoo_etf_release(raw_root, expected_raw_manifest_sha256, schedule, output):
    """Build a fresh separately identified release from all eight full responses."""
    if not isinstance(schedule, XNYSSchedule):
        raise YahooSourceError("Actual XNYS schedule required")
    raw_root, root = Path(raw_root).resolve(), Path(output).resolve()
    if root == raw_root or raw_root in root.parents:
        raise YahooSourceError("Separate immutable output outside raw capture required")
    raw_manifest, capture, bodies = _raw_bundle(raw_root, expected_raw_manifest_sha256)
    code = _source_code()
    signal = date(2007, 2, 28)
    position = schedule.sessions.index(signal)
    required_start, required_end = schedule.sessions[position - 252], date(2017, 11, 10)
    required = tuple(
        d.isoformat() for d in schedule.sessions if required_start <= d <= required_end
    )
    parsed = {}
    for symbol, body in bodies.items():
        rows, meta, events = _observations(body, symbol, schedule)
        missing = sorted(set(required) - {row.session for row in rows})
        if missing:
            raise YahooSourceError(
                f"{symbol}: missing exact required warmup/research source sessions: {missing}"
            )
        parsed[symbol] = rows, meta, events
    root.mkdir(parents=True, exist_ok=False)
    try:
        _write(root / "raw-manifest.json", raw_manifest)
        _write(root / "policy.json", _raw(POLICY))
        _write(root / "calendar.json", _raw(schedule.provenance))
        metadata = {}
        for symbol, (rows, meta, events) in parsed.items():
            _write(root / "raw" / (symbol + ".json"), bodies[symbol])
            _write(root / "observations" / (symbol + ".json"), _raw([asdict(row) for row in rows]))
            _write(root / "actions" / (symbol + ".json"), _raw(events))
            metadata[symbol] = {
                "provider_meta": meta,
                "row_count": len(rows),
                "first_observed": rows[0].session,
                "last_observed": rows[-1].session,
                "raw_sha256": _sha(bodies[symbol]),
                "raw_capture": capture["files"][symbol],
                "max_absolute_close_roundoff": max(abs(row.close_roundoff) for row in rows),
            }
        # Documentation is independently bound in the input capture; never execute it.
        for name, item in capture.get("primary_documents", {}).items():
            if Path(name).name != name:
                raise YahooSourceError("Flat primary-document name required")
            _write(root / "primary-documents" / name, _bound(raw_root / name, item["sha256"]))
        _write(root / "source-metadata.json", _raw(metadata))
        if (
            _source_code() != code
            or _sha(_raw_bundle(raw_root, expected_raw_manifest_sha256)[0])
            != expected_raw_manifest_sha256
        ):
            raise YahooSourceError("Source/input changed during release construction")
        manifest = {
            "schema": "yahoo_eight_etf_price_index_release_v1",
            "complete": True,
            "raw_bundle_manifest_sha256": expected_raw_manifest_sha256,
            "source_namespace": "Yahoo_current_download_vintage",
            "original_stock_archive_is_not_parent": True,
            "source_code_sha256": code,
            "python": platform.python_version(),
            "policy_sha256": _sha(_raw(POLICY)),
            "calendar": schedule.provenance,
            "required_source_start": required_start.isoformat(),
            "required_source_end": required_end.isoformat(),
            "symbols": list(SYMBOLS),
            "metadata": metadata,
            "artifacts": {
                p.relative_to(root).as_posix(): {
                    "sha256": file_sha256(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        }
        _write(root / "manifest.json", _raw(manifest))
        digest = file_sha256(root / "manifest.json")
        _write(root / "COMPLETE", (digest + "\n").encode())
        return {
            "manifest_sha256": digest,
            "raw_bundle_manifest_sha256": expected_raw_manifest_sha256,
            "output": str(root),
            "symbols": list(SYMBOLS),
        }
    except BaseException as error:
        _write(
            root / "FAILURE.json",
            _raw({"complete": False, "type": type(error).__name__, "message": str(error)}),
        )
        raise


class YahooETFRelease:
    """Distinct authenticated source type; never impersonates MarketSnapshot."""

    def __init__(self, root, expected_manifest_sha256, schedule):
        self.root, self.schedule = Path(root), schedule
        self.manifest_sha256 = expected_manifest_sha256
        self._authenticate()
        self.source_bundle_sha256 = self.manifest["raw_bundle_manifest_sha256"]
        self.asset_ids = tuple("etf:" + s.lower() + ".us" for s in SYMBOLS)
        self.provenance = {
            "source_type": "YahooETFRelease",
            "release_manifest_sha256": self.manifest_sha256,
            "raw_bundle_manifest_sha256": self.source_bundle_sha256,
            "price_basis_policy_sha256": self.manifest["policy_sha256"],
            "calendar_sha256": self.schedule.provenance["schedule_sha256"],
            "price_basis": "provider_adjusted_close_ratio_price_index_research_proxy",
        }

    def _authenticate(self):
        if not isinstance(self.schedule, XNYSSchedule) or (self.root / "FAILURE.json").exists():
            raise YahooSourceError("Actual calendar and completed release required")
        if (self.root / "COMPLETE").read_text().strip() != self.manifest_sha256:
            raise YahooSourceError("Completion marker does not match external manifest pin")
        manifest = json.loads(_bound(self.root / "manifest.json", self.manifest_sha256))
        if (
            manifest.get("schema") != "yahoo_eight_etf_price_index_release_v1"
            or manifest.get("complete") is not True
            or manifest.get("symbols") != list(SYMBOLS)
            or manifest.get("calendar") != self.schedule.provenance
            or manifest.get("source_code_sha256") != _source_code()
            or manifest.get("policy_sha256") != _sha(_raw(POLICY))
            or manifest.get("python") != platform.python_version()
        ):
            raise YahooSourceError("Release schema/source/policy/calendar mismatch")
        actual_files = {
            p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()
        }
        if actual_files != set(manifest["artifacts"]) | {"manifest.json", "COMPLETE"}:
            raise YahooSourceError("Release artifact inventory mismatch")
        for name, item in manifest["artifacts"].items():
            path = self.root / name
            if self.root.resolve() not in path.resolve().parents:
                raise YahooSourceError("Artifact escapes release")
            if len(_bound(path, item["sha256"])) != item["bytes"]:
                raise YahooSourceError("Artifact byte count mismatch")
        self.manifest = manifest

    def rows(self, asset_id):
        if asset_id not in self.asset_ids:
            raise YahooSourceError("Declared ETF source identity required")
        self._authenticate()
        symbol = asset_id.split(":", 1)[1].removesuffix(".us").upper()
        raw_member = "raw/" + symbol + ".json"
        raw = _bound(self.root / raw_member, self.manifest["artifacts"][raw_member]["sha256"])
        rows, _, _ = _observations(raw, symbol, self.schedule)
        derived_member = "observations/" + symbol + ".json"
        stored = json.loads(
            _bound(self.root / derived_member, self.manifest["artifacts"][derived_member]["sha256"])
        )
        if [asdict(row) for row in rows] != stored:
            raise YahooSourceError(
                "Derived cells/provenance differ from authenticated observed source"
            )
        self._authenticate()
        return rows

    def bars(self, asset_type, symbol):
        if asset_type != "etf":
            raise YahooSourceError("This release contains ETFs only")
        symbol = symbol.lower().removesuffix(".us")
        rows = self.rows("etf:" + symbol + ".us")
        member = "observations/" + symbol.upper() + ".json"
        parent_sha = self.manifest["artifacts"][member]["sha256"]
        return tuple(
            Bar(
                row.asset_id,
                date.fromisoformat(row.session),
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
                math.nan,
                member,
                row.raw_array_index + 1,
                parent_sha,
                row.derived_row_sha256,
            )
            for row in rows
        )
