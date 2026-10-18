"""Immutable, truthful source identities for archive and provider-release data."""

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceIdentity:
    source_type: str
    calendar_sha256: str
    price_basis: str
    manifest_sha256: str | None = None
    archive_sha256: str | None = None
    raw_bundle_sha256: str | None = None
    price_basis_policy_sha256: str | None = None

    def __post_init__(self):
        for name in (
            "calendar_sha256",
            "manifest_sha256",
            "archive_sha256",
            "raw_bundle_sha256",
            "price_basis_policy_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            ):
                raise ValueError("Explicit valid source SHA256 required: " + name)
        if (
            not self.calendar_sha256
            or not isinstance(self.price_basis, str)
            or not self.price_basis
        ):
            raise ValueError("Actual calendar and explicit source price basis required")
        if self.source_type == "original_archive":
            if (
                self.archive_sha256 is None
                or self.raw_bundle_sha256 is not None
                or self.price_basis_policy_sha256 is not None
            ):
                raise ValueError("Original archive identity cannot impersonate provider release")
        elif self.source_type == "yahoo_etf_release":
            if (
                self.archive_sha256 is not None
                or self.manifest_sha256 is None
                or self.raw_bundle_sha256 is None
                or self.price_basis_policy_sha256 is None
                or self.price_basis != "provider_adjusted_close_ratio_price_index_research_proxy"
            ):
                raise ValueError(
                    "Yahoo identity requires its own manifest/bundle/policy and no archive SHA"
                )
        else:
            raise ValueError("Unknown authenticated source type")

    @property
    def source_id(self):
        return self.manifest_sha256 or self.archive_sha256

    def as_dict(self):
        return asdict(self)

    def as_provenance(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def resolve_source_identity(calendar_sha256, *, source_identity=None, archive_sha256=None):
    if source_identity is None:
        source_identity = SourceIdentity(
            "original_archive",
            calendar_sha256,
            "source_OHLC_adjustment_unverified",
            archive_sha256=archive_sha256,
        )
    if (
        not isinstance(source_identity, SourceIdentity)
        or source_identity.calendar_sha256 != calendar_sha256
    ):
        raise ValueError("Frozen source identity and exact calendar required")
    if archive_sha256 is not None and (
        source_identity.source_type != "original_archive"
        or source_identity.archive_sha256 != archive_sha256
    ):
        raise ValueError("Contradictory archive/provider source identities")
    return source_identity


def identity_from_source(source, schedule):
    """Explicit authenticated types; no provider release pretends to be an archive."""
    from .snapshot import MarketSnapshot
    from .yahoo import YahooETFRelease

    calendar_sha = schedule.provenance["schedule_sha256"]
    if type(source) is MarketSnapshot:
        source._completion_guard(source.manifest_sha256)
        return SourceIdentity(
            "original_archive",
            calendar_sha,
            "source_OHLC_adjustment_unverified",
            manifest_sha256=source.manifest_sha256,
            archive_sha256=source.archive_sha256,
        )
    if type(source) is YahooETFRelease:
        source._authenticate()
        if source.schedule.provenance != schedule.provenance:
            raise ValueError("Yahoo release requires its exact authenticated calendar")
        return SourceIdentity(
            "yahoo_etf_release",
            calendar_sha,
            source.provenance["price_basis"],
            manifest_sha256=source.manifest_sha256,
            raw_bundle_sha256=source.source_bundle_sha256,
            price_basis_policy_sha256=source.manifest["policy_sha256"],
        )
    raise ValueError("Explicit authenticated MarketSnapshot or YahooETFRelease required")
