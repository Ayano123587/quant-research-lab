"""Package a completed full stock panel without changing its scientific bytes.

The .bin payload is explicitly a ZIP container, avoiding automatic dataset
extraction. No upload, model fitting or kernel submission occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from quantlab.data.archive import file_sha256
from quantlab.stocks.prepare import _source_code, verify_prepared_panel


def _raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def package_stock_inputs(prepared_root, expected_manifest_sha256, output):
    """Require all 7,195 source histories before creating a transport package."""
    source, output = Path(prepared_root).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Transport output and immutable preparation must be separate")
    manifest = verify_prepared_panel(source, expected_manifest_sha256)
    code = _source_code()
    if manifest["config"]["source_code_sha256"] != code:
        raise ValueError("Current preparation source differs from completed panel")
    names = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    inventory = {}
    for name in names:
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Regular source members inside the prepared panel required")
        inventory[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    output.mkdir(parents=True, exist_ok=False)
    try:
        container = output / "stock-panel-v1.bin"
        with zipfile.ZipFile(
            container, "x", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for name in names:
                archive.write(source / name, arcname=name)
        # Independently stream every completed container member, not just its CRC.
        with zipfile.ZipFile(container) as archive:
            if archive.namelist() != names:
                raise ValueError("Transport inventory changed")
            for name, item in inventory.items():
                digest, size = hashlib.sha256(), 0
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                if digest.hexdigest() != item["sha256"] or size != item["bytes"]:
                    raise ValueError("Transport member differs from prepared source")
        verify_prepared_panel(source, expected_manifest_sha256)
        if _source_code() != code:
            raise ValueError("Preparation source changed during packaging")
        for name, item in inventory.items():
            path = source / name
            if file_sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("Source changed during packaging")
        transport = {
            "schema": "quant_full_stock_panel_transport_v1",
            "container_format": "ZIP_STORED_explicit_bin_extension",
            "container": container.name,
            "container_sha256": file_sha256(container),
            "container_bytes": container.stat().st_size,
            "prepared_manifest_sha256": expected_manifest_sha256,
            "original_archive_sha256": manifest["config"]["archive_sha256"],
            "original_snapshot_id": manifest["config"]["snapshot_id"],
            "preparation_source_sha256": code,
            "research_scope_complete": manifest["research_scope_complete"],
            "counts": manifest["counts"],
            "members": inventory,
            "roundtrip_all_member_sha256_verified": True,
            "model_fitting": False,
        }
        raw = _raw(transport)
        (output / "transport-manifest.json").write_bytes(raw)
        receipt = {
            "status": "packaged_not_uploaded",
            "transport_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "container_sha256": transport["container_sha256"],
            "member_count": len(inventory),
            "prepared_manifest_sha256": expected_manifest_sha256,
        }
        (output / "packaging-receipt.json").write_bytes(_raw(receipt))
        return receipt
    except BaseException as error:
        # Preserve the partial container and diagnostic; never silently replace
        # or consume an incomplete package on retry.
        (output / "FAILURE.json").write_bytes(
            _raw({"complete": False, "type": type(error).__name__, "message": str(error)})
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    package_stock_inputs(**vars(parser.parse_args()))
