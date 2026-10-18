"""Restore the exact frozen source into a fresh directory; standard library only."""
import argparse
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_SHA = "a86e3d0dd2941801ae769ced0cfda6f57407e3ba3cc371eb7123373f26a57d5e"
PROVENANCE_SHA = "53ebcbb1adb47130c7a495ad2286cd6f0617ea6604c760adf362b925cdc48871"
PLAN_SHA = "c4c6f872d8335e22c1b20a46494d878c68b2a157b50f624a8b0724e05e69b8ba"
RELEASE_SHA = "f520966f25f426e6688f895bf51fdac25e887aecf2d26129d1b3243cfd663620"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and path.as_posix() == name,
            "Noncanonical source path")
    require("\\" not in name and ":" not in name, "Nonportable source path")
    for part in path.parts:
        require(part not in {".", "..", ".git"} and part.rstrip(" .") == part,
                "Unsafe source path")
        require(part.split(".")[0].upper() not in {
            "CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
            *[f"LPT{i}" for i in range(1, 10)]}, "Reserved source path")
    return path


def restore(output):
    repository = Path(__file__).resolve().parents[1]
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Output must not exist")
    require(output.parent.is_dir(), "Output parent must already exist")
    require(not any(p.is_symlink() or p.is_junction() for p in output.parents),
            "Output ancestors must not be symlinks")
    require(not output.resolve().is_relative_to(repository), "Output must be outside this repo")
    inputs = repository / "reproducibility"
    require(not any(p.is_symlink() for p in inputs.iterdir()), "Symlinked frozen input")
    raw = (inputs / "frozen-source-f520966f.zip").read_bytes()
    require(len(raw) == 4793361 and hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA,
            "Frozen archive differs before ZIP parsing")
    provenance_raw = (inputs / "frozen-source-f520966f.provenance.json").read_bytes()
    require(hashlib.sha256(provenance_raw).hexdigest() == PROVENANCE_SHA,
            "Frozen provenance differs")
    provenance = json.loads(provenance_raw)
    require(provenance["schema"] == "quant_frozen_source_provenance_v1"
            and provenance["release_sha256"] == RELEASE_SHA
            and provenance["exact_members"] == 332, "Unexpected provenance identity")
    mapping = provenance["reconstruction_mapping"]
    require(len(mapping) == 331, "Exact original reconstruction mapping required")
    targets = [safe(row["path"]).as_posix() for row in mapping.values()]
    require(len(set(name.casefold() for name in targets)) == 331, "Duplicate output path")
    folded = {name.casefold() for name in targets}
    require(not any(parent.as_posix().casefold() in folded
                    for name in targets for parent in PurePosixPath(name).parents
                    if parent.as_posix() != "."), "Conflicting file and directory paths")
    expected = {name: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for name, row in mapping.items()}
    # Provenance explicitly identifies this sole unmapped plan; its original pin is fixed here.
    expected["shared-holdout-plan.json"] = {"bytes": 134368, "sha256": PLAN_SHA}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(len(infos) == 332 and {i.filename for i in infos} == set(expected),
                "Unexpected, missing or duplicate frozen ZIP member")
        for info in infos:
            safe(info.filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(not info.is_dir() and mode in {0, stat.S_IFREG},
                    "Frozen ZIP members must be regular files")
            require(not info.flag_bits & 1, "Encrypted frozen ZIP member")
            pin = expected[info.filename]
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            require(info.file_size == pin["bytes"] and digest.hexdigest() == pin["sha256"],
                    "Frozen member differs: " + info.filename)
        # Every byte was authenticated before the first directory or file write.
        output.mkdir(exist_ok=False)
        for member, row in mapping.items():
            destination = output / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not any(p.is_symlink() or p.is_junction() for p in destination.parents),
                    "Output parent changed to a symlink")
            with archive.open(member) as source, destination.open("xb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
    print(f"Restored 331 authenticated frozen source files to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh directory outside this repository")
    restore(parser.parse_args().output)
