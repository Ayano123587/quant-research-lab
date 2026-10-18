"""Cloud-only byte export of the attempted risk family; no economic rerun."""
import hashlib
import json
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

PARENT = '077d7aa85065d31172194468f0781fe11de71e2e1d084504b612b9c3e0dfae04'
PARENT_KERNEL_ID = 139268917


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(root, output):
    started = time.monotonic()
    if sys.platform != 'linux' or Path(root).resolve() != Path('/kaggle/input'):
        raise ValueError('Actual cloud mount required; no local strategy run')
    output = Path(output)
    output.mkdir(exist_ok=False)
    (output / 'status.json').write_text('{"phase":"authenticating_original_attempted_family"}\n')

    def budget():
        if time.monotonic() - started >= 1800:
            raise TimeoutError('Whole export service budget exhausted')

    matches = []
    for path in Path(root).rglob('manifest.json'):
        budget()
        if path.is_file() and not path.is_symlink() and sha(path) == PARENT:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError('Exactly one frozen risk family manifest required')
    parent = matches[0].parent
    manifest = json.loads(matches[0].read_bytes())
    if not (manifest['complete'] is False and manifest['attempted_terminal_paths'] == 208
            and manifest['expected_paths'] == 208 and manifest['completed_paths'] == 104
            and len(manifest['unsupported_paths']) == 104
            and all(p.startswith('stock100/') for p in manifest['unsupported_paths'])
            and manifest['holdout_evaluated'] is False
            and manifest['forecast_outcomes_parsed'] is False
            and manifest['selection'] == 'none_all_prespecified_paths_and_fees_reported'
            and (parent / 'ATTEMPTED_UNSUPPORTED').read_text().strip() == PARENT
            and not (parent / 'COMPLETE').exists()):
        raise ValueError('Exact attempted208/completeETF104/unsupportedstock104 contract required')
    entries = manifest['artifacts']
    names = [entry['path'] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate artifact paths')
    for name in names:
        rel = PurePosixPath(name)
        if (rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name
                or not (rel.suffix in ('.json', '.parquet')
                        or rel.name in ('COMPLETE', 'UNSUPPORTED'))):
            raise ValueError('Only original derived metadata and economic ledgers allowed')
    if sum(name.endswith('/ledger.parquet') for name in names) != 104:
        raise ValueError('All104 original complete path ledgers required')
    if sum(name.endswith('/FAILURE.json') for name in names) != 104:
        raise ValueError('All104 unsupported diagnostics required')

    def guard():
        budget()
        if (sha(parent / 'manifest.json') != PARENT
                or (parent / 'ATTEMPTED_UNSUPPORTED').read_text().strip() != PARENT
                or (parent / 'COMPLETE').exists()):
            raise ValueError('Original family marker changed')
        for entry in entries:
            budget()
            path = parent / entry['path']
            if (not path.resolve().is_relative_to(parent.resolve())
                    or any(stat.S_ISLNK(p.lstat().st_mode) for p in (path, *path.parents))
                    or not path.is_file() or sha(path) != entry['sha256']):
                raise ValueError('Original manifest-bound artifact changed')
        for item in manifest['paths']:
            path = parent / 'paths' / item['path_id'] / 'manifest.json'
            if sha(path) != item['manifest_sha256']:
                raise ValueError('Path manifest changed')
            record = json.loads(path.read_bytes())
            is_etf = item['path_id'].startswith('etf8/')
            if record['complete'] is not is_etf:
                raise ValueError('Do not promote unsupported stock paths to complete')

    guard()
    archive = output / 'risk-economic-audit.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in sorted([*names, 'manifest.json', 'ATTEMPTED_UNSUPPORTED']):
            budget()
            bundle.write(parent / name, arcname=name)
    guard()
    result = {'export_complete': True, 'scientific_family_complete': False,
              'parent_manifest_sha256': PARENT, 'parent_kernel_id': PARENT_KERNEL_ID,
              'parent_kernel_version': 1, 'attempted_paths': 208,
              'completed_etf_paths': 104, 'unsupported_stock_paths': 104,
              'selected_artifacts': len(entries), 'all_original_study_artifacts_included': True,
              'model_files_included': 0, 'economic_paths_recomputed': False,
              'models_refitted': False, 'holdout_evaluated': False,
              'source_sha256': sha(Path(__file__)),
              'files': {archive.name: {'sha256': sha(archive), 'bytes': archive.stat().st_size}}}
    guard()
    target = output / 'export-manifest.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    budget()
    (output / 'EXPORT_COMPLETE').write_text(sha(target) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    destination = Path('/kaggle/working/risk-economic-audit-export')
    try:
        export(Path('/kaggle/input'), destination)
    except BaseException as error:
        destination.mkdir(exist_ok=True)
        (destination / 'FAILURE.json').write_text(json.dumps({'error_type': type(error).__name__,
            'message': str(error), 'scientific_family_complete': False}) + '\n')
        raise
