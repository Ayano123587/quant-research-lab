"""Build the prospective PRIVATE main workflow notebook; never execute or publish.

The later reviewed transport supplies an authenticated stdlib bootstrap cell.
It must bind whole containers/members/source/runtime before defining CONTEXT_PATH
and LOCKED_PYTHON. An unbound draft fails before reading any research input.
"""

import ast
import hashlib
import json
from pathlib import Path


def _cell(source, markdown=False):
    result = {
        "cell_type": "markdown" if markdown else "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if not markdown:
        result.update(execution_count=None, outputs=[])
    return result


def _locked_cell(code):
    if "'''" in code:
        raise ValueError("Unescaped workflow source delimiter")
    return _cell("run_locked(r'''\n" + code + "\ncheck_input_sources()\n''')\n")


BRIDGE = """import hashlib, json, os, subprocess, sys
from pathlib import Path
assert Path('/kaggle/working').is_dir(), 'Cloud notebook only'
assert Path(LOCKED_PYTHON).is_file() and Path(CONTEXT_PATH).is_file()
CONTEXT_SHA = hashlib.sha256(Path(CONTEXT_PATH).read_bytes()).hexdigest()
def run_locked(code):
    # Each visible cell below really executes this exact code in the locked
    # interpreter. There are no pasted logs, cached training outputs or host fits.
    env = LOCKED_ENV.copy()
    env['QUANTLAB_WIDER_CONTEXT'] = str(CONTEXT_PATH)
    env['QUANTLAB_WIDER_CONTEXT_SHA'] = CONTEXT_SHA
    try:
        result = subprocess.run([str(LOCKED_PYTHON), '-c', code], env=env, check=True)
    except BaseException as error:
        context = json.loads(Path(CONTEXT_PATH).read_bytes())
        failure = Path(context['artifact_root']).parent / 'NOTEBOOK_FAILURE.json'
        failure.write_text(json.dumps({'complete':False, 'error_type':type(error).__name__})+'\\n')
        raise
    return result.returncode
"""

PRELUDE = """import hashlib, json, os, time
from datetime import date
from pathlib import Path
context_raw = Path(os.environ['QUANTLAB_WIDER_CONTEXT']).read_bytes()
assert hashlib.sha256(context_raw).hexdigest() == os.environ['QUANTLAB_WIDER_CONTEXT_SHA']
context = json.loads(context_raw)
deadline = context['startup_deadline_monotonic']
def check_input_sources():
    paths = {str(Path(context['project'])/p): s for p,s in context['project_source_sha256'].items()}
    paths.update(context['input_container_sha256'])
    for name, expected_sha in paths.items():
        digest = hashlib.sha256()
        with Path(name).open('rb') as stream:
            while block := stream.read(4*1024*1024):
                assert time.monotonic() < deadline
                digest.update(block)
        assert digest.hexdigest() == expected_sha, 'Source/input postflight mismatch'
check_input_sources()
import platform, sys
from importlib.metadata import version
assert platform.python_version() == '3.12.15'
assert Path(os.path.abspath(sys.executable)) == Path(context['locked_interpreter'])
managed_root = Path(context['private_managed_root']).resolve()
assert Path(sys.executable).resolve().is_relative_to(managed_root)
assert {name:version(name) for name in context['runtime']} == context['runtime']
from quantlab.data.calendar import XNYSSchedule
from quantlab.stocks import development as d
from quantlab.stocks.wider_panel import PARENT_SHA
schedule = XNYSSchedule(date(1962,1,1), date(2018,12,31))
assert time.monotonic() < deadline
base_path = Path(context['project']) / 'configs/stock-development.toml'
base = d.load_spec(base_path)
"""


def build_notebook(output, *, project_root, bootstrap_source=None, expected_bootstrap_sha256=None):
    """Return an unexecuted notebook; bootstrap binding is an explicit later gate."""
    output, project_root = Path(output), Path(project_root)
    if output.suffix != ".ipynb" or "notebooks" in output.parts:
        raise ValueError("Unexecuted prospective notebook belongs in PRIVATE staging")
    if bootstrap_source is None:
        bootstrap = (
            "raise RuntimeError('SOURCE ONLY: authenticated transport bootstrap not yet frozen')\n"
        )
    else:
        bootstrap = Path(bootstrap_source).read_text(encoding="utf-8")
        if hashlib.sha256(bootstrap.encode()).hexdigest() != expected_bootstrap_sha256:
            raise ValueError("External bootstrap source SHA mismatch")
        ast.parse(bootstrap)
    module = (project_root / "src/quantlab/stocks/wider_study.py").read_text(encoding="utf-8")
    tree = ast.parse(module)
    fit = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_predict_visible"
    )
    visible_fit = ast.get_source_segment(module, fit)
    cells = [
        _cell(
            "# Stock feature preparation and expanding monthly training\n\n"
            "Prospective development study: four fixed RAW-target arms, all original cohorts, "
            "96 monthly predictions and94 mature paired months. The fresh six-column reference "
            "must match the original saved float64 predictions exactly. This draft has no "
            "executed outputs; only the actual cloud-executed copy may become a public result.\n",
            True,
        ),
        _cell(bootstrap),
        _cell(BRIDGE),
    ]
    cells += [
        _cell(
            "## Authenticate the full source population and original decision cohorts\n"
            "No current-survivor screen, no added-feature readiness filter, no label "
            "value whose maturity is after the 2014 information cutoff.\n",
            True,
        )
    ]
    code = (
        PRELUDE
        + """from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.ablation import authenticate_reference
snapshot = MarketSnapshot(context['market_root'], context['normalized_manifest_sha256'])
parent = d.verify_prepared_panel(context['prepared_root'], PARENT_SHA)
reference, old_metrics, old_fits = authenticate_reference(context['reference_root'], base)
data = d.read_development_inputs(context['prepared_root'], PARENT_SHA, base, schedule)
print('Full upstream population:', parent['counts'])
print('Original training feature rows:', len(data.features),
      'Original decision dates:', len(data.decision_times))
print('Original reference prediction dates:', len(reference))
print('Clock-rejected outcome partitions:', data.outcome_value_partitions_refused)
assert data.full_research_scope and len(reference) == 96
"""
    )
    cells.append(_locked_cell(code))
    cells.append(
        _cell(
            "## Actual past-only feature window and formula examples\n"
            "The window is observed source data. F22-F25 remain proof-gated; observed "
            "volume features have conditional source units and make no capacity claim.\n",
            True,
        )
    )
    code = (
        PRELUDE
        + """from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.wider_features import wider_candidate_from_history
snapshot = MarketSnapshot(context['market_root'], context['normalized_manifest_sha256'])
with snapshot.verified_read_session() as verified:
    bars = verified.bars('stock', 'IBM')
    candidate = wider_candidate_from_history('stock:ibm.us', bars, schedule,
        as_of=date(2010,1,29), archive_sha256=snapshot.archive_sha256)
assert candidate.original.eligible and len(candidate.original.source_rows) == 253
print('Actual window:', candidate.original.source_rows[0].session,
      candidate.original.source_rows[-1].session)
for value in candidate.values:
    print(value.feature_id, value.value, value.missing_reason, value.units)
"""
    )
    cells.append(_locked_cell(code))
    cells.append(
        _cell(
            "## Prepare all original training cohorts with added observed predictors\n"
            "This is the full preparation operation, not a displayed cached panel. Original "
            "parent raw-six bytes, source-row digests and clocks are retained exactly. "
            "Only the exact full-cohort audited F09/F19 whitelist can record "
            "a platform difference.\n",
            True,
        )
    )
    code = (
        PRELUDE
        + """from quantlab.data.snapshot import MarketSnapshot
from quantlab.stocks.wider_panel import prepare_wider_panel
from quantlab.stocks.feature_reproduction import POLICY, POLICY_SHA256, DIAGNOSTIC_SHA256
from quantlab.stocks.wider_study import load_spec as load_wider_spec
wider_spec_path = Path(context['project'])/'configs/stock-wider-development.toml'
portability = load_wider_spec(wider_spec_path)['feature_reproduction']
assert portability['policy_sha256'] == POLICY_SHA256
assert portability['diagnostic_sha256'] == DIAGNOSTIC_SHA256
snapshot = MarketSnapshot(context['market_root'], context['normalized_manifest_sha256'])
prepared = prepare_wider_panel(snapshot, context['prepared_root'], schedule,
    context['wider_root'], service_deadline_monotonic=deadline,
    reproduction_policy=POLICY, reproduction_parent_platform=portability['parent_platform'])
print('Wider preparation completed:', prepared['decision_count'],
      'dates;', prepared['feature_rows'], 'unchanged selected rows')
print('Exact whitelisted F09/F19 representation differences:',
      len(prepared['accepted_feature_representation_differences']))
"""
    )
    cells.append(_locked_cell(code))
    cells.append(
        _cell(
            "## Mature training labels and equal date influence\n"
            "Training expands only through labels available at each signal cutoff. Sample "
            "weights are inverse rows per date, normalized to mean one.\n",
            True,
        )
    )
    code = (
        PRELUDE
        + """import math
from quantlab.stocks.ablation import authenticate_reference
reference, _, _ = authenticate_reference(context['reference_root'], base)
data = d.read_development_inputs(context['prepared_root'], PARENT_SHA, base, schedule)
day = min(reference)
cutoff = dict(data.decision_times)[day]
batch = d.training_batch(data, day, cutoff, base)
assert batch.ready and batch.max_label_available_at <= cutoff
mass = {}
for row, weight in zip(batch.feature_rows, batch.weights, strict=True):
    mass.setdefault(str(row.signal_session), []).append(weight)
masses = [math.fsum(v) for v in mass.values()]
print('First actual fit:', day, cutoff, 'rows:', len(batch.feature_rows),
      'dates:', len(batch.date_counts), 'latest mature label:', batch.max_label_available_at)
print('Raw training SHA:', batch.input_sha256)
print('Weight mean:', math.fsum(batch.weights)/len(batch.weights),
      'date-mass min/max:', min(masses), max(masses))
"""
    )
    cells.append(_locked_cell(code))
    cells.append(
        _cell(
            "## Construct, fit and predict the fixed learner\n"
            "The following actual estimator code runs in the locked interpreter for all384 "
            "fits. The orchestrator validates its exact AST against the reviewed function, "
            "uses unchanged RAW targets and date weights, seals each prediction before joining "
            "current outcomes, and emits every completed date/arm checkpoint.\n",
            True,
        )
    )
    # Triple-quoted multiline source stays readable in notebook cells and is
    # executed, rather than pasted as an illustration beside another old study.
    training = (
        PRELUDE
        + "\n"
        + visible_fit
        + """
from quantlab.data.archive import file_sha256
from quantlab.stocks.wider_study import run_wider_study, validate_visible_fit
print('Visible estimator source certificate:', validate_visible_fit(fit_predict_visible))
print('Fixed recipe: L1,15 leaves,100 trees; four feature arms; RAW target;4 threads')
result = run_wider_study(context['prepared_root'], context['wider_root'],
    file_sha256(Path(context['wider_root'])/'manifest.json'), base_path,
    Path(context['project'])/'configs/stock-wider-development.toml', schedule,
    context['reference_root'], context['study_root'],
    service_deadline_monotonic=deadline, visible_fit_callback=fit_predict_visible)
print('Actual full study:', result)
check_input_sources()
"""
    )
    # inspect.getsource requires a real source file in the isolated interpreter.
    # The visible cell writes exactly this reviewed text, then executes that file.
    cells.append(
        _cell(
            "training_source = r'''" + training + "'''\n"
            "training_path = Path(CONTEXT_PATH).parent / 'visible-wider-training.py'\n"
            "with training_path.open('x', encoding='utf-8', newline='\\n') as stream:\n"
            "    stream.write(training_source)\n"
            'run_locked("import runpy; runpy.run_path(" + repr(str(training_path)) + '
            "\", run_name='__main__')\")\n"
        )
    )
    cells.append(
        _cell(
            "## Full paired development results and coverage\n"
            "All four arms and all94 mature months are reported together. Nov/Dec2014 "
            "unresolved labels remain explicit; no holdout, partial-grid winner or reliable "
            "edge claim is implied. Uncertainty is a separate later analysis.\n",
            True,
        )
    )
    code = (
        PRELUDE
        + """from quantlab.data.archive import file_sha256
study = Path(context['study_root'])
manifest_sha = file_sha256(study/'manifest.json')
assert (study/'COMPLETE').read_text().strip() == manifest_sha
manifest = json.loads((study/'manifest.json').read_bytes())
for item in manifest['artifacts']:
    assert file_sha256(study/item['path']) == item['sha256']
comparison = json.loads((study/'paired-comparison.json').read_bytes())
print(json.dumps(comparison, indent=2))
print('Fixed controls:', json.loads((study/'control-metrics.json').read_bytes())[:5])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
metrics = json.loads((study/'metrics.json').read_bytes())
fig, ax = plt.subplots(figsize=(11,5))
for arm in comparison['mean_ic']:
    rows = [r for r in metrics if r['arm_id']==arm and r['status']=='valid']
    ax.plot([date.fromisoformat(r['signal_session']) for r in rows],
            [r['ic'] for r in rows], label=arm, alpha=.7)
ax.axhline(0, color='black', linewidth=.7)
ax.set(ylabel='Within-month Spearman IC', title='All94 mature months; descriptive development')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(Path(context['artifact_root'])/'wider-monthly-ic.png', dpi=150)
print('Verified study manifest:', manifest_sha)
check_input_sources()
workflow = Path(context['artifact_root']).parent
assert not (workflow/'NOTEBOOK_FAILURE.json').exists()
artifacts = []
for path in sorted(workflow.rglob('*')):
    if path.is_file():
        assert time.monotonic() < deadline
        artifacts.append({'path':path.relative_to(workflow).as_posix(),
                          'sha256':file_sha256(path), 'bytes':path.stat().st_size})
assert sum(r['bytes'] for r in artifacts) <= 20*1024**3
proof = {'complete':True, 'cells':384, 'paired_mature_months':94,
         'holdout_evaluated':False, 'study_manifest_sha256':manifest_sha,
         'source_plan_sha256':context['source_plan_sha256'], 'artifacts':artifacts}
proof_path = workflow/'workflow-manifest.json'
with proof_path.open('x') as stream:
    json.dump(proof, stream, sort_keys=True, indent=2)
assert time.monotonic() < deadline
(workflow/'NOTEBOOK_COMPLETE').write_text(file_sha256(proof_path)+'\\n')
"""
    )
    cells.append(
        _cell(
            "run_locked(r'''\n" + code + "''')\n"
            "from IPython.display import display, Image\n"
            "context = json.loads(Path(CONTEXT_PATH).read_bytes())\n"
            "display(Image(filename=str(Path(context['artifact_root'])/'wider-monthly-ic.png')))\n"
        )
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "quantlab": {
                "source_only": True,
                "executed": False,
                "bootstrap_bound": bootstrap_source is not None,
            },
        },
        "cells": cells,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"wider-{index:02d}"
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(notebook, stream, indent=1)
        stream.write("\n")
    return {
        "path": str(output),
        "source_only": True,
        "executed": False,
        "cells": len(cells),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
