"""Execute the portable full-development stock comparison notebook."""

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from jupyter_client import KernelManager
from nbclient import NotebookClient


def build(report, output):
    root = Path(__file__).resolve().parents[1]
    relative = Path(report).resolve().relative_to(root).as_posix()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            "# Stock return prediction: development analysis\n\n"
            "Compare four fixed controls, four Ridge models and eight LightGBM models. "
            "The completed run uses expanding monthly fits on mature observed returns. "
            "This notebook analyses all 1,536 candidate-month records; it does not train models. "
            "The training workflow and paired target comparison are separate notebooks. "
            "These are development results, not trading returns or final holdout scores."
        ),
        code(
            "from pathlib import Path\nimport hashlib, json\nimport numpy as np\n"
            "import pandas as pd\nimport matplotlib.pyplot as plt\n"
            "from IPython.display import display\n"
            "root = next(p for p in (Path.cwd(), *Path.cwd().parents) "
            "if (p/'pyproject.toml').exists())\n"
            f"report = root / {relative!r}\n"
            "sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "manifest = json.loads((report/'export-manifest.json').read_text())\n"
            "assert (report/'COMPLETE').read_text().strip() == sha(report/'export-manifest.json')\n"
            "assert manifest['complete'] and not manifest['holdout_evaluated']\n"
            "for name, digest in manifest['outputs_sha256'].items():\n"
            "    assert sha(report/name) == digest, name\n"
            "metrics = pd.read_json(report/'metrics.json')\n"
            "selection = json.loads((report/'development-selection.json').read_text())\n"
            "config = json.loads((report/'config.json').read_text())\n"
            "uncertainty = json.loads((report/'uncertainty.json').read_text())\n"
            "assert len(metrics) == 1536 and metrics.candidate_id.nunique() == 16\n"
            "assert not metrics.duplicated(['candidate_id','signal_session']).any()\n"
            "display(metrics.groupby(['status','fit_status']).size().rename('candidate_months').to_frame())"
        ),
        md(
            "## Evaluation support\n\n"
            "All 96 scheduled development months remain in the source table. Comparisons use "
            "the same 94 mature months for every candidate. November and December 2014 outcomes "
            "mature after the selection cutoff and cannot contribute outcome values. "
            "The original archive has unverified price-adjustment and vintage semantics; "
            "the results are conditional on that source."
        ),
        code(
            "dates = selection['common_dates']\n"
            "assert len(dates) == 94 and dates[-1] == '2014-10-31'\n"
            "paired = metrics[metrics.signal_session.isin(dates)].copy()\n"
            "assert len(paired) == 94*16 and paired.status.eq('valid').all()\n"
            "wide = paired.pivot(index='signal_session', columns='candidate_id', "
            "values='ic').sort_index()\n"
            "assert wide.notna().all().all()\n"
            "summary = wide.agg(['mean','median','std']).T\n"
            "summary['positive_month_fraction'] = wide.gt(0).mean()\n"
            "summary['months'] = len(wide)\n"
            "summary = summary.sort_values('mean', ascending=False)\n"
            "expected = {row['candidate_id']: row['mean_monthly_ic'] "
            "for row in selection['candidates']}\n"
            "for name, value in summary['mean'].items():\n"
            "    assert np.isclose(value, expected[name], atol=1e-14)\n"
            "display(summary)\n"
            "summary['mean'].sort_values().plot.barh(figsize=(12,8), color='#386b9b')\n"
            "plt.xlabel('Mean monthly Spearman rank correlation')\n"
            "plt.axvline(0,color='gray',lw=.8)\n"
            "plt.title('All candidates on the same mature development months')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "## Behavior through time\n\n"
            "The best learned candidate below is selected on this development history. Its curve "
            "is exploratory, not a separately held-out result. Cumulative IC is a diagnostic sum "
            "of correlations, not cumulative portfolio return."
        ),
        code(
            "learned = [name for name in wide if not name.startswith('control_')]\n"
            "best_learned = wide[learned].mean().idxmax()\n"
            "shown = ['control_fixed_composite','control_momentum_F01',best_learned]\n"
            "timeline = wide[shown].copy(); timeline.index = pd.to_datetime(timeline.index)\n"
            "fig, axes = plt.subplots(2,1,figsize=(13,9),sharex=True)\n"
            "timeline.plot(ax=axes[0],alpha=.65); axes[0].axhline(0,color='gray',lw=.8)\n"
            "axes[0].set_ylabel('Monthly rank IC')\n"
            "timeline.cumsum().plot(ax=axes[1])\n"
            "axes[1].set_ylabel('Sum of monthly IC (not return)')\n"
            "fig.tight_layout(); plt.show()\n"
            "periods = ((2007,2008),(2009,2010),(2011,2012),(2013,2014))\n"
            "period_rows = []\n"
            "for first,last in periods:\n"
            "    years = pd.to_datetime(wide.index).year\n"
            "    group = wide.loc[(years>=first)&(years<=last)]\n"
            "    for name in wide:\n"
            "        period_rows.append({'period':f'{first}–{last}','candidate':name,\n"
            "                            'months':len(group),'IC':group[name].mean()})\n"
            "period_table = pd.DataFrame(period_rows)\n"
            "display(period_table.pivot(index='candidate',columns='period',values='IC'))\n"
            "print('Development-selected learned candidate:',best_learned)"
        ),
        md(
            "## Paired uncertainty\n\n"
            "Circular blocks resample whole observed months with identical indices across "
            "candidates. They generate no market prices. All 3-, 6- and 12-month block choices "
            "are shown. These exploratory intervals do not adjust for model selection."
        ),
        code(
            "fig, axes = plt.subplots(1,3,figsize=(18,9),sharey=True)\n"
            "order = list(summary.index[::-1])\n"
            "for ax, analysis in zip(axes,uncertainty['intervals'],strict=True):\n"
            "    assert analysis['months']==94 and not analysis['selection_adjusted']\n"
            "    rows = {row['candidate_id']:row for row in analysis['candidates']}\n"
            "    for position,name in enumerate(order):\n"
            "        row=rows[name]; lo,hi=row['paired_difference_interval']\n"
            "        ax.plot([lo,hi],[position,position],color='#386b9b')\n"
            "        ax.scatter(row['mean_IC_difference'],position,color='#bd7628',s=20)\n"
            "    ax.axvline(0,color='gray',lw=.8)\n"
            "    ax.set_title(f\"{analysis['block_months']}-month blocks\")\n"
            "    ax.set_xlabel('Mean IC difference versus fixed composite')\n"
            "axes[0].set_yticks(range(len(order)),order); fig.tight_layout(); plt.show()"
        ),
        md(
            "## Execution coverage\n\n"
            "The separate funded backtest attempted all 16 candidates at four fee levels. "
            "Every path stopped at an unavailable or invalid owned price. These stops were "
            "checked against the original archive. No full-period stock return or Sharpe "
            "is reported; the valid prefixes are not substitutes for the requested period."
        ),
        code(
            "support_root = root/'reports/stock_economic_support_v1'\n"
            "support_path = support_root/'export-manifest.json'\n"
            "support = json.loads(support_path.read_bytes())\n"
            "assert hashlib.sha256(support_path.read_bytes()).hexdigest() == "
            "(support_root/'COMPLETE').read_text().strip()\n"
            "assert support['source_manifest_sha256'] == "
            "'be9bbe820ffa16ec917434f8f6bb6ee18533425ec330b79f05a2e76f3ee607a7'\n"
            "assert support['attempted_paths'] == support['unsupported_paths'] == 64\n"
            "assert support['complete_paths'] == 0 "
            "and not support['aggregate_performance_authorized']\n"
            "coverage_path = support_root/'coverage.csv'\n"
            "assert hashlib.sha256(coverage_path.read_bytes()).hexdigest() == "
            "support['outputs_sha256']['coverage.csv']\n"
            "coverage = pd.read_csv(coverage_path)\n"
            "assert len(coverage) == 64 and coverage.candidate.nunique() == 16\n"
            "assert coverage.status.eq('UNSUPPORTED_VALID_PREFIX').all()\n"
            "assert not coverage.duplicated(['candidate','fee_bps']).any()\n"
            "assert coverage.groupby('candidate').fee_bps.nunique().eq(4).all()\n"
            "display(coverage.groupby('candidate',sort=True).agg(\n"
            "    fee_scenarios=('fee_bps','size'), stopped_session=('stopped_session','first'),\n"
            "    stopped_event=('stopped_event','first'),\n"
            "    valid_sessions=('valid_measured_sessions','first')))\n"
            "print('0 of 64 complete economic paths. Full-period performance unavailable.')"
        ),
        md(
            "## Interpretation\n\n"
            "The fixed composite has the highest mean IC in this run. The learned models do "
            "not improve on it, and the paired uncertainty is wide. The controlled "
            "target comparison uses within-month ranks while "
            "preserving features, candidates and validation clocks. Its full paired results "
            "are in 08_stock_target_comparison.ipynb. The execution study above cannot establish "
            "trading profitability on the supplied stock archive; "
            "rank correlation alone does not establish it."
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        NotebookClient(
            notebook, km=manager, timeout=240, startup_timeout=180,
            resources={"metadata": {"path": str(root)}}
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbf.write(notebook, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/stock_development_v1")
    parser.add_argument("--output", default="notebooks/05_stock_development_analysis.ipynb")
    build(**vars(parser.parse_args()))


