"""Export authenticated full-grid cloud summaries and execute their analysis notebook."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import nbformat as nbf
from jupyter_client import KernelManager
from nbclient import NotebookClient


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(study, report, notebook):
    root = Path(__file__).resolve().parents[1]
    expected = "ea7a801782345b38d2f3e0a1be038c5212b00263360eb78c1b910bfeba65fc8a"
    assert digest(study / "manifest.json") == expected
    assert (study / "COMPLETE").read_text().strip() == expected
    manifest = json.loads((study / "manifest.json").read_bytes())
    assert manifest["complete"] and manifest["actual_cells"] == 9120
    assert manifest["holdout_outcomes_constructed"] is False
    entries = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    names = ("forecast-comparison.json", "config.json", "input-support.json")
    assert all(digest(study / name) == entries[name] for name in names)
    comparison = json.loads((study / names[0]).read_bytes())
    assert len(comparison["panels"]) == 12 and comparison["global_winner"] is None
    assert comparison["economic_evaluation"] == "not_run"
    for panel in comparison["panels"]:
        assert panel["common_date_count"] == len(set(panel["common_dates"]))
        assert len(panel["mean_qlike_common"]) == 8
    if report.exists():
        for name in (*names, "manifest.json"):
            assert digest(report / name) == digest(study / name)
    report.mkdir(parents=True, exist_ok=True)
    for name in (*names, "manifest.json"):
        shutil.copyfile(study / name, report / name)
    exported = {
        "complete": True,
        "scope": "all12_panel_lookback_summaries_all8_recipes_from_completed9120_cell_cloud_run",
        "source_manifest_sha256": expected,
        "kernel_id": 139237851,
        "kernel_version": 1,
        "holdout_evaluated": False,
        "economic_evaluation": False,
        "per_date_9120_forecast_scores_independently_replayed": False,
        "outputs_sha256": {name: digest(report / name) for name in (*names, "manifest.json")},
    }
    (report / "export-manifest.json").write_text(json.dumps(exported, indent=2) + "\n")
    (report / "COMPLETE").write_text(digest(report / "export-manifest.json") + "\n")
    relative = report.resolve().relative_to(root).as_posix()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            "# Covariance forecast comparison\n\nEight covariance recipes a"
            "cross all twelve panel/lookback combinations. "
            "The cloud study contains 9,120 scheduled forecast cells. Thi"
            "s notebook examines the authenticated "
            "full-grid summary; it does not refit models or independently"
            " replay every forecast score. "
            "QLIKE measures variance forecast loss: lower is better withi"
            "n a panel. It is not a portfolio return."
        ),
        code(
            "from pathlib import Path\nimport hashlib, json\nimport numpy a"
            "s np\nimport pandas as pd\n"
            "import matplotlib.pyplot as plt\nfrom IPython.display import display\n"
            "root = next(p for p in (Path.cwd(), *Path.cwd().parents) if "
            "(p/'pyproject.toml').exists())\n"
            f"report = root / {relative!r}\n"
            "sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()\n"
            "export = json.loads((report/'export-manifest.json').read_text())\n"
            "assert (report/'COMPLETE').read_text().strip() == sha(report/'export-manifest.json')\n"
            "for name, expected in export['outputs_sha256'].items():\n    "
            "assert sha(report/name) == expected\n"
            "manifest = json.loads((report/'manifest.json').read_text())\n"
            "comparison = json.loads((report/'forecast-comparison.json').read_text())\n"
            "assert manifest['complete'] and manifest['actual_cells'] == 9120\n"
            "assert not manifest['holdout_outcomes_constructed']\n"
            "display(pd.Series({'scheduled_cells':9120, 'panel_lookback_c"
            "ombinations':12, 'recipes':8, "
            "'globally_paired_months':len(comparison['fully_paired_96_cel"
            "l_dates'])}).to_frame('count'))"
        ),
        md(
            "## Support and maturity\n\nEach comparison uses the same matur"
            "e dates for its eight recipes. "
            "The global intersection is smaller because long stock windows require more history. "
            "A missing or immature cell does not receive a fabricated score."
        ),
        code(
            "rows, support = [], []\nfor panel in comparison['panels']:\n"
            "    key = f\"{panel['panel_id']}_T{panel['observations']}\"\n"
            "    assert panel['common_date_count'] == len(set(panel['common_dates']))\n"
            "    for recipe, loss in panel['mean_qlike_common'].items():\n"
            "        rows.append({'panel':panel['panel_id'], 'lookback':panel['observations'], "
            "'combination':key, 'recipe':recipe, 'mean_qlike':loss, 'pair"
            "ed_months':panel['common_date_count']})\n"
            "    support.append({'combination':key, 'scheduled':panel['expected_dates'], "
            "'paired':panel['common_date_count']})\n"
            "scores = pd.DataFrame(rows)\nsupport = pd.DataFrame(support)."
            "set_index('combination')\n"
            "assert len(scores) == 96 and scores.mean_qlike.map(np.isfinite).all()\n"
            "display(support)\nsupport.plot.bar(figsize=(12,4), ylabel='Monthly forecast dates', "
            "title='Full development support by panel and lookback', rot="
            "35)\nplt.tight_layout()\nplt.show()"
        ),
        md(
            "## Relative forecast loss\n\nSubtract the sample covariance lo"
            "ss within each panel/lookback. "
            "Negative differences favor the alternative. This keeps compa"
            "risons within the same assets, "
            "window and paired dates; absolute QLIKE levels should not rank different universes."
        ),
        code(
            "wide = scores.pivot(index='combination', columns='recipe', v"
            "alues='mean_qlike').sort_index()\n"
            "relative = wide.sub(wide['sample_ddof1'], axis=0)\ndisplay(relative.round(5))\n"
            "limit = abs(relative.to_numpy()).max()\nfig, ax = plt.subplots(figsize=(12,7))\n"
            "image = ax.imshow(relative, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')\n"
            "ax.set_xticks(range(len(relative.columns)), relative.columns"
            ", rotation=40, ha='right')\n"
            "ax.set_yticks(range(len(relative)), relative.index)\n"
            "ax.set_title('Mean QLIKE minus sample covariance; lower is better')\n"
            "fig.colorbar(image, ax=ax, label='Paired mean QLIKE difference')\n"
            "plt.tight_layout()\nplt.show()"
        ),
        md(
            "## Lookback sensitivity\n\nPlot every recipe rather than prese"
            "nting one winning number. "
            "These development comparisons inform the separate economic s"
            "tudy. They do not establish "
            "out-of-sample superiority, statistical significance or trading performance."
        ),
        code(
            "panels = sorted(scores.panel.unique())\nfig, axes = plt.subplots(1,len(panels), "
            "figsize=(16,4), sharex=True)\nfor panel, ax in zip(panels, np.atleast_1d(axes)):\n"
            "    subset = scores[scores.panel.eq(panel)]\n"
            "    baseline = subset[subset.recipe.eq('sample_ddof1')].set_"
            "index('lookback').mean_qlike\n"
            "    for recipe, group in subset.groupby('recipe'):\n"
            "        group = group.sort_values('lookback')\n"
            "        ax.plot(group.lookback, group.mean_qlike - group.loo"
            "kback.map(baseline), marker='o', label=recipe)\n"
            "    ax.axhline(0, color='black', lw=.7)\n    ax.set_title(panel)\n"
            "    ax.set_xlabel('Observed training sessions')\n"
            "axes[0].set_ylabel('Mean QLIKE minus sample covariance')\n"
            "axes[-1].legend(bbox_to_anchor=(1.02,1), loc='upper left')\np"
            "lt.tight_layout()\nplt.show()\n"
            "display(scores.loc[scores.groupby('combination').mean_qlike.idxmin(), "
            "['combination','recipe','mean_qlike','paired_months']].set_index('combination'))"
        ),
    ]
    calibration = root / "reports/risk_forecast_calibration_v1"
    if calibration.exists():
        cells.extend(
            [
                md(
                    "## Full-grid calibration and uncertainty\n\n"
                    "All 9,120 scheduled cells are retained. The 8,808 valid scalar losses "
                    "were replayed from authenticated cloud exports. Calibration bins describe "
                    "development forecasts; they do not fit a calibration model. Paired intervals "
                    "use 2,000 circular scheduled-month block resamples, preserving missing months "
                    "and common recipe dates. These unadjusted intervals do not select "
                    "a trading winner."
                ),
                code(
                    "from IPython.display import Image\n"
                    "calibration = root/'reports/risk_forecast_calibration_v1'\n"
                    "audit = json.loads((calibration/'export-manifest.json').read_text())\n"
                    "assert (calibration/'COMPLETE').read_text().strip() == "
                    "sha(calibration/'export-manifest.json')\n"
                    "for name, expected in audit['outputs_sha256'].items():\n"
                    "    assert sha(calibration/name) == expected\n"
                    "full = pd.read_csv(calibration/'all_forecast_scores.csv', "
                    "float_precision='round_trip')\n"
                    "assert len(full) == 9120 and full.score_status.eq('valid').sum() == 8808\n"
                    "display(full.groupby(['panel','lookback','score_status']).size().unstack(fill_value=0))\n"
                    "display(Image(filename=str(calibration/'variance_calibration.png')))"
                ),
                code(
                    "intervals = pd.read_csv(calibration/'paired_loss_intervals.csv')\n"
                    "assert len(intervals) == 288\n"
                    "display(intervals[intervals.recipe.eq('ewma_halflife63')].round(6))\n"
                    "selected = intervals[intervals.block_months.eq(6) & "
                    "~intervals.recipe.eq('sample_ddof1')].copy()\n"
                    "selected['label'] = selected.panel + '_T' + selected.lookback.astype(str)\n"
                    "fig, axes = plt.subplots(4,3,figsize=(16,14))\n"
                    "for ax, (label, group) in zip(axes.flat, "
                    "selected.groupby('label'), strict=True):\n"
                    "    y = np.arange(len(group))\n"
                    "    ax.hlines(y, group.lower, group.upper)\n"
                    "    ax.scatter(group.mean_difference,y,s=12)\n"
                    "    ax.axvline(0,color='black',lw=.7)\n"
                    "    ax.set_yticks(y,group.recipe,fontsize=6)\n"
                    "    ax.set_title(label)\n"
                    "    ax.set_xlabel('QLIKE difference vs sample; 95% interval')\n"
                    "plt.tight_layout()\nplt.show()"
                ),
            ]
        )
    if calibration.exists() and (calibration / "annual_paired_losses.csv").exists():
        cells.extend(
            [
                md(
                    "## Calendar-year stability\n\n"
                    "Every year uses the common mature dates for all eight recipes. "
                    "These descriptive development slices show whether an aggregate loss "
                    "difference persists over time; no year is removed or used to tune a model."
                ),
                code(
                    "annual = pd.read_csv(calibration/'annual_paired_losses.csv')\n"
                    "assert len(annual) == 768\n"
                    "annual['panel_window'] = annual.panel + '_T' + annual.lookback.astype(str)\n"
                    "fig, axes = plt.subplots(4,2,figsize=(16,18))\n"
                    "bound = annual.mean_difference.abs().max()\n"
                    "for ax, (recipe, group) in zip(axes.flat, "
                    "annual.groupby('recipe'), strict=True):\n"
                    "    table = group.pivot(index='panel_window', columns='year', "
                    "values='mean_difference')\n"
                    "    image = ax.imshow(table,aspect='auto',cmap='RdBu_r', "
                    "vmin=-bound,vmax=bound)\n"
                    "    ax.set_xticks(range(len(table.columns)),table.columns,rotation=45)\n"
                    "    ax.set_yticks(range(len(table)),table.index,fontsize=7)\n"
                    "    ax.set_title(recipe)\n"
                    "    fig.colorbar(image,ax=ax,label='QLIKE minus sample')\n"
                    "plt.tight_layout()\nplt.show()\n"
                    "display(annual.groupby(['panel_window','year']).paired_months.first()"
                    ".unstack())"
                ),
            ]
        )
    document = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.15"},
        },
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    NotebookClient(
        document, km=manager, timeout=180, resources={"metadata": {"path": str(root)}}
    ).execute()
    nbf.write(document, notebook)
    print(
        json.dumps(
            {
                "notebook": str(notebook),
                "sha256": digest(notebook),
                "executed_cells": sum(c.cell_type == "code" for c in cells),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    build(args.study, args.report, args.notebook)
