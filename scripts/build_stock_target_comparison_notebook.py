"""Execute the full paired stock target comparison."""

import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

CELLS = [
    (
        "markdown",
        (
            "# Stock prediction: raw-return and rank-target training"
            "\n\nBoth completed experiments use the same sixteen candi"
            "dates and full monthly cohorts. Only the learned traini"
            "ng target changes. Evaluation uses all 94 common mature"
            " development months. The two final immature months rema"
            "in in the tables and are excluded by the original matur"
            "ity rule. No model is fitted in this notebook."
        ),
    ),
    (
        "code",
        (
            "from pathlib import Path\nimport hashlib,json\nimport pan"
            "das as pd\nimport numpy as np\nimport matplotlib.pyplot a"
            "s plt\nfrom IPython.display import display\npins = {'raw_"
            "return-metrics.json': 'a4ac02351747c5085b73a92d9e313b14"
            "b389b9b3c0b20ed00046bded7881af43', 'raw_return-developm"
            "ent-selection.json': 'c7019b966fc707aadc68fae896dc67821"
            "b6e4673af2174fb3f5231e276e7a377', 'rank_target-metrics."
            "json': '1cb8d23b1157879693bc01eaf0b050d998dbf9810de78d1"
            "e7e1f9994244a0dde', 'rank_target-development-selection."
            "json': 'e6db2d817f03f9f330533a620ff39d7a38dd4743a499516"
            "200a014c71d19ed5c'}\nroot=next(p for p in (Path.cwd(),*P"
            "ath.cwd().parents) if (p/'pyproject.toml').exists())\nre"
            "port=root/'reports/stock_target_comparison_v1'\nmanifest"
            "=json.loads((report/'export-manifest.json').read_text()"
            ")\nassert (report/'COMPLETE').read_text().strip()==hashl"
            "ib.sha256((report/'export-manifest.json').read_bytes())"
            ".hexdigest()\nassert manifest['complete'] and not manife"
            "st['holdout_evaluated']\nfor name,digest in manifest['ou"
            "tputs_sha256'].items():\n    assert hashlib.sha256((repo"
            "rt/name).read_bytes()).hexdigest()==digest\nfor name,pin"
            " in pins.items():\n    assert hashlib.sha256((report/nam"
            "e).read_bytes()).hexdigest() == pin\nframes, selections "
            "= {}, {}\nfor label in ['raw_return','rank_target']:\n   "
            " frames[label] = pd.DataFrame(json.loads((report/(label"
            "+'-metrics.json')).read_text()))\n    selections[label] "
            "= json.loads((report/(label+'-development-selection.jso"
            "n')).read_text())\n    assert len(frames[label]) == 1536"
            "\ndates = selections['raw_return']['common_dates']\nasser"
            "t dates == selections['rank_target']['common_dates'] an"
            "d len(dates)==94\nwide = {label: frame[frame.signal_sess"
            "ion.isin(dates)].pivot(index='signal_session',columns='"
            "candidate_id',values='ic').sort_index() for label,frame"
            " in frames.items()}\nassert wide['raw_return'].index.equ"
            "als(wide['rank_target'].index)\nassert wide['raw_return'"
            "].columns.equals(wide['rank_target'].columns)\nassert al"
            "l(t.notna().all().all() for t in wide.values())\nsummary"
            " = pd.DataFrame({label:t.mean() for label,t in wide.ite"
            "ms()})\nsummary['rank_minus_raw'] = summary.rank_target-"
            "summary.raw_return\ncontrols = summary.index.str.startsw"
            "ith('control_')\nnp.testing.assert_array_equal(wide['raw"
            "_return'].loc[:,controls], wide['rank_target'].loc[:,co"
            "ntrols])\ndisplay(summary.sort_values('raw_return',ascen"
            "ding=False))"
        ),
    ),
    (
        "code",
        (
            "summary[['raw_return','rank_target']].sort_values('raw_"
            "return').plot.barh(figsize=(13,9))\nplt.axvline(0,color="
            "'black',lw=.7)\nplt.xlabel('Mean monthly Spearman IC; sa"
            "me 94 months')\nplt.tight_layout()\nplt.show()\ndifference"
            "s = wide['rank_target']-wide['raw_return']\nrng = np.ran"
            "dom.default_rng(20261017)\nrecords=[]\nfor block in [3,6,"
            "12]:\n    starts=rng.integers(0,94,size=(2000,int(np.cei"
            "l(94/block))))\n    index=((starts[:,:,None]+np.arange(b"
            "lock))%94).reshape(2000,-1)[:,:94]\n    sampled=differen"
            "ces.to_numpy()[index].mean(axis=1)\n    low,high=np.quan"
            "tile(sampled,[.025,.975],axis=0)\n    for j,name in enum"
            "erate(differences.columns):\n        records.append(dict"
            "(candidate=name,block_months=block,mean_difference=diff"
            "erences[name].mean(),lower=low[j],upper=high[j]))\ninter"
            "vals=pd.DataFrame(records)\ndisplay(intervals.round(6))\n"
            "pd.testing.assert_frame_equal(summary,pd.read_csv(repor"
            "t/'paired-target-summary.csv',index_col=0),check_exact="
            "False,atol=1e-14,rtol=1e-12)\npd.testing.assert_frame_eq"
            "ual(intervals,pd.read_csv(report/'paired-target-interva"
            "ls.csv'),check_exact=False,atol=1e-14,rtol=1e-12)"
        ),
    ),
    (
        "markdown",
        (
            "Intervals are descriptive development comparisons with "
            "unadjusted multiplicity. Observed months are resampled "
            "together for all candidates; no artificial market recor"
            "ds are created. The fixed controls are identical across"
            " experiments. The rank-target experiment does not beat "
            "the best raw-return learned candidate or the fixed comp"
            "osite on mean IC. Portfolio economics and the registere"
            "d feature ablation are separate evaluations; final-peri"
            "od outcomes remain sealed."
        ),
    ),
]


def build():
    root = Path(__file__).resolve().parents[1]
    n = nbformat.v4.new_notebook(
        cells=[getattr(nbformat.v4, "new_" + kind + "_cell")(source) for kind, source in CELLS]
    )
    n.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(n, km=km, timeout=180, resources={"metadata": {"path": str(root)}}).execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    output = root / "notebooks/08_stock_target_comparison.ipynb"
    nbformat.write(n, output)
    print(output)


if __name__ == "__main__":
    build()
