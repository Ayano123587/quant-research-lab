"""Execute the full observed stock feature-ablation analysis."""

import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

CELLS = [
    (
        "markdown",
        (
            "# Stock feature ablation\n\nFive feature sets, 96 expandi"
            "ng-fit dates and 94 mature evaluation months. All obser"
            "vations come from the supplied market archive. These ar"
            "e development rank correlations, not investment returns"
            "."
        ),
    ),
    (
        "code",
        (
            "from pathlib import Path\nimport json, hashlib\nimport nu"
            "mpy as np\nimport pandas as pd\nimport matplotlib.pyplot "
            "as plt\nfrom IPython.display import display\nroot=next(p "
            "for p in (Path.cwd(),*Path.cwd().parents) if (p/'pyproj"
            "ect.toml').exists())\nstudy=root/'reports/stock_feature_"
            "ablation_v1'\nsha=lambda p:hashlib.sha256(p.read_bytes()"
            ").hexdigest()\nmanifest=json.loads((study/'export-manife"
            "st.json').read_text())\nassert (study/'COMPLETE').read_t"
            "ext().strip()==sha(study/'export-manifest.json')\nassert"
            " manifest['complete'] and not manifest['holdout_evaluat"
            "ed']\nfor name,digest in manifest['outputs_sha256'].item"
            "s():\n    assert sha(study/name)==digest\nmetrics=pd.Data"
            "Frame(json.loads((study/'metrics.json').read_text()))\np"
            "aired=json.loads((study/'paired-comparison.json').read_"
            "text())\naudit=json.loads((study/'audit-summary.json').r"
            "ead_text())\nassert audit['scientific_assertions_PASS'] "
            "and audit['independent_metric_cells']==480\nassert len(m"
            "etrics)==480 and not metrics.duplicated(['signal_sessio"
            "n','arm_id']).any()\ndisplay(metrics.groupby(['arm_id','"
            "status']).size().unstack(fill_value=0))"
        ),
    ),
    (
        "markdown",
        (
            "## Common evaluation months\n\nUse the same mature months"
            " for every arm. The final two development months remain"
            " recorded but do not contribute unavailable outcomes."
        ),
    ),
    (
        "code",
        (
            'dates=paired["common_mature_dates"]\nassert len(dates)=='
            "94\nwide=metrics[metrics.signal_session.isin(dates)].piv"
            'ot(index="signal_session",columns="arm_id",values="ic")'
            ".sort_index()\nassert wide.shape==(94,5) and wide.notna("
            ').all().all()\nsummary=wide.agg(["mean","median","std"])'
            '.T\nsummary["positive_month_fraction"]=wide.gt(0).mean()'
            '\nsummary["difference_from_all_features"]=summary["mean"'
            ']-summary.loc["all_features","mean"]\ndisplay(summary.so'
            'rt_values("mean",ascending=False))\nfig,ax=plt.subplots('
            'figsize=(9,4))\nsummary["mean"].sort_values().plot.barh('
            'ax=ax,color="#2878a5")\nax.axvline(0,color="black",lw=.7'
            ');ax.set(xlabel="Mean monthly Spearman IC",title="Full '
            'mature development comparison")\nfig.tight_layout();plt.'
            "show()"
        ),
    ),
    (
        "code",
        (
            "fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)\nc"
            'olors={"activity":"#2878a5","all_features":"#222222","m'
            'omentum_trend":"#e1812c","reversal":"#3a923a","risk":"#'
            'b63738"}\nwide.rolling(12,min_periods=12).mean().plot(ax'
            '=axes[0],color=colors)\naxes[0].axhline(0,color="black",'
            'lw=.7);axes[0].set(ylabel="12-month mean IC",title="Tem'
            'poral stability")\ndeltas=wide.drop(columns="all_feature'
            's").subtract(wide.all_features,axis=0)\ndeltas.cumsum().'
            "plot(ax=axes[1],legend=False,color=colors)\naxes[1].axhl"
            'ine(0,color="black",lw=.7);axes[1].set(ylabel="Cumulati'
            've paired IC difference",xlabel="Signal month")\nfig.tig'
            "ht_layout();plt.show()"
        ),
    ),
    (
        "markdown",
        (
            "## Interpretation\n\nThe full feature set retains the hig"
            "hest average IC. Removing any tested feature family red"
            "uces it. This ablation does not establish a trading edg"
            "e or independent test performance. The audit checked al"
            "l 480 prediction/fit cells and reconstructed training i"
            "nputs. Its original process reached the final receipt w"
            "ith a metadata-key error; the receipt repair is recorde"
            "d explicitly, with the original traceback retained."
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
    output = root / "notebooks/07_stock_feature_ablation.ipynb"
    nbformat.write(n, output)
    print(output)


if __name__ == "__main__":
    build()
