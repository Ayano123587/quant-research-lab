"""Execute the full saved-path risk-economic comparison."""

import sys
from pathlib import Path

import nbformat as nb
from jupyter_client import KernelManager
from nbclient import NotebookClient

CELLS = [
    (
        "markdown",
        (
            "# Covariance forecasts in portfolio decisions\n\nCompare "
            "26 allocation arms at 0, 5, 10 and 25 basis points per "
            "traded side. Each ETF path covers the same 1,975 develo"
            "pment sessions. This notebook uses saved, audited ledge"
            "rs; it does not fit a model or rerun a strategy. All 10"
            "4 stock paths remain unsupported and have no full-perio"
            "d performance score."
        ),
    ),
    (
        "code",
        (
            "from pathlib import Path\nimport hashlib, json, math\nfro"
            "m statistics import mean, stdev\nimport numpy as np\nimpo"
            "rt pandas as pd\nimport matplotlib.pyplot as plt\nfrom IP"
            "ython.display import display\nroot=next(p for p in (Path"
            ".cwd(),*Path.cwd().parents) if (p/'pyproject.toml').exi"
            "sts())\nreport=root/'reports/risk_economic_development_v"
            "1'\nsha=lambda p:hashlib.sha256(p.read_bytes()).hexdiges"
            "t()\nmanifest=json.loads((report/'export-manifest.json')"
            ".read_text())\nassert (report/'COMPLETE').read_text().st"
            "rip()==sha(report/'export-manifest.json')\nassert manife"
            "st['complete'] and not manifest['scientific_family_comp"
            "lete']\nassert not manifest['holdout_evaluated']\nfor nam"
            "e,digest in manifest['outputs_sha256'].items():\n    ass"
            "ert sha(report/name)==digest\nmetrics=pd.read_csv(report"
            "/'path-metrics.csv',float_precision='round_trip')\ndaily"
            "=pd.read_parquet(report/'daily-paths.parquet')\nsupport="
            "pd.DataFrame(json.loads((report/'stock-support.json').r"
            "ead_text()))\nassert len(metrics)==104 and len(daily)==2"
            "05400 and len(support)==104\nassert not daily.duplicated"
            "(['path_id','session']).any()\nassert metrics.groupby('f"
            "ee_bps').size().to_dict()=={0:26,5:26,10:26,25:26}\nasse"
            "rt daily.groupby('path_id').size().eq(1975).all()\ndispl"
            "ay(pd.DataFrame({'panel':['ETF8','stock100'],'complete_"
            "paths':[104,0], 'unsupported_paths':[0,104]}))\ndisplay("
            "support.groupby(['failure_session','reason']).size().re"
            "name('paths').to_frame())"
        ),
    ),
    (
        "markdown",
        (
            "## Recalculate the reported metrics\n\nRecompute returns,"
            " annualized volatility, zero-cash-reference Sharpe, clo"
            "se-to-close drawdown, fees and turnover from every save"
            "d ETF path. The separate full accounting audit also rec"
            "onciles trades, holdings and saved asset-PnL components"
            ". This is conditional on the source vintage and modeled"
            " execution assumptions; it does not independently recon"
            "struct every original price mark."
        ),
    ),
    (
        "code",
        (
            "replayed=[]\nfor path_id,frame in daily.groupby('path_id"
            "',sort=True):\n    frame=frame.sort_values('session')\n  "
            "  nav=frame.close_nav.to_numpy()\n    assert np.isfinite"
            "(nav).all() and (nav>0).all()\n    returns=[float(curren"
            "t/previous-1) for current,previous in zip(nav,np.r_[100"
            "000.,nav[:-1]])]\n    volatility=stdev(returns)*math.sqr"
            "t(252)\n    peak=np.maximum.accumulate(np.r_[100000.,nav"
            "])[1:]\n    calculated={'net_return':float(nav[-1]/10000"
            "0-1),\n      'annual_return_252_session_convention':floa"
            "t((nav[-1]/100000)**(252/1975)-1),\n      'annual_volati"
            "lity_sample_ddof1':volatility,\n      'annual_sharpe_zer"
            "o_cash_reference':mean(returns)*252/volatility,\n      '"
            "maximum_close_nav_drawdown':float(np.min(nav/peak-1)),\n"
            "      'total_fee_money':math.fsum(frame.fee),\n      'to"
            "tal_full_buy_plus_sell_turnover':math.fsum(frame.full_t"
            "urnover)}\n    saved=metrics.set_index('path_id').loc[pa"
            "th_id]\n    for name,value in calculated.items():\n      "
            "  assert math.isclose(value,float(saved[name]),rel_tol="
            "1e-12,abs_tol=1e-12),(path_id,name)\n    replayed.append"
            "({'path_id':path_id,**calculated})\nassert len(replayed)"
            "==104\ndisplay(metrics[['fee_bps','arm','annual_return_2"
            "52_session_convention',\n 'annual_volatility_sample_ddof"
            "1','annual_sharpe_zero_cash_reference',\n 'maximum_close"
            "_nav_drawdown','total_full_buy_plus_sell_turnover']].so"
            "rt_values(['arm','fee_bps']))"
        ),
    ),
    (
        "markdown",
        (
            "## Fees and allocation stability\n\nRead each row across "
            "fee scenarios. The covariance estimator and turnover pe"
            "nalty identify an arm; all arms are retained, without s"
            "electing a winner from this display. Sharpe ratios here"
            " are descriptive development estimates, not independent"
            " confirmation of an investment edge."
        ),
    ),
    (
        "code",
        (
            "arms=sorted(metrics.arm.unique())\nwide=metrics.pivot(in"
            "dex='arm',columns='fee_bps',values='annual_sharpe_zero_"
            "cash_reference').reindex(arms)\nfig,ax=plt.subplots(figs"
            "ize=(9,11))\nim=ax.imshow(wide.to_numpy(),aspect='auto',"
            "cmap='viridis')\nax.set_yticks(range(len(arms)),arms)\nax"
            ".set_xticks(range(4),[str(x) for x in wide.columns])\nax"
            ".set(xlabel='Fee per traded side (basis points)',title="
            "'All 104 development Sharpe estimates')\nfor i in range("
            "len(arms)):\n    for j in range(4):ax.text(j,i,f'{wide.i"
            "loc[i,j]:.2f}',ha='center',va='center',fontsize=8,color"
            "='black' if im.norm(wide.iloc[i,j])>.55 else 'white')\nf"
            "ig.colorbar(im,ax=ax,label='Annualized Sharpe; cash ref"
            "erence zero')\nfig.tight_layout();plt.show()"
        ),
    ),
    (
        "code",
        (
            "colors={arm:plt.get_cmap('turbo')(i/(len(arms)-1)) for "
            "i,arm in enumerate(arms)}\nfig,axes=plt.subplots(2,2,fig"
            "size=(14,9),sharex=True,sharey=True)\nfor fee,ax in zip("
            "[0,5,10,25],axes.flat):\n    for arm in arms:\n        fr"
            "ame=daily[daily.path_id.eq(f'etf8/cost{fee}/{arm}')].so"
            "rt_values('session')\n        ax.plot(pd.to_datetime(fra"
            "me.session),frame.close_nav/100000,\n          color='bl"
            "ack' if arm=='equal_weight' else colors[arm],\n         "
            " linewidth=2.2 if arm=='equal_weight' else .7,alpha=.95"
            " if arm=='equal_weight' else .55)\n    ax.set(title=f'{f"
            "ee} bps per side',ylabel='Wealth / initial cash',yscale"
            "='log')\n    ax.grid(alpha=.2)\nfig.suptitle('All 26 arms"
            " in each fee scenario; equal weight shown in black')\nfi"
            "g.tight_layout();plt.show()\nfig,ax=plt.subplots(figsize"
            "=(10,6))\nfor fee,marker in zip([0,5,10,25],['o','s','^'"
            ",'D']):\n    subset=metrics[metrics.fee_bps.eq(fee)]\n   "
            " ax.scatter(subset.annual_volatility_sample_ddof1,\n    "
            "  subset.annual_return_252_session_convention,label=f'{"
            "fee} bps',marker=marker,alpha=.7)\nax.set(xlabel='Annual"
            "ized realized volatility',ylabel='Annualized return',ti"
            "tle='Risk and return: all 104 supported paths')\nax.lege"
            "nd();ax.grid(alpha=.2);fig.tight_layout();plt.show()"
        ),
    ),
    (
        "markdown",
        (
            "## Scope of the result\n\nAll ETF paths have full develop"
            "ment support. Stock portfolios stop when an owned asset"
            " lacks its required opening mark; no prices or exits ar"
            "e invented, and partial prefixes are not scored as full"
            "-period results. The broader 208-path family therefore "
            "remains incomplete. These plots do not replace paired u"
            "ncertainty, final recipe selection, or the untouched sh"
            "ared final-period evaluation."
        ),
    ),
]


def build():
    root = Path(__file__).resolve().parents[1]
    document = nb.v4.new_notebook(
        cells=[getattr(nb.v4, "new_" + kind + "_cell")(source) for kind, source in CELLS]
    )
    document.metadata.kernelspec = {
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
            document,
            km=manager,
            timeout=240,
            startup_timeout=180,
            resources={"metadata": {"path": str(root)}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    output = root / "notebooks/09_risk_economic_analysis.ipynb"
    nb.write(document, output)
    print(output)


if __name__ == "__main__":
    build()
