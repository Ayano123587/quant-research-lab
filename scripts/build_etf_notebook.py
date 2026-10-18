"""Build and execute the portable completed-ledger ETF analysis notebook."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import nbformat as nbf
from jupyter_client import KernelManager
from nbclient import NotebookClient


def build(report, output):
    root = Path(__file__).resolve().parents[1]
    relative = Path(report).resolve().relative_to(root).as_posix()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook(
        cells=[
            md(
                "# ETF allocation controls: a completed, full-support development study\n\n"
                "Does a fixed trend gate or volatility scaler change the return/risk tradeoff "
                "of eight strategic ETF sleeves? This analysis uses the completed cloud ledgers "
                "for **every one of 1,975 XNYS sessions, March 1, 2007–December 31, 2014**, "
                "for all four prespecified controls. It does not fit a model or rerun a strategy. "
                "**These are development results, not a final holdout evaluat"
                "ion or a reliable-edge claim.**\n\n"
                "The public CSV contains only derived portfolio accounting results. The original "
                "OHLC and provider responses remain private. Run `uv run --group research python "
                "scripts/build_etf_notebook.py --report reports/etf_controls_v3 --output "
                "notebooks/04_etf_controls.ipynb` to reproduce this notebook "
                "from the public results."
            ),
            md(
                "## Contract and source verification\n\n"
                "The four controls use SPY, EFA, EEM, TLT, LQD, GLD, DBC and VNQ. Each starts "
                "with $100,000 at the February 28, 2007 decision close, trades at the following "
                "scheduled open, pays 10 bp on each traded side, and earns zero on cash. "
                "There are no shorts or leverage. The exporter authenticated all fourteen "
                "scientific study outputs before reading any ledger. Here we independently "
                "authenticate all portable report artifacts before calculation.\n\n"
                "The Yahoo release is a separate whole-source vintage. Every adjusted OHLC "
                "proxy is observed provider OHLC × same-row published AdjClose/Close. "
                "It is not the original archive and has no archive-SHA alias. Adjusted opens "
                "are research proxies; actual auction fills, dividend cash settlement and "
                "historical publication/vintage availability are unverified."
            ),
            code(
                "from pathlib import Path\nimport hashlib, json, math\nimport numpy as np\n"
                "import pandas as pd\nfrom IPython.display import display, Image\n\n"
                "root = next(p for p in (Path.cwd(), *Path.cwd().parents) "
                "if (p / 'pyproject.toml').exists())\n"
                f"report = root / {relative!r}\n"
                "sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()\n"
                "manifest = json.loads((report / 'export_manifest.json').read_text())\n"
                "assert (report / 'COMPLETE').read_text().strip() == sha(repo"
                "rt / 'export_manifest.json')\n"
                "assert manifest['complete'] and manifest['portfolio_rows'] == 7900\n"
                "for name, expected in manifest['outputs_sha256'].items():\n"
                "    assert sha(report / name) == expected, name\n"
                "provenance = json.loads((report / 'provenance.json').read_text())\n"
                "assert provenance['source_identity']['archive_sha256'] is None\n"
                "assert not provenance['holdout_evaluated'] and not provenance['strategy_rerun']\n"
                "print('Verified portable report artifact hashes; provenance records exporter "
                "verification of 14 cloud outputs. Four controls × 1,975 sessions.')\n"
                "display(pd.Series({k: provenance[k] for k in "
                "['study_manifest_sha256','price_basis','sessions_per_policy'"
                ",'initial_cash','fee_bps_per_traded_side']}))"
            ),
            md(
                "## Independent accounting and metric reconstruction\n\n"
                "For daily net return, $r_t=V_t/V_{t-1}-1$, with $V_0=100000$. "
                "Annual return uses $(V_T/V_0)^{252/T}-1$; sample volatility is "
                "$s(r)\\sqrt{252}$ with ddof=1; Sharpe is $\\bar r/s(r)\\sqrt{252}$ "
                "against the declared zero cash rate. Drawdown uses the largest earlier "
                "close NAV **including initial capital**. Fees and full buy-plus-sell turnover "
                "are summed over every measured day, including initial entry.\n\n"
                "The following calculation uses the CSV directly and checks the cloud-derived "
                "metric export. It also verifies cash plus sleeve values equals NAV, "
                "10 bp × traded notional equals fees, and NAV changes equal daily money P&L."
            ),
            code(
                "daily = pd.read_csv(report / 'portfolio_daily.csv', parse_da"
                "tes=['session'], float_precision='round_trip')\nsaved = pd.re"
                "ad_csv(report / 'metrics.csv', index_col='policy', float_pre"
                "cision='round_trip')\npolicies = manifest['policies']\nrows = "
                "[]\nfor policy in policies:\n    f = daily.loc[daily.policy =="
                " policy].copy()\n    assert len(f) == 1975 and f.session.is_u"
                "nique and f.session.is_monotonic_increasing\n    nav = f.clos"
                "e_nav.to_numpy()\n    previous = np.r_[100000., nav[:-1]]\n   "
                " r = nav / previous - 1\n    values = f.filter(regex='^value_"
                "').sum(axis=1).to_numpy()\n    np.testing.assert_allclose(nav"
                ", values + f.cash, rtol=0, atol=1e-7)\n    np.testing.assert_"
                "allclose(f.fee, .001 * f.traded_notional, rtol=0, atol=1e-8)"
                "\n    np.testing.assert_allclose(f.money_pnl, nav - previous,"
                " rtol=0, atol=1e-7)\n    sigma = r.std(ddof=1)\n    row = dict"
                "(policy=policy, net_return=nav[-1]/100000.-1,\n        annual"
                "_return_252_session_convention=(nav[-1]/100000.)**(252/len(n"
                "av))-1,\n        annual_volatility_sample_ddof1=sigma*np.sqrt"
                "(252),\n        annual_sharpe_zero_cash_reference=r.mean()/si"
                "gma*np.sqrt(252),\n        maximum_close_nav_drawdown=(nav/np"
                ".maximum.accumulate(np.maximum(nav,100000.))-1).min(),\n     "
                "   mean_close_risky_weight=np.mean(values/nav), total_fee_mo"
                "ney=math.fsum(f.fee),\n        total_full_buy_plus_sell_turno"
                "ver=math.fsum(f.full_turnover))\n    for metric, value in row"
                ".items():\n        if metric != 'policy':\n            assert "
                "np.isclose(value, saved.loc[policy,metric], rtol=1e-10, atol"
                "=1e-10), metric\n    rows.append(row)\nmetrics = pd.DataFrame("
                "rows).set_index('policy')\nformatted = metrics.T.astype(objec"
                "t).copy()\npercent_rows = [name for name in metrics if name n"
                "ot in ['annual_sharpe_zero_cash_reference', 'total_fee_money"
                "', 'total_full_buy_plus_sell_turnover']]\nfor name in formatt"
                "ed.index:\n    template = '{:.2%}' if name in percent_rows el"
                "se ('${:,.2f}' if name == 'total_fee_money' else '{:.3f}')\n "
                "   formatted.loc[name] = metrics[name].map(lambda value: tem"
                "plate.format(value))\nformatted.columns = ['Strategic equal',"
                " 'Trend', 'Volatility', 'Trend + volatility']\ndisplay(format"
                "ted)\n"
            ),
            md(
                "## The four controls\n\n"
                "| Control | Rule held fixed throughout development |\n|---|---|\n"
                "| Strategic equal | Monthly rebalance to 12.5% in each of eight sleeves |\n"
                "| Trend | Keep a sleeve's 12.5% budget only when its trailing 252-return "
                "price-index return is positive; inactive budgets stay in cash |\n"
                "| Volatility | Scale equal-sleeve budgets using 63 past daily returns and "
                "sample covariance to 10% annual risk; scaling capped at 1 |\n"
                "| Combined | Apply the same trend gate, then the same volatility scaler |\n\n"
                "A 10% forecast target is not a guarantee about subsequently realized risk. "
                "No cost-free counterfactual is constructed. Terminal holdings are marked "
                "at the observed close; the final target remains pending, without an "
                "invented January 2015 fill or liquidation charge."
            ),
            code(
                "display(Image(filename=str(report / 'equity.png')))\n"
                "display(Image(filename=str(report / 'drawdown.png')))"
            ),
            md(
                "## Risk, cash allocation and trading costs\n\n"
                "The trend gate materially lowers invested exposure during this observed "
                "history. Cash budgets are retained rather than renormalized into the "
                "remaining sleeves. Volatility scaling further changes exposure and "
                "trading activity. Rolling realized volatility needs 63 observations: "
                "the first 62 values are unavailable, not filled. The 2007 row below is "
                "a partial calendar year starting March 1."
            ),
            code(
                "display(Image(filename=str(report / 'rolling_risk.png')))\ndi"
                "splay(Image(filename=str(report / 'exposure.png')))\nannual ="
                " daily.assign(year=daily.session.dt.year).groupby(['year','p"
                "olicy']).daily_net_return.agg(lambda r: np.prod(1+r)-1).unst"
                "ack('policy')[policies]\ndisplay(annual.map(lambda value: f'{"
                "value:.2%}'))\nsupport = pd.DataFrame(provenance['monthly_tar"
                "get_support']).T\ndisplay(support[['monthly_decisions','termi"
                "nal_pending_target','last_intended_execution_at']])"
            ),
            md(
                "## Paired uncertainty: descriptive, after seeing development results\n\n"
                "All three controls are compared with strategic equal sleeves using shared "
                "observed-session indices, circular contiguous blocks of 21/63/126 sessions, "
                "2,000 draws per block length and seed 20261017. This resamples the actual "
                "net-return observations; it does not invent prices, refit a model, or "
                "replay execution costs. The stored uncertainty report's exact return-input "
                "hashes were matched to these ledgers by the exporter.\n\n"
                "These intervals were added **after** the control scores were observed. "
                "They are not 2014-preregistered, not adjusted for strategy selection, "
                "and assume representative history and adequate dependence capture by "
                "the chosen block lengths. Every 95% interval crosses zero."
            ),
            code(
                "uncertainty = json.loads((report / 'uncertainty.json').read_text())\n"
                "intervals = []\n"
                "for policy, result in uncertainty['comparisons'].items():\n"
                "    assert not result['selection_adjusted']\n"
                "    for block in result['block_sensitivity']:\n"
                "        lo, hi = block['annual_sharpe_difference_interval']\n"
                "        assert lo <= 0 <= hi\n"
                "        intervals.append(dict(policy=policy, block_sessions="
                "block['block_sessions'],\n"
                "            observed_Sharpe_difference=result['observed_annu"
                "al_sharpe_difference'],\n"
                "            lower_95=lo, upper_95=hi))\n"
                "display(pd.DataFrame(intervals).set_index(['policy','block_sessions']).round(4))\n"
                "display(Image(filename=str(report / 'uncertainty.png')))"
            ),
            md(
                "## What the completed study supports\n\n"
                "The combined control has the highest **observed** Sharpe (about 0.697 versus "
                "0.420 for strategic equal sleeves) and lower close drawdown (about 11.2% "
                "versus 39.3%). Its cumulative net return is lower (49.2% versus 52.0%). "
                "Trend alone has a slightly higher cumulative return (52.8%), with lower "
                "risk and less invested exposure than strategic equal sleeves. These "
                "differences describe the same complete period and fixed accounting rules.\n\n"
                "The combined control's Sharpe difference is about +0.277, but its "
                "63-session-block 95% interval is roughly **[-0.254, +0.698]**. All other "
                "reported block choices and control pairs also span zero. The evidence "
                "does not establish statistically reliable superiority. Lower exposure "
                "and the observed crisis path help explain the drawdown comparison; "
                "they do not identify a universally better allocation.\n\n"
                "This is a fixed development control study, conditional on a current-vintage "
                "adjusted-price proxy and an explicit execution-cost model. There is no "
                "2015+ strategy score, final winner freeze, slippage/spread/impact study, "
                "tax analysis, explicit distribution settlement, or point-in-time "
                "publication-vintage proof. Later risk integration must first freeze a "
                "controlled economic protocol, then use the shared ledger and paired "
                "support; forecast losses alone are not portfolio performance."
            ),
        ]
    )
    episode_sha = hashlib.sha256(
        (root / "src/quantlab/evaluation/episodes.py").read_bytes()
    ).hexdigest()
    component_sha = hashlib.sha256(
        (root / "src/quantlab/evaluation/factorial.py").read_bytes()
    ).hexdigest()
    notebook.cells[-1:-1] = [
        md(
            "## Trend and scaling components\n\n"
            "Compare each switch with and without the other switch, plus the four-way "
            "interaction. Sharpe contrasts subtract the four separately calculated ratios; "
            "they are not the Sharpe of a return-difference series. Shared observed date "
            "blocks preserve the four-policy vector. These are post-hoc development "
            "intervals without an adjustment for strategy selection."
        ),
        code(
            "from datetime import date\n"
            "from quantlab.evaluation.factorial import ARMS, component_intervals\n"
            f"assert sha(root/'src/quantlab/evaluation/factorial.py') == {component_sha!r}\n"
            "component_returns = daily.pivot(index='session',columns='policy',\n"
            "    values='daily_net_return').sort_index()\n"
            "assert component_returns.shape == (1975,4)\n"
            "inputs = {arm:component_returns[arm].to_numpy() for arm in ARMS}\n"
            "component_rows = []\n"
            "for block in (21,63,126):\n"
            "    analysis = component_intervals([day.date() for day in component_returns.index],\n"
            "        inputs,block_sessions=block)\n"
            "    assert analysis['undefined_sharpe_draws'] == 0\n"
            "    for contrast in analysis['contrasts']:\n"
            "        lo,hi = contrast['annual_sharpe_95_interval']\n"
            "        component_rows.append(dict(component=contrast['name'],block_sessions=block,\n"
            "            Sharpe_difference=contrast['annual_sharpe_difference'],\n"
            "            lower_95=lo,upper_95=hi))\n"
            "display(pd.DataFrame(component_rows).set_index(['component','block_sessions']).round(4))"
        ),
        md(
            "## Reference drawdowns and missed rebounds\n\n"
            "This descriptive rule enumerates every equal-weight reference drawdown of at least "
            "10%, ending at its first recovery to the prior peak. It permits one active episode "
            "at a time and retains unrecovered episodes as censored. The rule was added after "
            "the development run; eventual troughs and recoveries are hindsight outcomes, "
            "not historical signals. Every policy is compared over identical reference dates."
        ),
        code(
            "from datetime import date\n"
            "from quantlab.evaluation.episodes import drawdown_episodes\n"
            "import matplotlib.pyplot as plt\n"
            f"assert sha(root/'src/quantlab/evaluation/episodes.py') == {episode_sha!r}\n"
            "navs = daily.pivot(index='session',columns='policy',values='close_nav').sort_index()\n"
            "exposures = daily.pivot(index='session',columns='policy',values='risky_weight')\n"
            "assert navs.shape == (1975,4) and navs.notna().all().all()\n"
            "episodes = drawdown_episodes([day.date() for day in navs.index],\n"
            "    navs['strategic_equal'],initial_session=date(2007,2,28),initial_nav=100000)\n"
            "display(pd.DataFrame(episodes))\n"
            "comparisons = []\n"
            "fig, axes = plt.subplots(len(episodes),1,figsize=(13,5*len(episodes)),squeeze=False)\n"
            "for ax, episode in zip(axes[:,0],episodes,strict=True):\n"
            "    peak,trough,end = [pd.Timestamp(episode[key]) for key in\n"
            "                      ('peak_session','trough_session','end_session')]\n"
            "    for policy in policies:\n"
            "        start_nav = 100000 if peak == pd.Timestamp('2007-02-28') "
            "else navs.loc[peak,policy]\n"
            "        comparisons.append(dict(episode=episode['episode_id'],policy=policy,\n"
            "            peak_to_reference_trough=navs.loc[trough,policy]/start_nav-1,\n"
            "            reference_trough_to_end=navs.loc[end,policy]/navs.loc[trough,policy]-1,\n"
            "            mean_rebound_risky_weight=exposures.loc[trough:end,policy].mean(),\n"
            "            right_censored=episode['right_censored']))\n"
            "        (navs.loc[peak:end,policy]/start_nav).plot(ax=ax,label=policy)\n"
            "    ax.axvline(trough,color='gray',linestyle='--',label='Reference trough')\n"
            "    ax.set_title(episode['episode_id'])\n"
            "    ax.set_ylabel('NAV / own NAV at reference peak')\n"
            "    ax.legend(); ax.grid(alpha=.2)\n"
            "fig.tight_layout(); plt.show()\n"
            "display(pd.DataFrame(comparisons).set_index(['episode','policy']).round(4))"
        ),
        md(
            "In the first reference episode, trend reduced the peak-to-trough loss, but captured "
            "much less of the subsequent recovery. In the second, trend alone did not reduce "
            "the loss. These are reference-window comparisons, not each policy's own maximum "
            "drawdown or evidence that an eventual trough could be identified in real time."
        ),
    ]
    notebook.cells[-1:-1] = [
        md(
            "## Trading costs and owned asset P&L\n\n"
            "The four policies were rerun on all 1,975 development sessions at 0, 5, 10 "
            "and 25 bp per traded side. Targets are identical across fee scenarios; fees "
            "are funded from each portfolio's own NAV. Independent accounting checks used "
            "saved holdings and observed opening/closing marks for all 31,600 daily records. "
            "These assumed fees are sensitivity scenarios, not observed auction costs."
        ),
        code(
            "cost_report = root/'reports/etf_cost_sensitivity_v2'\n"
            "cost_manifest = json.loads((cost_report/'export_manifest.json').read_text())\n"
            "assert (cost_report/'COMPLETE').read_text().strip() == "
            "sha(cost_report/'export_manifest.json')\n"
            "assert cost_manifest['paths']==16 and cost_manifest['sessions_per_path']==1975\n"
            "assert not cost_manifest['holdout_evaluated']\n"
            "for name,digest in cost_manifest['outputs_sha256'].items():\n"
            "    assert sha(cost_report/name)==digest,name\n"
            "costs = pd.read_csv(cost_report/'metrics.csv',float_precision='round_trip')\n"
            "assert len(costs)==16 and costs.measured_sessions.eq(1975).all()\n"
            "display(costs.pivot(index='policy',columns='fee_bps',\n"
            "    values='annual_sharpe_zero_cash_reference').round(4))\n"
            "display(Image(filename=str(cost_report/'cost_sensitivity.png')))\n"
            "owned = pd.read_csv(cost_report/'monetary_contributions.csv',\n"
            "    float_precision='round_trip')\n"
            "totals = owned.groupby(['fee_bps','policy']).money_pnl.sum()\n"
            "for row in costs.itertuples():\n"
            "    assert np.isclose(totals.loc[row.fee_bps,row.policy]/100000,\n"
            "        row.net_return,rtol=0,atol=1e-10)\n"
            "display(owned[owned.fee_bps==10].pivot(index='policy',columns='component',\n"
            "    values='money_pnl').round(2))\n"
            "display(Image(filename=str(cost_report/'monetary_attribution.png')))"
        ),
        md(
            "The combined policy's observed Sharpe falls from about 0.718 with zero fees "
            "to 0.666 at 25 bp. Its annual return remains below equal sleeves, despite "
            "its higher observed Sharpe. Asset contributions sum in money and initial-NAV "
            "units; they are not separately compounded asset-return contributions. No fee "
            "scenario is selected as a winning recipe."
        ),
        md(
            "## Requested and realized volatility\n\n"
            "The 10% rule scales exposure down and never adds leverage. It cannot "
            "guarantee 10% realized risk. These saved 10 bp policy trajectories cover "
            "94 complete signal intervals and all 1,975 sessions for each control. "
            "Realized volatility includes actual changing holdings and fees; this is "
            "a policy diagnostic, not a fixed-portfolio variance forecast test."
        ),
        code(
            "calibration = root/'reports/etf_target_calibration_v1'\n"
            "calibration_manifest = json.loads((calibration/'export-manifest.json').read_text())\n"
            "assert (calibration/'COMPLETE').read_text().strip() == "
            "sha(calibration/'export-manifest.json')\n"
            "for name,digest in calibration_manifest['outputs_sha256'].items():\n"
            "    assert sha(calibration/name)==digest,name\n"
            "monthly_risk = pd.read_csv(calibration/'monthly_policy_risk.csv')\n"
            "assert len(monthly_risk)==376\n"
            "assert monthly_risk.groupby('policy').sessions.sum().eq(1975).all()\n"
            "all_targets = pd.read_csv(calibration/'signal_targets.csv')\n"
            "assert len(all_targets)==380 and (~all_targets.executed_in_window).sum()==4\n"
            "display(pd.read_csv(calibration/'summary.csv').set_index('policy'))\n"
            "display(Image(filename=str(calibration/'forecast_vs_policy_risk.png')))\n"
            "display(Image(filename=str(calibration/'monthly_realized_policy_risk.png')))"
        ),
        md(
            "The combined rule exceeds 10% annualized volatility in 16 of 94 monthly "
            "intervals. Its average intended cash weight is about 33.6%. The final "
            "December 2014 target remains pending and has no January 2015 outcome here. "
            "The identity line is a visual reference, not a claim that the signal-close "
            "forecast and changing net portfolio trajectory have identical targets."
        ),
    ]
    notebook.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"}
    )
    manager = KernelManager(kernel_name="python3")
    # Select this command's interpreter in memory; no global kernelspec is changed.
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        NotebookClient(
            notebook, km=manager, timeout=240, resources={"metadata": {"path": str(root)}}
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbf.write(notebook, output)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/etf_controls_v3")
    parser.add_argument("--output", default="notebooks/04_etf_controls.ipynb")
    build(**vars(parser.parse_args()))
