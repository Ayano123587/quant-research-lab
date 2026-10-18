"""Execute the complete seven-path development sensitivity analysis."""

import sys
from pathlib import Path

import nbformat as nb
from jupyter_client import KernelManager
from nbclient import NotebookClient

CELLS = [
    (
        "markdown",
        """# ETF allocation sensitivity

Seven prespecified comparisons isolate the risk target, covariance estimate and
strategic weighting rule. Each path uses all 1,975 development sessions and a
10 bp fee per traded side. These are development comparisons, not held-out
performance. The reference reproduces the earlier combined-control ledger.
""",
    ),
    (
        "code",
        """from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'pyproject.toml').exists())
report = root/'reports/etf_sensitivity_v3'
raw = (report/'export-manifest.json').read_bytes()
manifest = json.loads(raw)
assert (report/'COMPLETE').read_text().strip() == hashlib.sha256(raw).hexdigest()
assert manifest['complete'] and not manifest['holdout_evaluated']
for name, pin in manifest['outputs_sha256'].items():
    assert hashlib.sha256((report/name).read_bytes()).hexdigest() == pin
daily = pd.read_parquet(report/'daily-paths.parquet')
metrics = pd.read_csv(report/'path-metrics.csv', index_col='arm', float_precision='round_trip')
wealth = daily.pivot(index='session', columns='arm', values='close_nav').sort_index()
returns = daily.pivot(index='session', columns='arm', values='daily_return').sort_index()
assert wealth.shape == returns.shape == (1975, 7)
assert wealth.notna().all().all() and returns.notna().all().all()
for arm in wealth:
    nav = np.r_[100000., wealth[arm].to_numpy()]
    r = nav[1:]/nav[:-1]-1
    np.testing.assert_array_equal(r, returns[arm])
    np.testing.assert_allclose(r.mean()/r.std(ddof=1)*np.sqrt(252),
        metrics.loc[arm, 'annual_sharpe_zero_cash_reference'], atol=1e-12, rtol=0)
display(metrics[['annual_return_252_session_convention',
    'annual_volatility_sample_ddof1', 'annual_sharpe_zero_cash_reference',
    'maximum_close_nav_drawdown', 'total_fee_money']].round(4))
""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
dates = pd.to_datetime(wealth.index)
for arm in wealth:
    nav = wealth[arm].to_numpy()
    width = 2.8 if arm == 'reference_sample_target10' else 1.2
    axes[0].plot(dates, nav/100000., label=arm, linewidth=width)
    peak = np.maximum.accumulate(np.r_[100000., nav])[1:]
    axes[1].plot(dates, nav/peak-1, linewidth=width)
axes[0].set_ylabel('Wealth / initial capital')
axes[1].set_ylabel('Close-to-close drawdown')
axes[0].legend(fontsize=8, ncol=2)
axes[0].set_title('Full development paths after 10 bp trading fees')
fig.tight_layout()
plt.show()
""",
    ),
    (
        "code",
        """reference = 'reference_sample_target10'
delta = returns.subtract(returns[reference], axis=0)
records = []
for block in [21, 63, 126]:
    rng = np.random.default_rng(20261018 + block)
    starts = rng.integers(0, len(delta), (2000, int(np.ceil(len(delta)/block))))
    # Prefix sums reproduce circular observed-date blocks without a large tensor.
    values = delta.to_numpy()
    doubled = np.concatenate([values, values], axis=0)
    prefix = np.vstack([np.zeros((1, 7)), doubled.cumsum(axis=0)])
    samples = np.zeros((2000, 7))
    remaining = len(delta)
    for j in range(starts.shape[1]):
        length = min(block, remaining)
        samples += prefix[starts[:, j]+length]-prefix[starts[:, j]]
        remaining -= length
    samples *= 252/len(delta)
    lower, upper = np.quantile(samples, [.025, .975], axis=0)
    for j, arm in enumerate(delta):
        records.append(dict(arm=arm, block_sessions=block,
            annualized_mean_difference=252*delta[arm].mean(),
            lower=lower[j], upper=upper[j]))
intervals = pd.DataFrame(records)
display(intervals.round(5))
fig, ax = plt.subplots(figsize=(11, 5))
subset = intervals[intervals.block_sessions == 63].set_index('arm')
for y, (arm, row) in enumerate(subset.iterrows()):
    ax.plot([row.lower, row.upper], [y, y], color='steelblue')
    ax.scatter(row.annualized_mean_difference, y, color='black')
ax.set_yticks(range(7), subset.index)
ax.axvline(0, color='grey', linestyle='--')
ax.set_xlabel('252 × mean daily return difference against reference')
ax.set_title('Descriptive paired intervals; 63-session observed-date blocks')
fig.tight_layout()
plt.show()
""",
    ),
    (
        "markdown",
        """The risk-budget weighting path has the highest development Sharpe
in this comparison. That does not establish a future edge or select the final
integrated strategy. The intervals resample observed dates; they generate no
market prices. They measure paired arithmetic mean-return differences, not
Sharpe or CAGR uncertainty, and do not adjust for multiple comparisons.

Accounting was independently reconciled from saved ownership components,
fees, trades and NAV. This is not an independent reconstruction of raw prices.
The 2015–2017 final period remains sealed.
""",
    ),
]


CELLS.insert(
    -1,
    (
        "markdown",
        """## Does the forecast choice improve the allocation?

The covariance study selects EWMA with a 63-observation half-life by mean
QLIKE on all 94 common mature dates for the eight-ETF, 63-return panel.
This criterion is separate from portfolio performance. The matched economic
comparison keeps all 95 signal clocks, source rows, trend masks, equal strategic
budgets, target volatility and fees fixed; only the covariance estimator changes.
""",
    ),
)
CELLS.insert(
    -1,
    (
        "code",
        """integration = root/'reports/etf_covariance_integration_v1'
raw = (integration/'export-manifest.json').read_bytes()
link = json.loads(raw)
assert (integration/'COMPLETE').read_text().strip() == hashlib.sha256(raw).hexdigest()
assert link['complete'] and not link['holdout_evaluated']
assert not link['selection_based_on_portfolio_scores']
for name, pin in link['outputs_sha256'].items():
    assert hashlib.sha256((integration/name).read_bytes()).hexdigest() == pin
losses = pd.read_csv(integration/'paired-forecast-losses.csv', index_col=0,
                     float_precision='round_trip')
assert losses.shape == (94, 8) and losses.notna().all().all()
assert losses.mean().idxmin() == link['selected_covariance_recipe']
display(losses.mean().sort_values().rename('Mean QLIKE; lower is better').to_frame())
economic = pd.read_csv(integration/'matched-path-metrics.csv', index_col='arm')
display(economic[['annual_return_252_session_convention',
    'annual_sharpe_zero_cash_reference', 'total_fee_money']])
print(link['interpretation'])
""",
    ),
)


def build():
    root = Path(__file__).resolve().parents[1]
    document = nb.v4.new_notebook(
        cells=[getattr(nb.v4, "new_" + kind + "_cell")(source) for kind, source in CELLS]
    )
    document.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
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
    output = root / "notebooks/10_etf_sensitivity.ipynb"
    nb.write(document, output)
    print(output)


if __name__ == "__main__":
    build()
