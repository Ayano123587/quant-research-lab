"""Build a readable results notebook only from an actual authenticated shared evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell(source, markdown=False):
    cell_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    return (
        {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(True)}
        if markdown
        else {
            "cell_type": "code",
            "metadata": {},
            "source": source.splitlines(True),
            "id": cell_id,
            "execution_count": None,
            "outputs": [],
        }
    )


def build_notebook(
    output,
    *,
    report_root,
    manifest_sha256,
    release_path,
    release_sha256,
    outer_complete_path,
    outer_complete_sha256,
    runtime_after_receipt_path,
    runtime_after_receipt_sha256,
    continuation_contract_sha256,
):
    """No market/model reads; unexecuted draft stays outside public notebooks/."""
    output, root = Path(output), Path(report_root)
    if output.suffix != ".ipynb" or "notebooks" in output.parts or output.exists():
        raise ValueError("New PRIVATE unexecuted draft required; publish only the executed copy")
    if (
        _sha(root / "manifest.json") != manifest_sha256
        or _sha(release_path) != release_sha256
        or _sha(outer_complete_path) != outer_complete_sha256
        or _sha(runtime_after_receipt_path) != runtime_after_receipt_sha256
    ):
        raise ValueError("Actual externally reviewed evaluation/release/outer completion required")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest.get("schema") != "shared_quant_frozen_evaluation_v1" or not manifest.get(
        "global_protocol_complete"
    ):
        raise ValueError(
            "An actual completed shared evaluation is required, not development outputs"
        )
    outer = json.loads(Path(outer_complete_path).read_bytes())
    inner = json.loads((root / "COMPLETE").read_bytes())
    if (
        not manifest["global_protocol_complete"]
        or manifest["release_sha256"] != release_sha256
        or outer["global_protocol_complete"] is not True
        or outer["release_sha256"] != release_sha256
        or outer["manifest_sha256"] != manifest_sha256
        or outer["contract_sha256"] != continuation_contract_sha256
        or outer["original_kernel_id"] != 139401694 or outer["original_version"] != 1
        or outer["runtime_after_receipt_sha256"] != runtime_after_receipt_sha256
        or inner["manifest_sha256"] != manifest_sha256
        or manifest["declared_scope"]
        != {
            "signal_dates": 35,
            "measured_sessions": 722,
            "covariance_forecast_cases": 12,
            "risk_economic_paths": 208,
        }
    ):
        raise ValueError("Actual full registered protocol scope required; support remains separate")
    for name, receipt in manifest["reporting_tables"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or receipt != manifest["artifacts"][name]
        ):
            raise ValueError("Authenticated contained readable table required")
        path = root.joinpath(*relative.parts)
        if (
            path.is_symlink()
            or path.stat().st_size != receipt["bytes"]
            or _sha(path) != receipt["sha256"]
        ):
            raise ValueError("Actual readable table bytes differ")
    setup = f"""import hashlib, json, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = Path(os.environ.get('QUANTLAB_SHARED_REPORT', 'reports/shared_holdout_v1'))
RELEASE = Path(os.environ.get('QUANTLAB_SHARED_RELEASE', str(ROOT/'final-release.json')))
OUTER = Path(os.environ.get('QUANTLAB_SHARED_OUTER_COMPLETE', str(ROOT/'outer-COMPLETE.json')))
RUNTIME_AFTER = Path(os.environ.get(
    'QUANTLAB_SHARED_RUNTIME_AFTER', str(OUTER.parent/'runtime-after-continuation.json')))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(ROOT/'manifest.json') == {manifest_sha256!r}
assert sha(RELEASE) == {release_sha256!r}
assert sha(OUTER) == {outer_complete_sha256!r}
manifest = json.loads((ROOT/'manifest.json').read_bytes())
release = json.loads(RELEASE.read_bytes())
outer = json.loads(OUTER.read_bytes())
assert outer['global_protocol_complete']
assert outer['manifest_sha256']==sha(ROOT/'manifest.json')
assert outer['contract_sha256']=={continuation_contract_sha256!r}
assert outer['original_kernel_id']==139401694 and outer['original_version']==1
assert outer['runtime_after_receipt_sha256']==sha(RUNTIME_AFTER)=={runtime_after_receipt_sha256!r}
assert json.loads((ROOT/'COMPLETE').read_bytes())['manifest_sha256']==sha(ROOT/'manifest.json')
tables = {{}}
for name, receipt in manifest['reporting_tables'].items():
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = ROOT/relative
    assert not path.is_symlink() and path.stat().st_size==receipt['bytes']
    assert sha(path)==receipt['sha256']
    tables[name] = json.loads(path.read_bytes())
def frame(suffix):
    rows = [r for name, values in tables.items() if name.endswith(suffix) for r in values]
    return pd.DataFrame(rows)
assert manifest['declared_scope']==dict(signal_dates=35, measured_sessions=722,
    covariance_forecast_cases=12, risk_economic_paths=208)
print('Verified release:', manifest['release_sha256'])
display(pd.Series(manifest['declared_scope'], name='Registered scope').to_frame())
support_keys=['global_protocol_complete','full_economic_common_support',
              'full_mature_statistical_support']
display(pd.Series({{k:manifest[k] for k in support_keys}}, name='Actual support').to_frame())
"""
    cells = [
        _cell(
            "# Shared final evaluation\n\n"
            "Frozen recipes on the 2015Ã¢â‚¬â€œ10 November 2017 window. Every registered forecast "
            "date and economic attempt is retained. Protocol completion and complete performance "
            "support are reported separately. No model is refitted by this analysis notebook.\n",
            True,
        ),
        _cell(setup),
        _cell(
            "## Forecast coverage and mature refits\n\n"
            "All 35 dates remain visible, including unsupported forecasts. Repeated fee arms "
            "share predictions but retain separate execution ledgers.\n",
            True,
        ),
        _cell("""support = frame('forecast_support.json')
assert support.groupby('arm_id').signal_session.nunique().eq(35).all()
display(support.groupby(['arm_id','status'], dropna=False).size().unstack(fill_value=0))
fit = support[support.fit_id.notna()].copy()
if not fit.empty:
    assert (pd.to_datetime(fit.maximum_feature_available_at)<=pd.to_datetime(fit.fit_cutoff)).all()
    display(fit.groupby('arm_id').agg(dates=('signal_session','nunique'),
        minimum_rows=('training_rows','min'), maximum_rows=('training_rows','max')))
    fit['signal_session']=pd.to_datetime(fit.signal_session)
    fit.pivot(index='signal_session',columns='arm_id',values='training_rows').plot(
        figsize=(12,5), legend=False, ylabel='Actual mature training rows')
    plt.show()
"""),
        _cell(
            "## Stock prediction\n\n"
            "Within-date IC uses observed mature labels, with its actual cross-sectional support. "
            "Right-censored months stay in the coverage table. Identical score streams across "
            "fee scenarios are not pooled as independent evidence.\n",
            True,
        ),
        _cell("""stock=frame('stock_date_metrics.json')
display(stock.groupby(['comparison_id','arm_id','outcome_status','status'],dropna=False).size().to_frame('dates'))
allowed={k for k,v in manifest['mature_paired_performance_allowed'].items() if v}
valid=stock[stock.status.eq('valid') & stock.comparison_id.isin(allowed)]
display(valid.groupby(['comparison_id','arm_id']).agg(mature_dates=('signal_session','nunique'),mean_IC=('ic','mean')))
primary=valid[valid.comparison_id.eq('stock_IC_fee10')].copy()
if not primary.empty:
    primary['signal_session']=pd.to_datetime(primary.signal_session)
    series=primary.pivot(index='signal_session',columns='arm_id',values='ic')
    series.plot(figsize=(12,5),ylabel='Within-date Spearman IC')
    plt.axhline(0,color='black',linewidth=.7); plt.show()
"""),
        _cell(
            "## Covariance forecast loss and support\n\n"
            "All 12 panel/window cases retain all eight estimators. QLIKE and variance errors "
            "remain in daily units on the same dated reference portfolio and outcome proxy.\n",
            True,
        ),
        _cell("""cov=frame('covariance_scores.json')
assert cov.groupby('comparison_id').estimator_id.nunique().eq(8).all()
assert cov.groupby(['comparison_id','estimator_id']).signal_session.nunique().eq(35).all()
display(cov.groupby(['comparison_id','estimator_id','status'],dropna=False).size().to_frame('dates'))
allowed={k for k,v in manifest['mature_paired_performance_allowed'].items() if v}
scored=cov[cov.status.eq('scored') & cov.comparison_id.isin(allowed)]
means=(pd.DataFrame() if scored.empty else
       scored.groupby(['comparison_id','estimator_id']).qlike.mean().unstack())
if means.empty:
    print('No covariance case has complete mature paired support; loss summaries are withheld.')
else:
    fig,ax=plt.subplots(figsize=(12,7))
    image=ax.imshow(means.to_numpy(),aspect='auto')
    ax.set_xticks(range(len(means.columns)),means.columns,rotation=60,ha='right')
    ax.set_yticks(range(len(means.index)),means.index)
    fig.colorbar(image,ax=ax,label='Mean observed daily QLIKE (descriptive support above)')
    fig.tight_layout(); plt.show()
"""),
        _cell(
            "## Economic support before performance\n\n"
            "A failed or unexecuted path has no full-window score. Initial fees and the first "
            "execution open-to-close interval are included. Terminal positions are marked, "
            "without an invented liquidation.\n",
            True,
        ),
        _cell("""economics=frame('portfolio_support_metrics.json')
display(economics.groupby(['comparison_id','complete','full_common_economic_support'],dropna=False).size().to_frame('paths'))
display(economics[~economics.complete][['comparison_id','arm_id','observed_session_count','failure','endpoint_status']])
supported=economics[economics.full_common_economic_support].copy()
assert supported.metrics.notna().all()
if not supported.empty:
    metric_rows=pd.json_normalize(supported.metrics).set_index(supported.index)
    display(pd.concat([supported[['comparison_id','arm_id']],metric_rows],axis=1))
daily=frame('portfolio_daily.json')
if daily.empty:
    print('No economic path was executed; no money rows or performance were invented.')
else:
    display(daily.groupby(['arm_id','full_path_support']).session.nunique().to_frame('actual_closes'))
    eligible=(daily.full_path_support & ~daily.is_initial_cash_close &
              daily.arm_id.isin(set(supported.arm_id)))
    full=daily[eligible]
    if not full.empty:
        assert full.groupby('arm_id').session.nunique().eq(722).all()
        display(full.groupby('arm_id').agg(paid_fees=('fee','sum'),
            maximum_reconciliation_residual=('reconciliation_residual',lambda s:s.abs().max())))
        etf_ids={r['arm_id'] for r in release['portfolio_arms'] if r['component']=='etf'}
        observed=full[full.arm_id.isin(etf_ids)].copy()
        if not observed.empty:
            observed['session']=pd.to_datetime(observed.session)
            wealth=observed.pivot(index='session',columns='arm_id',values='nav')
            wealth.plot(figsize=(12,5),ylabel='Actual net marked NAV')
            plt.show()
"""),
        _cell(
            "## Fixed paired uncertainty\n\n"
            "All declared block lengths are retained. Covariance uses the original "
            "sample-reference contrasts and scheduled-month index convention. "
            "Intervals are withheld for unsupported "
            "mature common cohorts or incomplete economic windows; they do not correct for "
            "development searches or establish superiority.\n",
            True,
        ),
        _cell("""intervals=frame('paired_intervals.json')
display(intervals.groupby(['kind','status'],dropna=False).size().to_frame('reported rows'))
display(intervals)
loss=intervals[(intervals.kind=='covariance_proxy') & intervals.status.eq('scored')]
for case,rows in loss.groupby('comparison_id'):
    fig,ax=plt.subplots(figsize=(10,4))
    for block,group in rows.groupby('block_length'):
        ax.vlines(group.estimator_a,group.lower,group.upper,alpha=.5)
        ax.scatter(group.estimator_a,group.mean_difference,label=f'{int(block)} months')
    ax.axhline(0,color='black',linewidth=.7); ax.set_title(case)
    ax.set_ylabel('Mean QLIKE difference from sample'); ax.tick_params(axis='x',rotation=45)
    ax.legend(); fig.tight_layout(); plt.show()
"""),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "actual_shared_manifest_sha256": manifest_sha256,
            "actual_release_sha256": release_sha256,
            "scope": "analysis_of_actual_saved_shared_evaluation_no_refit",
        },
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(notebook, stream, indent=1)
        stream.write("\n")
    return {
        "path": str(output),
        "sha256": _sha(output),
        "executed": False,
        "requires_actual_execution_before_publication": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--release-path", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--outer-complete-path", type=Path, required=True)
    parser.add_argument("--outer-complete-sha256", required=True)
    parser.add_argument("--runtime-after-receipt-path", type=Path, required=True)
    parser.add_argument("--runtime-after-receipt-sha256", required=True)
    parser.add_argument("--continuation-contract-sha256", required=True)
    print(json.dumps(build_notebook(**vars(parser.parse_args())), indent=2))
