# Quant Research Lab

Three real-data studies share a trading calendar, past-only decisions, fee-funded accounting and chronological evaluation:

- **Stock ranking:** compare price/volume factors, fixed scores and learned rankings; separate prediction quality from portfolio support.
- **ETF allocation:** compare strategic weights, trend filters, volatility targets and covariance choices.
- **Portfolio risk:** compare eight covariance estimators, common-reference risk forecasts and capped allocations.

The supplied archive contains **8,539 unique files and 17,453,243 observations**, spanning January 1962–November 2017. Mirrored directory trees are counted once. Normalization retains source rows and quality flags, including 32 empty files; it does not repair prices or fill gaps.

The [archive publisher](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) describes adjusted prices, but historical membership, adjustment vintages and complete delisting outcomes are unverified. ETF allocation uses a separate Yahoo vintage, with same-row adjusted-close ratios applied to OHLC. Adjusted opens are research execution proxies, not verified auction fills. Cash earns zero interest; fees are explicit assumptions.

Development uses observations through 2014. The [shared protocol](configs/research.toml) defines a separate 2015–10 November 2017 evaluation: recipes freeze before outcomes, expanding fits use only mature labels, forecasts precede outcome scoring, and every registered comparison retains missing support and right-censoring. Full-period performance is reported only for supported complete paths. All 64 development stock portfolio paths stopped on missing or invalid owned prices; their partial ledgers are not full-period return estimates.

## Reproduce

Place the supplied archive at `data/raw/archive.zip`; set `QUANTLAB_ARCHIVE` to use another path. Raw data, derived panels, fitted models and private run artifacts are excluded from Git.

```sh
uv sync --locked --group ml --group optimization --group research
uv run --locked quantlab audit --archive data/raw/archive.zip --output data/interim/source-audit
uv run --locked pytest
```

This interim release contains nine genuine executed notebooks. The ongoing shared evaluation has no published final results or final notebook yet.

## Notebooks

- [Source audit](notebooks/00_source_audit.ipynb)
- [ETF controls](notebooks/04_etf_controls.ipynb) and [seven sensitivities](notebooks/10_etf_sensitivity.ipynb)
- [Stock models and portfolio support](notebooks/05_stock_development_analysis.ipynb), [feature ablation](notebooks/07_stock_feature_ablation.ipynb), and [training targets](notebooks/08_stock_target_comparison.ipynb)
- [Covariance forecasts](notebooks/06_risk_forecast_analysis.ipynb) and [risk-aware allocations](notebooks/09_risk_economic_analysis.ipynb)
- [Wider-feature main workflow](notebooks/01_stock_wider_training.ipynb): training, prediction and feature comparisons.
- Shared final evaluation: the full 2015–10 November 2017 evaluation is still RUNNING. Its final results notebook will follow after completion and verification. The [generator](scripts/build_shared_holdout_notebook.py) builds that analysis from authenticated saved evaluation tables.

The [ETF notebook builder](scripts/build_etf_notebook.py) reproduces plots from saved portfolio tables without refitting models.

See [reproduction instructions](REPRODUCE.md) for saved-report analysis and the exact frozen evaluation source.
