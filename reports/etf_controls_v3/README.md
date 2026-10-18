# ETF control results: development only

Four fixed controls, each with all 1,975 XNYS sessions from 2007-03-01 to 2014-12-31. Initial capital is $100,000; fees are 10 basis points per traded side; cash earns zero. The CSV contains derived portfolio accounting results and sleeve values, never original market OHLC or provider responses.

Controls: eight strategic equal sleeves; positive 252-return trend with inactive budgets left in cash; 63-return sample-covariance scaling to 10% annual risk, capped at one; and trend followed by that same scaling. There are no shorts or leverage.

The whole-vintage Yahoo OHLC/AdjClose ratio produces adjusted-price research proxies. Adjusted opens are not observed auction fills. Historical publication availability is unverified; no dividend cash settlement, spread, impact, slippage or taxes are modeled.

Metrics are independently recalculated from the published CSV. Annualization uses 252 sessions, daily volatility uses ddof=1, and close drawdown peaks include initial capital. Turnover includes both buys and sells and the initial purchase. Rolling risk needs 63 observations; its first 62 entries are explicitly unavailable.

Attached paired circular-block intervals describe the already observed development results: three pairs against strategic equal sleeves, 21/63/126 session blocks, 2,000 draws each, seed 20261017. All 95% intervals cross zero. They are posthoc, not 2014-preregistered, and not selection-adjusted. No reliable edge, final recipe, or 2015+ evaluation is claimed.

`export_manifest.json` binds every report artifact; `provenance.json` binds all fourteen cloud study outputs and the distinct Yahoo source identity. The executed portable analysis is `notebooks/04_etf_controls.ipynb`.
