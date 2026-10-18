# Reproduce the analysis

Install the locked environment from the repository root:

```sh
uv sync --locked --group dev --group ml --group optimization --group research
uv run --locked python scripts/validate_saved_notebook_index.py
uv run --locked pytest
```

The nine published notebooks contain genuine executed outputs. The full shared 2015–10 November 2017 evaluation is still RUNNING; its final results notebook will follow after completion and verification. The operational-correction source archive records submitted code and pending evidence, not a completed scientific evaluation. Rebuild the ETF analysis from the public derived reports without refitting or rerunning strategies:

```sh
uv run --locked python scripts/build_etf_notebook.py --report reports/etf_controls_v3 --output etf-controls-reproduced.ipynb
```

For an independent source audit, download the [stock/ETF archive](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) to `data/raw/archive.zip`, then run:

```sh
uv run --locked quantlab audit --archive data/raw/archive.zip --output data/interim/source-audit
```

The archive must match the SHA256 recorded in the [source audit notebook](notebooks/00_source_audit.ipynb). Audit outputs retain source anomalies; normalization does not repair prices. Use fresh output directories.

## Recompute the studies

| Study | Source entrypoint | Required real inputs |
|---|---|---|
| Stock ranking | [`run_stock_development`](src/quantlab/stocks/development.py), [`run_wider_study`](src/quantlab/stocks/wider_study.py) | Authenticated prepared features and mature labels, registered configs; wider partitions and reference predictions for feature comparisons |
| Covariance forecasts | [`run_covariance_study`](src/quantlab/risk/study.py) | Authenticated market snapshot, XNYS calendar, risk config and separate ETF release |
| ETF allocation | [`run_development`](src/quantlab/allocation/study.py) | Authenticated source histories, calendar and explicit dates, capital and fees |

Full stock/risk fitting uses the guarded Kaggle runners in [`cloud/`](cloud/). Supply your own authenticated inputs and compute account; recorded private dataset identifiers do not grant access. Exact published ETF trajectories require the original Yahoo response vintage, authenticated through [`data/yahoo.py`](src/quantlab/data/yahoo.py). The stock archive alone is insufficient; a new download is a different vintage.

The [`shared evaluator`](cloud/stage_shared_holdout.py) additionally requires an actual frozen release, authorization, source/runtime inventories and input descriptors. Its forecast-before-outcome and one-shot release guards remain mandatory. Saved-report analysis is separate from an independent full research rerun.

## Frozen evaluation source

The exact evaluation source is [frozen-source-f520966f.zip](reproducibility/frozen-source-f520966f.zip). Its SHA-256 is `a86e3d0dd2941801ae769ced0cfda6f57407e3ba3cc371eb7123373f26a57d5e`; the frozen release SHA-256 is `f520966f25f426e6688f895bf51fdac25e887aecf2d26129d1b3243cfd663620`.

The [provenance and reconstruction map](reproducibility/frozen-source-f520966f.provenance.json) records all 331 mapped source, recipe and protocol files. Verify the full archive first, then all member hashes, and restore the mapped paths into a new directory. This preserves the exact frozen code independently of later development formatting or documentation.

The archive contains 66 release-pinned code/config files and 250 recipe files, plus the original protocol and authentication metadata. It contains no market observations or fitted model weights. Scientific reproduction also requires the separately authenticated data and frozen runtime. Saved report analysis uses the published report artifacts and requires no refitting.

Restore the exact frozen release source from the repository root:

```sh
python scripts/restore_frozen_source.py --output ../quant-frozen-f520966f
```

The destination must be a new directory outside this checkout. The script verifies the archive, provenance and all 332 members before restoring the 331 mapped source files; the pinned data and scientific runtime described above remain separate prerequisites.
