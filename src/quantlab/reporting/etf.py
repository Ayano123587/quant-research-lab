"""Export authenticated completed ETF ledgers without rerunning a strategy.

Only portfolio accounting results are exported. Original OHLC, provider raw
responses, and model/source input partitions are never copied into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

POLICIES = ("strategic_equal", "trend_252", "volatility_63", "trend_252_volatility_63")
LABELS = {
    "strategic_equal": "Strategic equal sleeves",
    "trend_252": "Positive 252-return trend",
    "volatility_63": "63-return volatility scaling",
    "trend_252_volatility_63": "Trend + volatility scaling",
}
SYMBOLS = ("SPY", "EFA", "EEM", "TLT", "LQD", "GLD", "DBC", "VNQ")
STUDY_SHA = "109976808a1196c62111b14877271b94049ce6df7964e27b8fcdd4a82e054f3a"
RELEASE_SHA = "d749000ac4116a8f74ea0c42a7d28e94822369ffcee413ff6dc22100d3666ab3"
RAW_SHA = "c7cc178eb60b3674388deac4b84315d6776ed66800ba5f7e5418cbf6f736c240"
CALENDAR_SHA = "9e5c0a972ce2c7f83f5ffdde86234c96332807fb278eb8a3954422e8da7a83ce"
BASIS = "provider_adjusted_close_ratio_price_index_research_proxy"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def authenticate_study(study, expected_manifest_sha256):
    """Authenticate all fourteen scientific outputs before any Parquet read."""
    study = Path(study)
    if expected_manifest_sha256 != STUDY_SHA:
        raise ValueError("This report release requires the independently reviewed v3 study pin")
    if digest(study / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Externally pinned scientific manifest differs")
    if (study / "COMPLETE").read_bytes() != (expected_manifest_sha256 + "\n").encode():
        raise ValueError("Completed scientific output marker required")
    if (study / "FAILURE.json").exists():
        raise ValueError("Failed output cannot supply a completed report")
    manifest = json.loads((study / "manifest.json").read_bytes())
    if manifest.get("complete") is not True or manifest.get("status") != "complete":
        raise ValueError("Scientific study is incomplete")
    expected = {"contract.json", "metrics.json"} | {
        policy + suffix
        for policy in POLICIES
        for suffix in (".parquet", "-monthly-targets.json", "-decision-events.json")
    }
    inventory = manifest["outputs_sha256"]
    actual = {p.name for p in study.iterdir() if p.is_file()}
    if set(inventory) != expected or actual != expected | {"manifest.json", "COMPLETE"}:
        raise ValueError("Exact full four-control study inventory required")
    for name, sha in inventory.items():
        if digest(study / name) != sha:
            raise ValueError("Scientific output hash mismatch: " + name)
    contract = json.loads((study / "contract.json").read_bytes())
    if contract != manifest["contract"]:
        raise ValueError("Scientific contract copies differ")
    identity = contract["source_identity"]
    if not (
        contract["phase"] == "development_only_not_final_selection_or_test"
        and contract["archive_sha256"] is None
        and contract["initial_cash"] == 100000.0
        and contract["fee_bps_per_traded_side"] == 10.0
        and contract["cash"] == "explicit_zero_interest_assumption"
        and contract["start_signal"] == "2007-02-28"
        and contract["measurement_start"] == "2007-03-01"
        and contract["end_close"] == "2014-12-31"
        and contract["policies"] == list(POLICIES)
        and contract["symbols"] == list(SYMBOLS)
        and identity["source_type"] == "yahoo_etf_release"
        and identity["archive_sha256"] is None
        and identity["manifest_sha256"] == RELEASE_SHA
        and identity["raw_bundle_sha256"] == RAW_SHA
        and identity["price_basis"] == BASIS
        and identity["calendar_sha256"] == CALENDAR_SHA
        and contract["calendar"]["schedule_sha256"] == CALENDAR_SHA
    ):
        raise ValueError("Approved full-period controls/source price-basis contract differs")
    return manifest


def metrics_from_ledger(frame, initial_cash=100000.0):
    """Independent direct NAV arithmetic; no use of the engine metric function."""
    import numpy as np

    nav = frame["close_nav"].to_numpy(dtype=float)
    previous = np.concatenate(([initial_cash], nav[:-1]))
    returns = nav / previous - 1
    peaks = np.maximum.accumulate(np.maximum(nav, initial_cash))
    drawdowns = nav / peaks - 1
    sigma = float(np.std(returns, ddof=1))
    values = frame[["value_" + symbol for symbol in SYMBOLS]].sum(axis=1).to_numpy()
    return {
        "measured_sessions": len(nav),
        "net_return": float(nav[-1] / initial_cash - 1),
        "annual_return_252_session_convention": float(
            (nav[-1] / initial_cash) ** (252 / len(nav)) - 1
        ),
        "annual_volatility_sample_ddof1": sigma * math.sqrt(252),
        "annual_sharpe_zero_cash_reference": float(np.mean(returns)) / sigma * math.sqrt(252)
        if sigma > 0
        else None,
        "maximum_close_nav_drawdown": float(np.min(drawdowns)),
        "mean_close_risky_weight": float(np.mean(values / nav)),
        "total_fee_money": math.fsum(frame["fee"]),
        "total_full_buy_plus_sell_turnover": math.fsum(frame["full_turnover"]),
        "maximum_money_reconciliation_residual": float(
            frame["reconciliation_residual"].abs().max()
        ),
        "drawdown_basis": "measured_session_closes_plus_initial_cash_no_intraday_extrema",
    }


def verify_metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Recalculated metric inventory differs")
    for name, value in actual.items():
        other = expected[name]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Independent ledger metric differs: " + name)
        elif value != other:
            raise ValueError("Independent ledger metric differs: " + name)


def _figures(frame, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130}
    )
    colors = ("#243e73", "#007f73", "#cd6c27", "#8d4d9d")
    plots = (
        ("equity", "close_nav", "Portfolio close value, net of 10 bp per traded side", "USD"),
        (
            "drawdown",
            "drawdown",
            "Drawdown from prior close peak, including initial capital",
            "Drawdown",
        ),
        (
            "rolling_risk",
            "rolling_volatility_63",
            "63-session realized risk: sample daily volatility × √252",
            "Annual volatility",
        ),
        (
            "exposure",
            "risky_weight",
            "Observed close exposure: inactive sleeve budgets remain in cash",
            "Risky assets / NAV",
        ),
    )
    for name, column, title, ylabel in plots:
        fig, ax = plt.subplots(figsize=(11, 5.1), layout="constrained")
        for policy, color in zip(POLICIES, colors, strict=True):
            rows = frame.loc[frame["policy"] == policy]
            ax.plot(rows["session"], rows[column], label=LABELS[policy], color=color, lw=1.3)
        if name == "equity":
            ax.axhline(100000, color="#8a8a8a", lw=0.8, ls=":")
            ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        else:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if name == "rolling_risk":
            ax.axhline(0.10, color="#777777", lw=0.8, ls="--", label="10% ex-ante scaling target")
        ax.set(title=title, ylabel=ylabel, xlabel="XNYS session · March 2007–December 2014")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.16)
        ax.legend(loc="upper left", ncols=2, fontsize=8, frameon=False)
        fig.suptitle(
            "ETF controls · development only · adjusted-price research proxies",
            fontsize=10,
            color="#555555",
        )
        fig.savefig(output / (name + ".png"), dpi=180)
        fig.savefig(output / (name + ".svg"))
        plt.close(fig)


def _uncertainty_report(path, expected_sha256, frame, output):
    """Link already computed intervals to these exact observed returns; no sampling."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    if digest(path) != expected_sha256:
        raise ValueError("Externally pinned uncertainty output differs")
    report = json.loads(Path(path).read_bytes())
    if not (
        report["schema"] == "etf_control_uncertainty_report_v1"
        and report["study_manifest_sha256"] == STUDY_SHA
        and report["reference"] == "strategic_equal"
        and report["scope"]
        == "full_development_observed_after_control_scores_not_preregistered_inference"
        and set(report["comparisons"]) == set(POLICIES[1:])
    ):
        raise ValueError("Matching full-development posthoc uncertainty report required")
    reference = frame.loc[frame["policy"] == POLICIES[0]]
    nav = reference["close_nav"].to_numpy()
    base = nav / np.r_[100000.0, nav[:-1]] - 1
    rows = []
    for policy in POLICIES[1:]:
        result = report["comparisons"][policy]
        candidate = frame.loc[frame["policy"] == policy]
        nav = candidate["close_nav"].to_numpy()
        returns = nav / np.r_[100000.0, nav[:-1]] - 1
        observed = {
            "dates": candidate["session"].dt.strftime("%Y-%m-%d").tolist(),
            "candidate_reference_returns": np.column_stack((returns, base)).tolist(),
        }
        input_sha = hashlib.sha256(
            json.dumps(observed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if not (
            result["input_sha256"] == input_sha
            and result["sessions"] == 1975
            and result["method"] == "paired_circular_moving_block_percentile"
            and result["draws"] == 2000
            and result["seed"] == 20261017
            and result["confidence"] == 0.95
            and result["annual_sessions"] == 252
            and result["selection_adjusted"] is False
            and [r["block_sessions"] for r in result["block_sensitivity"]] == [21, 63, 126]
        ):
            raise ValueError("Uncertainty method or exact observed-return input hash differs")
        mean_difference = float((returns.mean() - base.mean()) * 252)
        sharpe_difference = float(
            math.sqrt(252) * (returns.mean() / returns.std(ddof=1) - base.mean() / base.std(ddof=1))
        )
        if not (
            math.isclose(
                mean_difference, result["observed_annual_arithmetic_mean_difference"], abs_tol=1e-10
            )
            and math.isclose(
                sharpe_difference, result["observed_annual_sharpe_difference"], abs_tol=1e-10
            )
        ):
            raise ValueError("Observed uncertainty comparison disagrees with ledger arithmetic")
        for block in result["block_sensitivity"]:
            for key in (
                "annual_sharpe_difference_interval",
                "annual_arithmetic_mean_difference_interval",
            ):
                lower, upper = block[key]
                if not (math.isfinite(lower) and math.isfinite(upper) and lower <= 0 <= upper):
                    raise ValueError("Report's all-intervals-cross-zero description is unsupported")
            rows.append(
                {
                    "policy": policy,
                    "block_sessions": block["block_sessions"],
                    "sharpe": result["observed_annual_sharpe_difference"],
                    "mean": result["observed_annual_arithmetic_mean_difference"],
                    "sharpe_interval": block["annual_sharpe_difference_interval"],
                    "mean_interval": block["annual_arithmetic_mean_difference_interval"],
                }
            )
    write_json(output / "uncertainty.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", sharey=True)
    for index, row in enumerate(rows):
        color = {21: "#243e73", 63: "#007f73", 126: "#cd6c27"}[row["block_sessions"]]
        for ax, key in zip(axes, ("mean", "sharpe"), strict=True):
            lower, upper = row[key + "_interval"]
            ax.hlines(index, lower, upper, color=color, lw=2)
            ax.plot(row[key], index, "o", color=color, markersize=4)
    axes[0].set_yticks(
        range(len(rows)),
        [LABELS[r["policy"]] + " · " + str(r["block_sessions"]) + " sessions" for r in rows],
        fontsize=8,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Annual arithmetic mean-return difference")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Annual Sharpe difference")
    for ax in axes:
        ax.axvline(0, color="#555555", ls="--", lw=0.9)
        ax.grid(axis="x", alpha=0.15)
        ax.set_xlabel("Control minus strategic equal sleeves")
    fig.suptitle(
        "Posthoc paired circular-block 95% intervals · 2,000 draws per block length\n"
        "All intervals cross zero · not selection-adjusted",
        fontsize=11,
    )
    fig.savefig(output / "uncertainty.png", dpi=180)
    fig.savefig(output / "uncertainty.svg")
    plt.close(fig)
    return {
        "report_sha256": expected_sha256,
        "input_return_hashes_verified": True,
        "posthoc_descriptive": True,
        "selection_adjusted": False,
        "pairs": 3,
        "block_lengths": [21, 63, 126],
        "draws_per_block": 2000,
    }


def export_report(
    study,
    expected_manifest_sha256,
    output,
    *,
    uncertainty_path=None,
    expected_uncertainty_sha256=None,
):
    """Fresh, portable derived-result export; never reconstruct or rerun trades."""
    import numpy as np
    import pandas as pd

    from quantlab.data.calendar import XNYSSchedule

    study, output = Path(study), Path(output)
    manifest = authenticate_study(study, expected_manifest_sha256)
    contract = manifest["contract"]
    schedule = XNYSSchedule(date(1962, 1, 1), date(2018, 12, 31))
    if schedule.provenance != contract["calendar"]:
        raise ValueError("Exact authenticated study calendar/runtime required")
    expected_days = pd.DatetimeIndex(
        [d for d in schedule.sessions if date(2007, 3, 1) <= d <= date(2014, 12, 31)]
    )
    if len(expected_days) != 1975:
        raise ValueError("Full actual 1975-session development support required")
    cloud_metrics = json.loads((study / "metrics.json").read_bytes())
    frames, metrics, target_summary = [], {}, {}
    for policy in POLICIES:
        frame = pd.read_parquet(study / (policy + ".parquet"))
        frame["session"] = pd.to_datetime(frame["session"])
        if not pd.DatetimeIndex(frame["session"]).equals(expected_days):
            raise ValueError("Incomplete or mismatched source ledger dates")
        rename = {"etf:" + symbol.lower() + ".us": "value_" + symbol for symbol in SYMBOLS}
        frame = frame.rename(columns=rename)
        numeric = frame.drop(columns="session")
        if not np.isfinite(numeric.to_numpy()).all() or (frame["close_nav"] <= 0).any():
            raise ValueError("Finite real ledger values and positive NAV required")
        values = frame[["value_" + symbol for symbol in SYMBOLS]].sum(axis=1)
        if (frame["cash"] < -1e-7).any() or (
            frame[["value_" + symbol for symbol in SYMBOLS]].to_numpy() < -1e-7
        ).any():
            raise ValueError("Unfunded or short positions contradict the declared controls")
        if not np.allclose(frame["close_nav"], values + frame["cash"], rtol=0, atol=1e-7):
            raise ValueError("NAV does not reconcile to observed holdings plus cash")
        if not np.allclose(frame["fee"], frame["traded_notional"] * 0.001, rtol=0, atol=1e-8):
            raise ValueError("Actual fee arithmetic differs from 10 bp per traded side")
        if not np.allclose(
            frame["full_turnover"],
            frame["traded_notional"] / frame["pretrade_nav"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Full buy-plus-sell turnover units differ")
        previous = np.r_[100000.0, frame["close_nav"].to_numpy()[:-1]]
        if not np.allclose(frame["money_pnl"], frame["close_nav"] - previous, rtol=0, atol=1e-7):
            raise ValueError("Full daily money P&L reconciliation failed")
        frame["policy"] = policy
        frame["daily_net_return"] = frame["close_nav"] / previous - 1
        frame["drawdown"] = (
            frame["close_nav"] / np.maximum.accumulate(np.maximum(frame["close_nav"], 100000)) - 1
        )
        frame["risky_weight"] = values / frame["close_nav"]
        frame["cash_weight"] = frame["cash"] / frame["close_nav"]
        frame["rolling_volatility_63"] = frame["daily_net_return"].rolling(63, min_periods=63).std(
            ddof=1
        ) * math.sqrt(252)
        metrics[policy] = metrics_from_ledger(frame)
        verify_metrics(metrics[policy], cloud_metrics[policy])
        targets = json.loads((study / (policy + "-monthly-targets.json")).read_bytes())
        if not targets or any(
            row["source_identity"] != contract["source_identity"] for row in targets
        ):
            raise ValueError("Monthly targets disagree with the authenticated source identity")
        target_summary[policy] = {
            "monthly_decisions": len(targets),
            "terminal_pending_target": targets[-1]["execution_within_requested_period"] is False,
            "last_signal_at": targets[-1]["signal_at"],
            "last_intended_execution_at": targets[-1]["intended_execution_at"],
        }
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    try:
        csv_path = output / "portfolio_daily.csv"
        combined.to_csv(csv_path, index=False, float_format="%.17g", date_format="%Y-%m-%d")
        # Verify the published CSV itself, rather than assuming serialization was lossless.
        public = pd.read_csv(csv_path, float_precision="round_trip", parse_dates=["session"])
        for policy in POLICIES:
            verify_metrics(
                metrics_from_ledger(public.loc[public["policy"] == policy]), metrics[policy]
            )
        pd.DataFrame.from_dict(metrics, orient="index").rename_axis("policy").to_csv(
            output / "metrics.csv", float_format="%.17g"
        )
        write_json(output / "metrics.json", metrics)
        provenance = {
            "schema": "etf_control_derived_portfolio_report_v1",
            "development_only": True,
            "holdout_evaluated": False,
            "strategy_rerun": False,
            "cloud_kernel_id": 139222404,
            "cloud_kernel_version": 3,
            "study_manifest_sha256": expected_manifest_sha256,
            "verified_scientific_outputs_sha256": manifest["outputs_sha256"],
            "source_identity": contract["source_identity"],
            "study_source_sha256": contract["source_code_sha256"],
            "calendar_sha256": CALENDAR_SHA,
            "initial_cash": 100000.0,
            "sessions_per_policy": 1975,
            "policies": list(POLICIES),
            "symbols": list(SYMBOLS),
            "fee_bps_per_traded_side": 10,
            "cash": contract["cash"],
            "price_basis": BASIS,
            "monthly_target_support": target_summary,
            "metric_conventions": {
                "annualization": 252,
                "volatility_ddof": 1,
                "drawdown": "close NAV peaks including initial capital; no intraday extrema",
                "rolling_risk": (
                    "63 complete observed daily net returns; first 62 values unavailable"
                ),
                "turnover": (
                    "daily buy plus sell notional divided by pretrade NAV; initial trade included"
                ),
            },
            "limitations": [
                "Development observations only; no final recipe selected here.",
                "Whole current-vintage Yahoo OHLC times observed same-row AdjClose/Close ratio.",
                "Derived adjusted opens are research proxies; vendor adjusted opens and auction "
                "fills are not observed.",
                "No explicit dividend cash settlement; historical publication/vintage "
                "availability is unverified.",
                "Cash earns zero; fees are 10 bp per traded side; no slippage, spread, "
                "impact or taxes.",
                "No leverage, shorts, target-risk guarantee, forced terminal liquidation, "
                "or pending-order charge.",
                "No out-of-sample superiority established. Any attached intervals are posthoc "
                "descriptions, not 2014-preregistered or selection-adjusted inference.",
            ],
            "report_exporter_sha256": digest(__file__),
        }
        _figures(public, output)
        if (uncertainty_path is None) != (expected_uncertainty_sha256 is None):
            raise ValueError("Uncertainty output and external SHA required together")
        if uncertainty_path is not None:
            provenance["uncertainty"] = _uncertainty_report(
                uncertainty_path, expected_uncertainty_sha256, public, output
            )
        write_json(output / "provenance.json", provenance)
        (output / "README.md").write_text(
            "# ETF control results: development only\n\n"
            "Four fixed controls, each with all 1,975 XNYS sessions from 2007-03-01 to "
            "2014-12-31. Initial capital is $100,000; fees are 10 basis points per traded "
            "side; cash earns zero. The CSV contains derived portfolio accounting results "
            "and sleeve values, never original market OHLC or provider responses.\n\n"
            "Controls: eight strategic equal sleeves; positive 252-return trend with "
            "inactive budgets left in cash; 63-return sample-covariance scaling to 10% "
            "annual risk, capped at one; and trend followed by that same scaling. "
            "There are no shorts or leverage.\n\n"
            "The whole-vintage Yahoo OHLC/AdjClose ratio produces adjusted-price research "
            "proxies. Adjusted opens are not observed auction fills. Historical publication "
            "availability is unverified; no dividend cash settlement, spread, impact, "
            "slippage or taxes are modeled.\n\n"
            "Metrics are independently recalculated from the published CSV. Annualization "
            "uses 252 sessions, daily volatility uses ddof=1, and close drawdown peaks "
            "include initial capital. Turnover includes both buys and sells and the "
            "initial purchase. Rolling risk needs 63 observations; its first 62 entries "
            "are explicitly unavailable.\n\n"
            "Attached paired circular-block intervals describe the already observed "
            "development results: three pairs against strategic equal sleeves, 21/63/126 "
            "session blocks, 2,000 draws each, seed 20261017. All 95% intervals cross zero. "
            "They are posthoc, not 2014-preregistered, and not selection-adjusted. "
            "No reliable edge, final recipe, or 2015+ evaluation is claimed.\n\n"
            "`export_manifest.json` binds every report artifact; `provenance.json` binds "
            "all fourteen cloud study outputs and the distinct Yahoo source identity. "
            "The executed portable analysis is `notebooks/04_etf_controls.ipynb`.\n",
            encoding="utf-8",
        )
        if authenticate_study(study, expected_manifest_sha256) != manifest:
            raise ValueError("Scientific source changed during report export")
        outputs = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        report = {
            "complete": True,
            "report_kind": "derived_portfolio_results_no_market_OHLC",
            "study_manifest_sha256": expected_manifest_sha256,
            "portfolio_rows": len(public),
            "policies": list(POLICIES),
            "outputs_sha256": outputs,
        }
        write_json(output / "export_manifest.json", report)
        (output / "COMPLETE").write_text(digest(output / "export_manifest.json") + "\n")
        return report
    except BaseException as error:
        write_json(output / "FAILURE.json", {"complete": False, "error_type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uncertainty-path")
    parser.add_argument("--expected-uncertainty-sha256")
    print(json.dumps(export_report(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
