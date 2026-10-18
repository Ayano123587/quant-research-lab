"""Describe realized policy volatility against saved targets; never rerun a policy."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MANIFEST = "38a5927ea6060db1b4c126e92abb91a58003ed5d209346a598fc5aad5b0cd8dd"
POLICIES = ("strategic_equal", "trend_252", "volatility_63", "trend_252_volatility_63")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(study, output):
    assert sha(study / "manifest.json") == MANIFEST
    assert (study / "COMPLETE").read_text().strip() == MANIFEST
    manifest = json.loads((study / "manifest.json").read_bytes())
    # The complete saved ledgers were independently checked against actual prices.
    replay = study.parents[1] / "root-independent-accounting-replay.json"
    replay_sha = "3afc5dde349973beaacd71125ab901a2908c1471898e894f67a5eb82e9d86aba"
    assert sha(replay) == replay_sha
    assert manifest["complete"] and not manifest["contract"]["holdout_evaluated"]
    inventory = manifest["outputs_sha256"]
    rows, targets_out = [], []
    for policy in POLICIES:
        ledger_name = f"cost-10bps-{policy}.parquet"
        target_name = f"cost-10bps-{policy}-targets.json"
        for name in (ledger_name, target_name):
            entry = inventory[name]
            expected = entry["sha256"] if isinstance(entry, dict) else entry
            assert sha(study / name) == expected
        ledger = pd.read_parquet(study / ledger_name).sort_values("session")
        assert len(ledger) == 1975 and not ledger.session.duplicated().any()
        assert str(ledger.session.iloc[0]) == "2007-03-01"
        assert str(ledger.session.iloc[-1]) == "2014-12-31"
        previous = ledger.close_nav.shift(1, fill_value=100000.0)
        ledger["net_return"] = ledger.close_nav / previous - 1
        assert np.isfinite(ledger.net_return).all()
        targets = json.loads((study / target_name).read_bytes())
        assert len(targets) == 95
        dates = [pd.Timestamp(t["signal_at"]).date() for t in targets]
        assert dates == sorted(set(dates)) and str(dates[-1]) == "2014-12-31"
        covered = []
        for index, target in enumerate(targets):
            target_row = {
                "policy": policy,
                "signal": str(dates[index]),
                "scale": target["scale"],
                "cash_weight": target["cash_weight"],
                "forecast_before": target["forecast_annual_volatility_before"],
                "forecast_after": target["forecast_annual_volatility_after"],
                "executed_in_window": index < 94,
            }
            targets_out.append(target_row)
            if index == 94:
                continue  # The final target is pending; no January 2015 outcome read.
            measured = ledger[ledger.session.gt(dates[index]) & ledger.session.le(dates[index + 1])]
            assert len(measured) > 1
            covered.extend(measured.session.tolist())
            rows.append(
                {
                    **target_row,
                    "through_close": str(dates[index + 1]),
                    "sessions": len(measured),
                    "realized_annual_volatility": float(
                        measured.net_return.std(ddof=1) * np.sqrt(252)
                    ),
                    "net_return": float(np.prod(1 + measured.net_return) - 1),
                }
            )
        assert covered == ledger.session.tolist()
    frame, targets = pd.DataFrame(rows), pd.DataFrame(targets_out)
    assert len(frame) == 376 and len(targets) == 380
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "monthly_policy_risk.csv", index=False)
    targets.to_csv(output / "signal_targets.csv", index=False)
    summary = frame.groupby("policy").agg(
        months=("signal", "size"),
        sessions=("sessions", "sum"),
        median_monthly_annual_volatility=("realized_annual_volatility", "median"),
        mean_monthly_annual_volatility=("realized_annual_volatility", "mean"),
        fraction_months_over_requested_10pct=(
            "realized_annual_volatility",
            lambda x: float(x.gt(0.1).mean()),
        ),
        mean_intended_cash=("cash_weight", "mean"),
        fraction_unscaled=("scale", lambda x: float(np.isclose(x, 1).mean())),
    )
    summary.to_csv(output / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for policy, ax in zip(POLICIES[2:], axes, strict=True):
        data = frame[frame.policy.eq(policy)]
        ax.scatter(data.forecast_after, data.realized_annual_volatility, s=19, alpha=0.65)
        limit = max(0.12, data.realized_annual_volatility.max() * 1.05)
        ax.plot([0, limit], [0, limit], color="gray", lw=1)
        ax.axvline(0.1, color="black", ls=":", lw=1)
        ax.set_title(policy)
        ax.set_xlabel("Signal-close annual volatility forecast")
    axes[0].set_ylabel("Next signal interval: realized net policy volatility")
    fig.tight_layout()
    fig.savefig(output / "forecast_vs_policy_risk.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    for policy in POLICIES:
        data = frame[frame.policy.eq(policy)]
        ax.plot(pd.to_datetime(data.through_close), data.realized_annual_volatility, label=policy)
    ax.axhline(0.1, color="black", ls=":", lw=1, label="10% requested target")
    ax.set(
        ylabel="Annualized standard deviation of net daily returns", xlabel="Interval ending close"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(output / "monthly_realized_policy_risk.png", dpi=160)
    plt.close(fig)
    result = {
        "complete": True,
        "source_manifest_sha256": MANIFEST,
        "accounting_replay_sha256": replay_sha,
        "holdout_evaluated": False,
        "policy_months": 376,
        "all_saved_targets": 380,
        "sessions_per_policy": 1975,
        "fee_bps": 10,
        "scope": (
            "descriptive_saved_policy_trajectories; "
            "not_common_fixed_portfolio_variance_forecast_calibration"
        ),
        "interval": (
            "signal_close_exclusive_to_next_signal_close_inclusive; "
            "actual_old_overnight_new_intraday_owners_and_fees"
        ),
        "target": "requested10pct_is_a_cap_down_rule_not_a_realized_volatility_guarantee",
        "final_pending_target": "all4_2014-12-31_targets_preserved_without_2015_outcomes",
        "outputs_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "export-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETE").write_text(sha(output / "export-manifest.json") + "\n")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.study, args.output)
