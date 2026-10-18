"""Portable full-period cost and monetary-attribution report; no strategy replay."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def export_cost_report(
    study, expected_manifest_sha256, replay_path, expected_replay_sha256, output
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    study, output = Path(study), Path(output)

    def sha(raw):
        return hashlib.sha256(raw).hexdigest()

    manifest_raw = (study / "manifest.json").read_bytes()
    replay_raw = Path(replay_path).read_bytes()
    if (
        sha(manifest_raw) != expected_manifest_sha256
        or (study / "COMPLETE").read_text().strip() != expected_manifest_sha256
        or sha(replay_raw) != expected_replay_sha256
    ):
        raise ValueError("Externally pinned completed study and accounting replay required")
    manifest, replay = json.loads(manifest_raw), json.loads(replay_raw)
    if (
        not manifest["complete"]
        or not replay["PASS"]
        or replay["manifest_sha256"] != expected_manifest_sha256
        or replay["paths"] != 16
        or replay["observed_ledger_sessions"] != 31600
        or replay["asset_session_ownership_checks"] != 252800
        or manifest["contract"]["holdout_evaluated"] is not False
    ):
        raise ValueError("All sixteen full-support reconciled development paths required")
    for name, digest in manifest["outputs_sha256"].items():
        if Path(name).name != name or sha((study / name).read_bytes()) != digest:
            raise ValueError("Scientific artifact differs: " + name)
    metrics = json.loads((study / "metrics.json").read_bytes())
    rows, contributions = [], []
    for fee in (0, 5, 10, 25):
        for policy in manifest["contract"]["policies"]:
            row = metrics[f"cost-{fee}bps"][policy]
            if row["measured_sessions"] != 1975:
                raise ValueError("Common full-period economic support required")
            rows.append({"fee_bps": fee, "policy": policy, **row})
            owned = json.loads((study / f"cost-{fee}bps-{policy}-attribution.json").read_bytes())
            parts = {**owned["asset_pnl"], "cash": owned["cash_pnl"], "fees": owned["fee_pnl"]}
            if not math.isclose(math.fsum(parts.values()), owned["net_pnl"], abs_tol=1e-7):
                raise ValueError("Cumulative actual monetary attribution differs")
            for asset, value in parts.items():
                contributions.append(
                    {
                        "fee_bps": fee,
                        "policy": policy,
                        "component": asset,
                        "money_pnl": value,
                        "initial_nav_contribution": value / 100000,
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    table, parts = pd.DataFrame(rows), pd.DataFrame(contributions)
    table.to_csv(output / "metrics.csv", index=False)
    parts.to_csv(output / "monetary_contributions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for policy, group in table.groupby("policy"):
        axes[0].plot(
            group.fee_bps, group.annual_sharpe_zero_cash_reference, marker="o", label=policy
        )
        axes[1].plot(
            group.fee_bps,
            group.annual_return_252_session_convention * 100,
            marker="o",
            label=policy,
        )
    axes[0].set_ylabel("Annualized Sharpe, zero-cash reference")
    axes[1].set_ylabel("Annual return, 252-session convention (%)")
    for ax in axes:
        ax.set_xlabel("Assumed fee per traded side (basis points)")
        ax.set_xticks([0, 5, 10, 25])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("All four fee assumptions; identical policy targets and 1,975 sessions")
    for extension in ("png", "svg"):
        fig.savefig(output / ("cost_sensitivity." + extension), dpi=160)
    plt.close(fig)
    primary = parts[parts.fee_bps == 10].pivot(
        index="policy", columns="component", values="initial_nav_contribution"
    )
    primary = primary.rename(
        columns=lambda name: name.removeprefix("etf:").removesuffix(".us").upper()
    )
    ax = primary.plot.barh(stacked=True, figsize=(14, 6))
    ax.legend(title="Component", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_xlabel("Additive P&L / initial NAV (not standalone compounded asset returns)")
    ax.set_title("Primary 10 bp scenario: actual owned asset P&L, cash and fees")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.figure.tight_layout()
    for extension in ("png", "svg"):
        ax.figure.savefig(output / ("monetary_attribution." + extension), dpi=160)
    plt.close(ax.figure)
    provenance = {
        "complete": True,
        "study_manifest_sha256": expected_manifest_sha256,
        "accounting_replay_sha256": expected_replay_sha256,
        "paths": 16,
        "sessions_per_path": 1975,
        "holdout_evaluated": False,
        "strategy_rerun": False,
        "cost_scenario_selection": False,
        "outputs_sha256": {p.name: sha(p.read_bytes()) for p in output.iterdir()},
    }
    raw = (json.dumps(provenance, indent=2) + "\n").encode()
    (output / "export_manifest.json").write_bytes(raw)
    (output / "COMPLETE").write_text(sha(raw) + "\n")
    return provenance
