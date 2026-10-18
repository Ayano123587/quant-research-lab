"""Metadata-only recipe compiler; unresolved research decisions cannot open release."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bound_json(path, expected):
    raw = Path(path).read_bytes()
    _require(_sha(raw) == expected, "Externally supplied actual metadata SHA differs")
    return json.loads(raw)


def prepared_metadata(manifest, expected_sha256):
    """Read only authenticated JSON; never open feature/label Parquet columns."""
    value = _bound_json(manifest, expected_sha256)
    selected = [
        {**r, "session": Path(r["path"]).stem}
        for r in value["outputs"]
        if r.get("kind") in {"features", "labels"} and Path(r["path"]).stem <= "2014-12-31"
    ]
    for kind in ("features", "labels"):
        all_rows = [r for r in value["outputs"] if r.get("kind") == kind]
        rows = [r for r in selected if r["kind"] == kind]
        _require(
            len(all_rows) == 214 and len(rows) == 180,
            "Actual214 original partitions must filter to180 pre2015 partitions",
        )
    _require(
        len({(r["kind"], r["session"]) for r in selected}) == len(selected),
        "Duplicate original partition metadata",
    )
    return value, selected


def covariance_parameters(spec, count):
    """Expand unchanged registered defaults into the adapter's explicit contract."""
    from quantlab.risk.study import recipes

    _require(
        count in spec["panels"]["return_observations"], "Registered covariance window required"
    )
    return {
        r.recipe_id: {
            "method": r.method,
            "min_observations": count,
            "ewma_lambda": 2 ** (-1 / spec["estimators"]["ewma_half_life_observations"]),
            "n_factors": 1,
            "residual_floor": 0.0,
            "constant_correlation_intensity": None,
            "relative_tolerance": 1e-10,
            "absolute_tolerance": 1e-15,
            **dict(r.parameters),
        }
        for r in recipes(spec)
    }


def _evidence(value, input_files):
    _require(
        set(value) == {"stocks", "covariance", "etf", "integrations"},
        "All reviewed development components required",
    )
    for row in value.values():
        _require(
            row["reviewed_terminal_evidence"] is True,
            "Pending development evidence cannot authorize final recipes",
        )
        support = row["economic_support"]
        _require(
            support["classification"]
            in {"supported_common_cohort", "unsupported_preserved_prefix", "not_applicable"}
            and isinstance(support.get("original_status"), str)
            and bool(support["original_status"])
            and support["full_period_performance_allowed"]
            is (support["classification"] == "supported_common_cohort"),
            "Reviewed unsupported evidence must not become supported performance",
        )

        matches = [item for item in input_files.values() if item["path"] == row["path"]]
        _require(
            len(matches) == 1
            and matches[0]["sha256"] == row["sha256"]
            and type(matches[0]["bytes"]) is int
            and matches[0]["bytes"] > 0,
            "Every actual development evidence file must be an explicit authenticated input",
        )


def _stock_spec(candidate, config, partitions, panel_sha, *, projection, wider):
    from quantlab.stocks.development import FEATURE_IDS

    result = {
        "kind": "stock",
        "panel_root": "prepared",
        "partitions": partitions,
        "candidate_id": candidate.candidate_id,
        "candidate_kind": candidate.kind,
        "candidate_parameters": dict(candidate.parameters),
        "feature_columns": ["rank_" + f for f in FEATURE_IDS],
        "raw_columns": ["raw_" + f for f in FEATURE_IDS],
        "target_policy": config["target_transform"],
        "target_spec": config,
        "threads": config["threads"],
        "minimum_rows": config["minimum_training_rows"],
        "minimum_dates": config["minimum_training_signal_dates"],
        "projection": projection,
        "final_preparation": {
            "policy": "original_formulas_full7195_past_features_mature_labels_after_exports",
            "top_n": 500,
            "base_manifest_sha256": panel_sha,
        },
    }
    if projection["kind"] == "registered_wider":
        _require(
            wider is not None
            and wider["parent_kernel_id"] == 139317315
            and wider["parent_version"] == 3
            and wider["parent_canonical"],
            "Selected wider recipe requires actual immutable enriched parent metadata",
        )
        manifest = _bound_json(wider["manifest_path"], wider["manifest_sha256"])
        from cloud.shared_holdout_runner import WIDER_METADATA, normalize_wider_partitions

        _require(
            wider["manifest_sha256"] == WIDER_METADATA["panel"][1]
            and wider["input_root"] == "wider",
            "Actual fixed parent manifest and restored wider root required",
        )
        rows = normalize_wider_partitions(manifest)
        result["wider_panel_root"] = wider["input_root"]
        result["wider_partitions"] = rows
    return result


def compile_documents(
    repo,
    *,
    root_decision,
    descriptors,
    prepared_manifest_path,
    market_manifest_path,
    yahoo_manifest_path,
    development_evidence,
    input_files,
    source_sha256,
    runtime,
    calendar,
    solver_policy,
    selectors,
    wider=None,
):
    """Return bytes in memory only. Caller must independently review/freeze/publish.

    root_decision supplies actual dated/root-authenticated stock choices, resolved
    wider disposition, covariance integration IDs, and ETF scenario IDs. This
    function never ranks scores or chooses a winner. No evaluation data is read.
    """
    _require(
        root_decision and root_decision["wider_resolution"] != "UNRESOLVED",
        "Wider selection unresolved: no release documents may be compiled",
    )
    _require(
        root_decision["wider_resolution"] in {"original_six", "registered_wider"},
        "Explicit reviewed wider disposition required",
    )
    _require(
        root_decision["approved"] is True and root_decision["stock_choices"],
        "Explicit ROOT chosen learned controls required",
    )
    approved_at = datetime.fromisoformat(root_decision["approved_at"])
    _require(
        approved_at.tzinfo is not None and approved_at <= datetime.now(UTC),
        "Actual research approval clock required; never backdate preregistration",
    )
    _evidence(development_evidence, input_files)
    repo = Path(repo)
    for name in (
        "src/quantlab/evaluation/holdout_release.py",
        "configs/research.toml",
        "configs/risk-development.toml",
        "configs/risk-economic-development.toml",
        "configs/stock-development.toml",
        "scripts/evaluate_risk_calibration.py",
        "cloud/shared_holdout_runner.py",
    ):
        _require(
            source_sha256.get(name) == _sha((repo / name).read_bytes()),
            "Compiler and registered protocol/config bytes must be externally frozen",
        )
    protocol = tomllib.loads((repo / "configs/research.toml").read_text())
    risk = tomllib.loads((repo / "configs/risk-development.toml").read_text())
    economic = tomllib.loads((repo / "configs/risk-economic-development.toml").read_text())
    stock_config = tomllib.loads((repo / "configs/stock-development.toml").read_text())
    from quantlab.evaluation.holdout_tables import REPORTING_POLICY

    _require(
        risk["uncertainty"]["replicates"] == REPORTING_POLICY["draws"]
        and risk["uncertainty"]["seed"] == REPORTING_POLICY["seed"],
        "Actual registered risk bootstrap parameters differ",
    )
    freezes = {
        kind: _bound_json(row["freeze_path"], row["freeze_sha256"])
        for kind, row in descriptors.items()
    }
    manifest_shas = {
        "prepared": freezes["prepared"]["prepared_manifest_sha256"],
        "market": freezes["market"]["snapshot_manifest_sha256"],
        "etf": freezes["etf"]["release_manifest_sha256"],
    }
    panel_sha = manifest_shas["prepared"]
    panel, partitions = prepared_metadata(prepared_manifest_path, panel_sha)
    market = _bound_json(market_manifest_path, manifest_shas["market"])
    yahoo = _bound_json(yahoo_manifest_path, manifest_shas["etf"])
    _require(
        panel["config"]["calendar"] == calendar
        and panel["source_manifest_sha256"] == manifest_shas["market"],
        "Prepared market/calendar metadata differs",
    )
    for selector in selectors.values():
        actual = _bound_json(selector["path"], selector["sha256"])
        _require(
            actual["status"] == "development_selected_not_final_recipe",
            "Actual completed development selector required",
        )
        _require(
            actual["selected_candidate_id"] == selector["selected_candidate_id"],
            "Caller must preserve actual selector result, not relabel a learned winner",
        )
    from quantlab.data.provenance import SourceIdentity
    from quantlab.risk.economic_study import arms
    from quantlab.risk.study import recipes
    from quantlab.stocks.development import registered_candidates

    stocks = [r for r in market["source_inventory"] if r["asset_id"].startswith("stock:")]
    _require(len(stocks) == 7195, "All original7195 stock source identities required")
    market_id = SourceIdentity(
        "original_archive",
        calendar["schedule_sha256"],
        "source_OHLC_adjustment_unverified",
        archive_sha256=protocol["source_archive_sha256"],
        manifest_sha256=manifest_shas["market"],
    )
    # Yahoo metadata is authenticated here; actual identity follows its own release.
    yahoo_id = SourceIdentity(**root_decision["yahoo_source_identity"])
    _require(
        yahoo_id.manifest_sha256 == manifest_shas["etf"]
        and yahoo_id.calendar_sha256 == calendar["schedule_sha256"],
        "Actual Yahoo source identity required",
    )
    _require(yahoo["calendar"] == calendar, "Actual Yahoo calendar differs")
    histories = {
        "market": {
            "kind": "market_snapshot",
            "root": "market",
            "manifest_sha256": manifest_shas["market"],
            "assets": [["stock", r["asset_id"].split(":", 1)[1]] for r in stocks],
        },
        "etf": {
            "kind": "yahoo_etf_release",
            "root": "etf",
            "manifest_sha256": manifest_shas["etf"],
        },
    }
    documents, declarations, forecast_declarations, comparisons = {}, [], [], []
    policy = {
        "stock_prediction": {
            "aggregation": "equal_date_mean_IC_full_mature_common_paired_support",
            "paired_block_lengths": [3, 6, 12],
        },
        "covariance_proxy": {
            "aggregation": "paired_same_q_daily_losses_full_mature_dates_explicit_censoring",
            "paired_block_lengths": [3, 6, 12],
        },
        "portfolio": {
            "aggregation": "matched_fees_full_window_metrics_only_common_economic_support",
            "paired_block_lengths": [21, 63, 126],
        },
    }

    def add(name, component, spec, fee, *, history="market", refit="none", forecast_only=False):
        _require(
            name not in {r["arm_id"] for r in declarations + forecast_declarations},
            "Duplicate declared arm",
        )
        identity = market_id if history == "market" else yahoo_id
        row = {
            "arm_id": name,
            "component": component,
            "fee_bps": fee,
            "refit_policy": refit,
            "source_identity": identity.as_dict(),
            "input_id": history + "_manifest",
        }
        recipe = {
            "schema": "shared_quant_concrete_recipe_v1",
            **row,
            "history": histories[history],
            "decision": spec,
            "signal_policy": "initial_then_actual_month_end",
            "support_policy": "preserve_incomplete_no_fill_no_name_substitution",
            "initial_signal": "2014-12-31",
            "terminal_close": "2017-11-10",
            "development_evidence_sha256": development_evidence[component]["sha256"],
            "development_economic_support": development_evidence[component]["economic_support"],
        }
        path = "release/recipes/" + name + ".json"
        documents[path] = _raw(recipe)
        target = forecast_declarations if forecast_only else declarations
        target.append({**row, "recipe_path": path, "recipe_sha256": _sha(documents[path])})
        return name

    def compare(name, kind, names, **fields):
        comparisons.append(
            {
                "comparison_id": name,
                "kind": kind,
                "arm_ids": names,
                "analysis_policy": policy[kind],
                **fields,
            }
        )

    choices = [
        {
            "candidate_id": "control_fixed_composite",
            "config_path": "configs/stock-development.toml",
            "projection": {"kind": "original_six"},
        },
        {
            "candidate_id": "control_momentum_F01",
            "config_path": "configs/stock-development.toml",
            "projection": {"kind": "original_six"},
        },
        *root_decision["stock_choices"],
    ]
    stock_specs, stock_arms = {}, {}
    for choice in choices:
        config_path = repo / choice["config_path"]
        config = tomllib.loads(config_path.read_text())
        _require(
            source_sha256.get(choice["config_path"]) == _sha(config_path.read_bytes()),
            "Chosen config must be externally source-frozen",
        )
        candidate_map = {c.candidate_id: c for c in registered_candidates(config)}
        candidate = candidate_map[choice["candidate_id"]]
        tag = choice.get("recipe_id", candidate.candidate_id)
        _require(tag not in stock_specs, "Explicit stock recipes must be distinct")
        spec = _stock_spec(
            candidate, config, partitions, panel_sha, projection=choice["projection"], wider=wider
        )
        stock_specs[tag] = spec
        names = []
        for fee in economic["fees_bps"]:
            name = add(
                "stock_" + tag + "_fee" + format(fee, "g"),
                "stocks",
                spec,
                fee,
                refit="none" if candidate.kind == "control" else "monthly_expanding_mature",
            )
            stock_arms[tag, fee] = name
            names.append(name)
        compare(
            "same_stock_scores_" + tag,
            "stock_prediction",
            names,
            same_score_bytes=True,
            outcome_arm_id=names[0],
            minimum_support=stock_config["minimum_scored_cross_section"],
        )
    for fee in economic["fees_bps"]:
        names = [stock_arms[tag, fee] for tag in stock_specs]
        compare(
            "stock_IC_fee" + format(fee, "g"),
            "stock_prediction",
            names,
            same_score_bytes=False,
            outcome_arm_id=names[0],
            minimum_support=stock_config["minimum_scored_cross_section"],
        )
        compare("stock_economics_fee" + format(fee, "g"), "portfolio", names)

    selected_wider = any(
        s["projection"]["kind"] == "registered_wider" for s in stock_specs.values()
    )
    _require(
        selected_wider == (root_decision["wider_resolution"] == "registered_wider"),
        "Wider disposition must agree with actual selected recipe projections",
    )
    wider_binding = None
    input_files = dict(input_files)
    if selected_wider:
        from cloud.shared_holdout_runner import (
            WIDER_METADATA,
            WIDER_PARENT,
            authenticate_wider_metadata,
            authenticate_wider_review,
        )

        raw_metadata = {
            role: Path(wider["metadata_paths"][role]).read_bytes() for role in WIDER_METADATA
        }
        normalized = authenticate_wider_metadata(
            raw_metadata, project=repo, expected_runtime=runtime
        )
        review_raw = Path(wider["full_replay_review_path"]).read_bytes()
        authenticate_wider_review(review_raw, wider["full_replay_review_sha256"])
        wider_binding = {
            "parent": WIDER_PARENT,
            "destination": "wider",
            "partitions": normalized,
            "metadata_package_paths": {
                role: "release/wider-parent/" + role + ".json" for role in WIDER_METADATA
            },
            "review_package_path": "release/wider-parent/root-full-replay-review.json",
            "full_replay_review_sha256": wider["full_replay_review_sha256"],
        }
        additions = {
            "wider_manifest": {
                "path": "wider/manifest.json",
                "bytes": len(raw_metadata["panel"]),
                "sha256": WIDER_METADATA["panel"][1],
            },
            **{
                "wider_partition_" + row["session"]: {
                    "path": "wider/" + row["path"],
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in normalized
            },
        }
        _require(not set(additions) & set(input_files), "Explicit wider input identifier collision")
        input_files.update(additions)

    registered = recipes(risk)

    def parameters(count):
        return covariance_parameters(risk, count)

    estimator_ids = [r.recipe_id for r in registered]
    _require(len(estimator_ids) == 8, "Original eight covariance recipes required")

    def risk_spec(panel, count, arm, cap):
        return {
            "kind": "risk_portfolio",
            "observations": count,
            "panel": panel,
            "estimator_parameters": parameters(count),
            "economic_arm": dataclasses.asdict(arm),
            "cap": cap,
            "solver_policy": solver_policy,
            "projection_input_id": "execution_projection",
            "sample_reference_id": "sample_ddof1",
        }

    forecast_registers = {}
    for panel_name, panel, count, cap in (
        ("etf8", {"kind": "etf8_fixed"}, 63, economic["etf_cap"]),
        ("stock100", {"kind": "stock_activity", "asset_count": 100}, 252, economic["stock_cap"]),
    ):
        for fee in economic["fees_bps"]:
            names = []
            for arm in arms(risk):
                spec = risk_spec(panel, count, arm, cap)
                name = add(
                    "risk_" + panel_name + "_" + arm.arm_id + "_fee" + format(fee, "g"),
                    "covariance",
                    spec,
                    fee,
                    history="etf" if panel_name == "etf8" else "market",
                )
                names.append(name)
                if arm.arm_id == "equal_weight" and fee == 10:
                    forecast_registers[panel_name, count] = name
            compare("risk_economics_" + panel_name + "_fee" + format(fee, "g"), "portfolio", names)
    # Ten remaining forecast cells have no economic experiment in the protocol.
    # They require the explicitly reviewed forecast-only seam, not invented trades.
    for panel_name in ("etf8", "stock50", "stock100", "stock150"):
        for count in risk["panels"]["return_observations"]:
            if (panel_name, count) not in forecast_registers:
                panel = (
                    {"kind": "etf8_fixed"}
                    if panel_name == "etf8"
                    else {"kind": "stock_activity", "asset_count": int(panel_name[5:])}
                )
                name = add(
                    "forecast_" + panel_name + "_n" + str(count),
                    "covariance",
                    {
                        "kind": "risk_forecast",
                        "observations": count,
                        "panel": panel,
                        "estimator_parameters": parameters(count),
                    },
                    None,
                    history="etf" if panel_name == "etf8" else "market",
                    forecast_only=True,
                )
                forecast_registers[panel_name, count] = name
            name = forecast_registers[panel_name, count]
            compare(
                "covariance_" + panel_name + "_n" + str(count),
                "covariance_proxy",
                [name],
                estimator_ids=estimator_ids,
                pairs=[[a, b] for a in estimator_ids for b in estimator_ids if a < b],
            )
    _require(
        len(forecast_registers) == 12 and len(forecast_declarations) == 10,
        "All12 forecast cells with two exact economic-stream reuses required",
    )
    _require(
        sum(r["component"] == "covariance" for r in declarations) == 208,
        "Registered risk economics must remain exactly208 paths",
    )
    etf_names, etf_by_policy = [], {}
    for control in protocol["etf"]["policies"]:
        etf_by_policy[control] = add(
            "etf_" + control,
            "etf",
            {"kind": "etf_control", "policy": control, "covariance_method": "sample_ddof1"},
            protocol["execution"]["fee_bps_per_traded_side"],
            history="etf",
        )
        etf_names.append(etf_by_policy[control])
    compare("etf_four_controls", "portfolio", etf_names)
    from quantlab.allocation.sensitivity import SCENARIOS

    allowed_scenarios = {s.scenario_id for s in SCENARIOS}
    chosen_scenarios = root_decision["etf_scenario_ids"]
    _require(
        len(set(chosen_scenarios)) == len(chosen_scenarios)
        and set(chosen_scenarios) <= allowed_scenarios,
        "Only explicitly ROOT selected registered one-change ETF scenarios allowed",
    )
    sensitivity_names = [
        add(
            "etf_" + scenario,
            "etf",
            {"kind": "etf_sensitivity", "scenario_id": scenario},
            10.0,
            history="etf",
        )
        for scenario in chosen_scenarios
    ]
    if sensitivity_names:
        compare("etf_registered_sensitivities", "portfolio", sensitivity_names)
    selected_method = root_decision["etf_integration_covariance_method"]
    _require(
        selected_method in risk["estimators"]["methods"],
        "Explicit registered covariance method required for ETF integration",
    )
    selected_etf = add(
        "integration_etf_covariance",
        "integrations",
        {
            "kind": "etf_control",
            "policy": "trend_252_volatility_63",
            "covariance_method": selected_method,
        },
        10.0,
        history="etf",
    )
    compare(
        "integration_etf_same_policy_covariance",
        "portfolio",
        [etf_by_policy["trend_252_volatility_63"], selected_etf],
    )
    # Integration choices are explicit reviewed parameters, not a new selection.
    from quantlab.risk.economic_study import Arm

    integration = root_decision["stock_risk_integration"]
    _require(
        integration["stock_recipe_id"] in stock_specs
        and integration["recipe_id"] in estimator_ids
        and integration["eta"] in economic["turnover_etas"],
        "ROOT frozen stock/covariance integration choices required",
    )
    tag = integration["stock_recipe_id"]
    _require(
        integration["observations"] in risk["panels"]["return_observations"],
        "Stock risk integration requires a registered observation window",
    )
    integration_arm = Arm("selected_stock_risk", None, integration["recipe_id"], integration["eta"])
    for fee in economic["fees_bps"]:
        spec = {
            "kind": "stock_risk_integration",
            "stock": stock_specs[tag],
            "risk": risk_spec(
                {"kind": "stock_activity", "asset_count": 50},
                integration["observations"],
                integration_arm,
                economic["stock_cap"],
            ),
        }
        name = add(
            "integration_" + tag + "_fee" + format(fee, "g"),
            "integrations",
            spec,
            fee,
            refit="none"
            if stock_specs[tag]["candidate_kind"] == "control"
            else "monthly_expanding_mature",
        )
        equal = stock_arms[tag, fee]
        compare(
            "integration_scores_fee" + format(fee, "g"),
            "stock_prediction",
            [equal, name],
            same_score_bytes=True,
            outcome_arm_id=equal,
            minimum_support=stock_config["minimum_scored_cross_section"],
        )
        compare("integration_economics_fee" + format(fee, "g"), "portfolio", [equal, name])
    full_sources = dict(source_sha256)
    for path, raw in documents.items():
        _require(path not in full_sources, "Recipe source path collision")
        full_sources[path] = _sha(raw)
    # These are in-memory documents only, never written/reserved by this compiler.
    release = {
        "schema": "shared_quant_holdout_release_v1",
        "phase": "all_components_frozen_before_outcomes",
        "release_id": root_decision["release_id"],
        "approved_at": root_decision["approved_at"],
        "initial_signal": "2014-12-31",
        "terminal_close": "2017-11-10",
        "historical_information_cutoff": root_decision["historical_information_cutoff"],
        "cash_policy": "explicit_zero_interest_assumption",
        "terminal_policy": "mark_actual_positions_no_implicit_liquidation",
        "missing_mark_policy": "preserve_incomplete_valid_prefix_no_gap_repair",
        "first_interval": "include_initial_fee_and_first_open_to_close",
        "label_policy": "prior_signals_and_available_at_no_later_than_fit",
        "training_sample_weight": "equal_signal_date_mean_one",
        "initial_cash": economic["initial_cash"],
        "source_sha256": full_sources,
        "runtime": runtime,
        "calendar": calendar,
        "input_files": input_files,
        "development_evidence": development_evidence,
        "portfolio_arms": declarations,
        "forecast_arms": forecast_declarations,
        "comparisons": comparisons,
        "root_decision": root_decision,
        "descriptor_bindings": descriptors,
        "scope": "registered_evaluation_protocol_not_universal_performance_support",
        "reporting_policy": REPORTING_POLICY,
        "wider_parent": wider_binding,
    }
    return {
        "recipes": documents,
        "release": _raw(release),
        "counts": {
            "portfolio_arms": len(declarations),
            "forecast_only_arms": 10,
            "covariance_forecast_cells": 12,
            "risk_economic_paths": 208,
        },
    }
