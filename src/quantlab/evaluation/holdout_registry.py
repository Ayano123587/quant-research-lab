"""Metadata-authenticated concrete recipes; no recipe selection or data loading."""

from __future__ import annotations

import json
import math
import platform
import struct
from bisect import bisect_right
from datetime import date, datetime
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

from quantlab.evaluation.holdout import (
    END,
    START,
    Arm,
    _bound,
    _sha,
    calendar_window,
    file_sha,
    source_closure,
)
from quantlab.evaluation.holdout_adapters import StockPanelProvider


class ForecastSupportError(ValueError):
    """Only diagnosed real-data support failure, never authentication/schema drift."""

    CODES = {"insufficient_past_eligible_panel", "insufficient_mature_training_support"}

    def __init__(self, code, detail):
        if code not in self.CODES:
            raise ValueError("Unregistered data-support failure code")
        self.code = code
        super().__init__(detail)


class StockSignalMemo:
    """Share only an identically frozen stock signal, never economic state."""

    def __init__(self, provider, binding):
        from quantlab.evaluation.holdout_preparation import DeferredStockProvider

        if not isinstance(provider, DeferredStockProvider):
            raise ValueError("Memo requires the actual deferred stock provider")
        self.provider = provider
        self.binding = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.history_binding = None
        self.fits, self.targets = {}, {}
        self.latest_fit = None
        self.export_bindings = None

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def bind_history(self, loaded, window, *, source_loader, export_root):
        if self.history_binding is None:
            self.provider.bind_history(
                loaded, window, source_loader=source_loader, export_root=export_root
            )
            self.history_binding = (loaded, window, source_loader)
        else:
            prior, prior_window, prior_loader = self.history_binding
            if loaded is not prior or source_loader is not prior_loader or window != prior_window:
                raise ValueError("Shared stock history/source/window differs")

    def bind_forecast_exports(self, bindings):
        if self.export_bindings is not None and bindings is not self.export_bindings:
            raise ValueError("Shared stock all-arm export binding differs")
        self.provider.bind_forecast_exports(bindings)
        self.export_bindings = bindings

    def fit(self, view, selected):
        from quantlab.evaluation.holdout import FitReceipt

        key = (view.session, view.cutoff)
        if key not in self.fits:
            if self.latest_fit is not None and key < self.latest_fit:
                raise ValueError("Shared stock refits must remain chronological")
            receipt = self.provider.fit(view, selected)
            if (
                not isinstance(receipt, FitReceipt)
                or receipt.cutoff != view.cutoff
                or receipt.used_keys != selected.keys
                or receipt.maximum_feature_available_at > view.cutoff
            ):
                raise ValueError("Shared stock fit violated its approved mature keys")
            self.fits[key] = (receipt, selected)
            self.latest_fit = key
        receipt, original = self.fits[key]
        if (
            original != selected
            or receipt.used_keys != selected.keys
            or receipt.cutoff != view.cutoff
            or receipt.maximum_feature_available_at > view.cutoff
        ):
            raise ValueError("Shared stock cached mature selection differs")
        return receipt

    def decide(self, view, state):
        if state is not None:
            raise ValueError("Shared stock memo accepts pure signals only")
        key = (view.session, view.cutoff)
        if key not in self.targets:
            if self.latest_fit is not None and key < self.latest_fit:
                raise ValueError("Cannot predict an older signal using a later fitted model")
            self.targets[key] = self.provider.decide(view, None)
        return self.targets[key]


def _stock_signal_provider(cache, spec, *, refit_policy, history, identity, release_sha, **context):
    binding = {
        "stock_spec": spec,
        "refit_policy": refit_policy,
        "history": history,
        "source_identity": identity.as_dict(),
        "release_sha256": release_sha,
        "runtime": context["release_context"][0]["runtime"],
    }
    key = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if key not in cache:
        cache[key] = StockSignalMemo(construct_stock_provider(spec, **context), binding)
    return cache[key]


class RegisteredProvider:
    """Source-bound gate callbacks with the gate's declared loader signature."""

    def __init__(self, input_id, loader, decision_provider, *, arm_id, schedule, refit_policy):
        self.input_id, self.loader, self.provider = input_id, loader, decision_provider
        self.loaded = None
        self.arm_id, self.schedule, self.refit_policy = arm_id, schedule, refit_policy
        self.export_root = None
        self.targets, self.fit_receipts, self.stream_exports, self.stream_status = {}, {}, {}, {}

    def load(self, input_id, window):
        if input_id != self.input_id:
            raise ValueError("Registered loader input differs")
        if self.export_root is None:
            raise ValueError("Exclusive durable stream export location required before loading")
        self.loaded = self.loader.load()
        if hasattr(self.provider, "bind_history"):
            self.provider.bind_history(
                self.loaded, window, source_loader=self.loader.loader, export_root=self.export_root
            )
        return self.loaded

    def initialize_stream(self, window):
        if self.export_root is None:
            raise ValueError("Exclusive durable stream export location required")
        Path(self.export_root).mkdir(parents=True, exist_ok=False)
        self.window = window

    def step_signal(self, day):
        from quantlab.evaluation.holdout import FitReceipt, mature_training_keys
        from quantlab.evaluation.holdout_pipeline import _write

        if day not in self.window.rebalance_signals or day in self.stream_status:
            raise ValueError("Exactly one forecast attempt per registered signal required")
        view = self.loader.view(day, self.schedule)
        window = self.window
        if hasattr(self.provider, "prepare_features"):
            self.provider.prepare_features(view)
        try:
            due = self.refit_policy == "monthly_expanding_mature" or (
                self.refit_policy == "initial_only_mature" and day == window.initial_signal
            )
            if due:
                selected = mature_training_keys(
                    self.provider.clocks(view), signal_session=day, decision_at=view.cutoff
                )
                stock = getattr(self.provider, "stock", self.provider)
                if (
                    len(selected.keys) < stock.minimum_rows
                    or len({d for d, _ in selected.keys}) < stock.minimum_dates
                ):
                    raise ForecastSupportError(
                        "insufficient_mature_training_support",
                        "Verified mature feature/label join falls below frozen training support",
                    )
                receipt = self.provider.fit(view, selected)
                if (
                    not isinstance(receipt, FitReceipt)
                    or receipt.cutoff != view.cutoff
                    or receipt.used_keys != selected.keys
                    or receipt.maximum_feature_available_at > view.cutoff
                ):
                    raise ValueError("Independent refit violated its historical information set")
                self.fit_receipts[day] = receipt
            if hasattr(self.provider, "prepare_forecast"):
                self.provider.prepare_forecast(view)
            else:
                self.targets[day] = self.provider.decide(view, None)
            record = {
                "session": day,
                "status": "forecast_exported",
                "target": self.targets.get(day),
                "fit": self.fit_receipts.get(day),
            }
            forecast = getattr(self.provider, "forecast_provider", None)
            if forecast is not None and day in forecast.forecasts:
                record.update(reference=forecast.references[day], forecasts=forecast.forecasts[day])
            stock = getattr(self.provider, "stock", self.provider)
            if hasattr(stock, "predictions"):
                record["predictions"] = tuple(r for r in stock.predictions if r["session"] == day)
            self.stream_status[day] = "forecast_exported"
        except ForecastSupportError as error:
            # Record every attempted date, without a stale-model/cash fallback.
            self.stream_status[day] = "unsupported_forecast"
            record = {
                "session": day,
                "status": "unsupported_forecast",
                "reason": str(error),
                "support_code": error.code,
            }
            stock = getattr(self.provider, "stock", self.provider)
            if hasattr(stock, "predictions"):
                record["predictions"] = tuple(r for r in stock.predictions if r["session"] == day)
        path = Path(self.export_root) / (day.isoformat() + ".json")
        self.stream_exports[day] = _write(path, record)

    def finalize_stream(self):
        if set(self.stream_status) != set(self.window.rebalance_signals):
            raise ValueError("Every frozen forecast date must be attempted")
        for item in self.stream_exports.values():
            if file_sha(item["path"]) != item["sha256"]:
                raise ValueError("Independent forecast export changed before economics")

    def prepare_stream(self, loaded, window):
        """Single-arm development engineering; actual releases use date-outer coordination."""
        self.initialize_stream(window)
        for day in window.rebalance_signals:
            self.step_signal(day)
        self.finalize_stream()

    def decide(self, view, state):
        from quantlab.backtest.engine import Decision

        if view.session not in self.stream_status:
            return Decision("HOLD", None, view.cutoff, view.next_open_at, "between_frozen_signals")
        if self.stream_status[view.session] != "forecast_exported":
            raise ValueError("Dated forecast unsupported; no stale-target fallback")
        export = self.stream_exports[view.session]
        if file_sha(export["path"]) != export["sha256"]:
            raise ValueError("Dated forecast export changed before economic execution")
        if hasattr(self.provider, "prepare_forecast"):
            # State-dependent solver/turnover uses ONLY this path's actual holdings.
            return self.provider.decide(view, state)
        return self.targets[view.session]

    def clocks(self, view):
        return self.provider.clocks(view)

    def fit(self, view, selected):
        receipt = self.fit_receipts[view.session]
        if receipt.used_keys != selected.keys or receipt.cutoff != view.cutoff:
            raise ValueError("Economic refit receipt differs from independent dated stream")
        return receipt


class SharedHistory:
    """Share immutable authenticated real bars, never fitted models or holdings."""

    def __init__(self, loader):
        self.loader, self.value = loader, None
        self.indices, self.current_view = None, None

    def load(self):
        from quantlab.evaluation.holdout import LoadedHistory

        if self.value is None:
            loaded = self.loader.load()
            self.value = LoadedHistory(
                MappingProxyType(dict(loaded.histories)),
                loaded.source_identity,
                loaded.source_empty_assets,
                loaded.source_empty_evidence,
            )
        return self.value

    def view(self, day, schedule):
        from quantlab.backtest.engine import HistoricalView

        loaded = self.load()
        if self.indices is None:
            self.indices = {
                a: tuple(b.session for b in rows) for a, rows in loaded.histories.items()
            }
        if self.current_view is None or self.current_view.session != day:
            self.current_view = HistoricalView(
                tuple(
                    (a, rows[: bisect_right(self.indices[a], day)])
                    for a, rows in loaded.histories.items()
                ),
                day,
                schedule.session(day).decision_at,
                schedule.next_session(day).open_at,
            )
        if self.current_view.cutoff != schedule.session(day).decision_at:
            raise ValueError("Shared source view calendar differs")
        return self.current_view


def prepare_registered_streams(registry, histories, window):
    """All arms at date t export before any new mature labels at date t+1."""
    wrappers = []
    for name, arm in registry.items():
        wrapper = getattr(arm.load, "__self__", None)
        if not isinstance(wrapper, RegisteredProvider) or wrapper.arm_id != name:
            raise ValueError("Concrete source-bound forecast stream provider required")
        if wrapper.loaded is not histories[name]:
            raise ValueError("Independent stream must use the actual loaded source")
        wrapper.initialize_stream(window)
        wrappers.append(wrapper)
    bindings = {wrapper.arm_id: wrapper.stream_exports for wrapper in wrappers}
    for wrapper in wrappers:
        if hasattr(wrapper.provider, "bind_forecast_exports"):
            wrapper.provider.bind_forecast_exports(bindings)
    for day in window.rebalance_signals:
        for wrapper in wrappers:
            wrapper.step_signal(day)
    for wrapper in wrappers:
        wrapper.finalize_stream()


class CausalRiskProvider:
    """Existing risk universe ordering, estimated from the restricted view only."""

    def __init__(
        self,
        forecast_provider,
        signals,
        panel,
        economic_arm,
        cap,
        solver_policy,
        projection_path,
        projection_sha,
        sample_id,
    ):
        self.forecast_provider, self.signals = forecast_provider, frozenset(signals)
        self.panel, self.arm, self.cap = panel, economic_arm, cap
        self.solver_policy = solver_policy
        self.projection_path, self.projection_sha = projection_path, projection_sha
        self.sample_id, self.decisions, self.panel_support = sample_id, [], []

    def prepare_forecast(self, view):
        import hashlib

        from quantlab.allocation.policies import ASSET_IDS
        from quantlab.risk.scoring import ReferencePortfolio
        from quantlab.risk.study import asset_window, select_nested

        if view.session not in self.signals:
            raise ValueError("Forecast outside the frozen signal calendar")
        schedule = self.forecast_provider.schedule
        end = schedule.sessions.index(view.session)
        days = frozenset(schedule.sessions[end - 252 : end + 1])
        histories = dict(view.histories)
        candidates, refused = [], {}
        population = tuple(histories) if self.panel["kind"] == "stock_activity" else ASSET_IDS
        if self.panel["kind"] == "stock_activity" and len(population) != 7195:
            raise ValueError("Risk stock panel must start from all original7195 histories")
        for asset in population:
            rows = tuple(b for b in histories[asset] if b.session in days)
            candidate, reasons = asset_window(asset, rows, schedule, signal_session=view.session)
            if candidate is None:
                refused[asset] = reasons
            else:
                candidates.append(candidate)
        if self.panel["kind"] == "stock_activity":
            count = self.panel["asset_count"]
            if count not in (50, 100, 150):
                raise ValueError("Registered nested stock panel size required")
            selected = select_nested(candidates, sizes=(count,))[count]
        elif self.panel["kind"] == "etf8_fixed":
            selected_map = {c.asset_id: c for c in candidates}
            selected = (
                tuple(selected_map[a] for a in ASSET_IDS)
                if set(selected_map) == set(ASSET_IDS)
                else ()
            )
        else:
            raise ValueError("Unknown causal risk panel")
        self.panel_support.append(
            {
                "session": view.session,
                "refused": refused,
                "selected": tuple(c.asset_id for c in selected),
            }
        )
        if not selected:
            raise ForecastSupportError(
                "insufficient_past_eligible_panel",
                "Full original population verified but frozen past-only risk panel lacks support",
            )
        ids = tuple(c.asset_id for c in selected)
        input_id = hashlib.sha256(
            json.dumps([c.input_sha256 for c in selected]).encode()
        ).hexdigest()
        reference = ReferencePortfolio(
            ids,
            (1 / len(ids),) * len(ids),
            view.cutoff,
            view.cutoff,
            "past_panel_equal_q_common_estimators",
            input_id,
        )
        estimates = self.forecast_provider.forecast(view, reference)
        return estimates

    def decide(self, view, state):
        from quantlab.risk.economic_study import decide

        estimates = self.forecast_provider.forecasts[view.session]
        value, audit = decide(
            view,
            state,
            self.arm,
            estimates[self.arm.recipe_id],
            estimates[self.sample_id],
            self.cap,
            self.solver_policy,
            self.projection_path,
            self.projection_sha,
        )
        self.decisions.append({"session": view.session, "decision": value, "audit": audit})
        return value


class WiderStockProvider(StockPanelProvider):
    """Prospective projection bridge; original cohorts/clocks/bytes stay exact."""

    def __init__(self, *args, projection_arm_id, expected_columns, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm_id = projection_arm_id
        self.expected_columns = tuple(expected_columns)

    def _features(self, day):
        from dataclasses import replace

        from quantlab.stocks.development import FEATURE_IDS
        from quantlab.stocks.wider_study import project_rows

        base = super()._features(day)
        records = self._read(self.partitions["wider_features", day], None)
        indexed = {r["asset_id"]: r for r in records}
        if len(indexed) != len(records) or set(indexed) != {r.asset_id for r in base}:
            raise ValueError("Wider projection changed original complete eligible cohort")
        ordered = []
        for row in base:
            wide = indexed[row.asset_id]
            if (
                wide["signal_session"] != day
                or wide["signal_at"] != row.signal_at
                or wide["max_input_available_at"] > row.signal_at
                or wide["intended_execution_at"] != row.intended_execution_at
                or wide["input_rows_sha256"] != row.input_rows_sha256
                or any(
                    struct.pack("<d", wide["rank_" + f]) != struct.pack("<d", v)
                    for f, v in zip(FEATURE_IDS, row.values, strict=True)
                )
                or any(
                    struct.pack("<d", wide["raw_" + f]) != struct.pack("<d", v)
                    for f, v in zip(FEATURE_IDS, row.raw_values, strict=True)
                )
            ):
                raise ValueError("Wider original six/clocks/source inputs changed")
            ordered.append(wide)
        columns, values = project_rows(ordered, self.arm_id)
        if columns != self.expected_columns:
            raise ValueError("Actual wider model columns differ from the frozen recipe")
        return tuple(replace(row, values=value) for row, value in zip(base, values, strict=True))


class StockRiskProvider:
    """Same learned scores/selected IDs; change only their portfolio weights."""

    def __init__(
        self,
        stock,
        forecasts,
        economic_arm,
        cap,
        solver_policy,
        projection_path,
        projection_sha,
        sample_id,
    ):
        self.stock, self.forecast_provider, self.arm = stock, forecasts, economic_arm
        self.cap, self.solver_policy = cap, solver_policy
        self.projection_path, self.projection_sha, self.sample_id = (
            projection_path,
            projection_sha,
            sample_id,
        )
        self.decisions = []
        self.equal_targets = {}

    def clocks(self, view):
        return self.stock.clocks(view)

    def bind_history(self, loaded, window, **context):
        if hasattr(self.stock, "bind_history"):
            self.stock.bind_history(loaded, window, **context)

    def bind_forecast_exports(self, bindings):
        if hasattr(self.stock, "bind_forecast_exports"):
            self.stock.bind_forecast_exports(bindings)

    def prepare_features(self, view):
        if hasattr(self.stock, "prepare_features"):
            self.stock.prepare_features(view)

    def fit(self, view, selected):
        return self.stock.fit(view, selected)

    def export_predictions(self, path):
        return self.stock.export_predictions(path)

    def prepare_forecast(self, view):
        import hashlib

        from quantlab.risk.scoring import ReferencePortfolio

        equal = self.stock.decide(view, None)
        self.equal_targets[view.session] = equal
        if equal.action == "HOLD":
            return equal
        ids, weights = zip(*equal.target, strict=True)
        reference = ReferencePortfolio(
            tuple(ids),
            tuple(weights),
            view.cutoff,
            view.cutoff,
            "same_frozen_stock_scores_and_topN_ids",
            hashlib.sha256(repr(self.stock.predictions[-1]).encode()).hexdigest(),
        )
        estimates = self.forecast_provider.forecast(view, reference)
        return estimates

    def decide(self, view, state):
        from quantlab.risk.economic_study import decide

        equal = self.equal_targets[view.session]
        if equal.action == "HOLD":
            return equal
        estimates = self.forecast_provider.forecasts[view.session]
        value, audit = decide(
            view,
            state,
            self.arm,
            estimates[self.arm.recipe_id],
            estimates[self.sample_id],
            self.cap,
            self.solver_policy,
            self.projection_path,
            self.projection_sha,
        )
        self.decisions.append(
            {
                "session": view.session,
                "equal_weight_target": equal,
                "risk_weight_target": value,
                "audit": audit,
            }
        )
        return value


def _directory(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
        or not path.is_dir()
        or not path.resolve().is_relative_to(root)
        or any(p.is_symlink() for p in path.parents if p != root.parent)
    ):
        raise ValueError("Authenticated contained input directory required")
    return path


def authenticate_factory_inputs(release_path, expected_sha256, *, project, inputs, schedule):
    """Only hashes/metadata before provider construction; gate repeats full checks."""
    if file_sha(release_path) != _sha(expected_sha256):
        raise ValueError("Actual external release SHA required")
    freeze = json.loads(Path(release_path).read_bytes())
    if (
        freeze.get("schema") != "shared_quant_holdout_release_v1"
        or freeze.get("phase") != "all_components_frozen_before_outcomes"
        or freeze.get("initial_signal") != START.isoformat()
        or freeze.get("terminal_close") != END.isoformat()
    ):
        raise ValueError("Development artifacts cannot construct a final registry")
    runtime = freeze["runtime"]
    if runtime != {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {p: version(p) for p in runtime["packages"]},
    }:
        raise ValueError("Actual frozen factory runtime differs")
    sources = freeze["source_sha256"]
    for name, digest in sources.items():
        if file_sha(_bound(project, name)) != _sha(digest):
            raise ValueError("Actual frozen factory source differs")
    own = Path(__file__).resolve().relative_to(Path(project).resolve()).as_posix()
    if sources.get(own) != file_sha(__file__):
        raise ValueError("Registry factory must itself be externally source-bound")
    for name, digest in source_closure(project, [own]).items():
        if sources.get(name) != digest:
            raise ValueError("Transitive factory source missing from actual freeze")
    for item in freeze["input_files"].values():
        path = _bound(inputs, item["path"])
        if path.stat().st_size != item["bytes"] or file_sha(path) != _sha(item["sha256"]):
            raise ValueError("Factory input differs before any provider/native read")
    if freeze["historical_information_cutoff"] != schedule.session(START).decision_at.isoformat():
        raise ValueError("Factory historical cutoff differs")
    return freeze


def construct_stock_provider(spec, *, inputs, schedule, signals, release_context=None):
    """Original-six and registered wider projections share unchanged cohorts."""
    from quantlab.stocks.development import FEATURE_IDS, Candidate

    _directory(inputs, spec["panel_root"])
    if "final_preparation" in spec:
        from quantlab.evaluation.holdout_preparation import DeferredStockProvider

        if release_context is None:
            raise ValueError("Actual frozen release context required for final preparation")
        release, release_sha = release_context
        return DeferredStockProvider(
            spec,
            inputs=inputs,
            schedule=schedule,
            signals=signals,
            release=release,
            release_sha256=release_sha,
        )
    if tuple(spec["feature_columns"]) != tuple("rank_" + f for f in FEATURE_IDS):
        raise ValueError("Base panel must retain the exact original six rank columns")
    if tuple(spec["raw_columns"]) != tuple("raw_" + f for f in FEATURE_IDS):
        raise ValueError("Base panel must retain the exact original six raw columns")
    projection = spec["projection"]
    extra = {}
    if projection["kind"] == "original_six":
        cls = StockPanelProvider
    elif projection["kind"] == "registered_wider":
        if spec["candidate_kind"] == "control":
            raise ValueError("Original-six fixed controls cannot be relabelled wider models")
        from quantlab.stocks.wider_study import project_rows

        model_columns = projection["model_columns"]
        if type(model_columns) is not int or model_columns < 1:
            raise ValueError("Frozen wider model column count must be a positive integer")
        columns = project_rows((), projection["arm_id"])[0]
        if len(columns) != model_columns:
            raise ValueError("Actual wider column IDs differ from the frozen model column count")
        cls = WiderStockProvider
        extra = {
            "projection_arm_id": projection["arm_id"],
            "expected_columns": columns,
        }
    else:
        raise ValueError("No guessed feature projection")
    candidate = Candidate(
        spec["candidate_id"],
        spec["candidate_kind"],
        tuple(sorted(spec["candidate_parameters"].items())),
    )
    provider = cls(
        _directory(inputs, spec["panel_root"]),
        spec["partitions"],
        schedule,
        signals,
        candidate=candidate,
        feature_columns=spec["feature_columns"],
        raw_columns=spec["raw_columns"],
        target_policy=spec["target_policy"],
        threads=spec["threads"],
        target_spec=spec["target_spec"],
        minimum_rows=spec["minimum_rows"],
        minimum_dates=spec["minimum_dates"],
        **extra,
    )
    if projection["kind"] == "registered_wider":
        wider_root = _directory(inputs, spec["wider_panel_root"])
        for item in spec["wider_partitions"]:
            day = date.fromisoformat(item["session"])
            path = _bound(wider_root, item["path"])
            if path.stat().st_size != item["bytes"] or file_sha(path) != _sha(item["sha256"]):
                raise ValueError("Existing original wider partition differs")
            if ("wider_features", day) in provider.partitions:
                raise ValueError("Duplicate existing wider partition")
            provider.partitions["wider_features", day] = path
            provider.partition_identity[path] = (item["bytes"], item["sha256"])
    return provider


def construct_risk_provider(spec, *, freeze, inputs, schedule, identity, signals, stock=None):
    from quantlab.evaluation.holdout_adapters import CovarianceForecastProvider
    from quantlab.risk.economic_study import Arm as EconomicArm
    from quantlab.risk.optimization import SolverPolicy

    forecasts = CovarianceForecastProvider(
        schedule,
        identity,
        observations=spec["observations"],
        estimator_parameters=spec["estimator_parameters"],
    )
    if spec["kind"] == "risk_forecast":
        if stock is not None:
            raise ValueError("Forecast-only registers cannot allocate a stock portfolio")
        return CausalRiskProvider(
            forecasts, signals, spec["panel"], None, None, None, None, None, "sample_ddof1"
        )
    policy = dict(spec["solver_policy"])
    policy["expected_versions"] = tuple(
        (name, value) for name, value in policy["expected_versions"]
    )
    policy["frozen_at"] = datetime.fromisoformat(policy["frozen_at"])
    solver = SolverPolicy(**policy)
    projection = freeze["input_files"][spec["projection_input_id"]]
    common = (
        EconomicArm(**spec["economic_arm"]),
        spec["cap"],
        solver,
        _bound(inputs, projection["path"]),
        projection["sha256"],
        spec["sample_reference_id"],
    )
    if stock is not None:
        return StockRiskProvider(stock, forecasts, *common)
    return CausalRiskProvider(forecasts, signals, spec["panel"], *common)


def build_registry(release_path, expected_sha256, *, project, inputs, schedule):
    """Build providers from actual externally frozen recipe files, without defaults."""
    freeze = authenticate_factory_inputs(
        release_path, expected_sha256, project=project, inputs=inputs, schedule=schedule
    )
    policies = {
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
    for comparison in freeze["comparisons"]:
        if (
            comparison["kind"] not in policies
            or comparison["analysis_policy"] != policies[comparison["kind"]]
        ):
            raise ValueError("Fixed aggregation and uncertainty policy required before release")
    from quantlab.allocation.sensitivity import SCENARIOS
    from quantlab.data.provenance import SourceIdentity
    from quantlab.evaluation.holdout_adapters import (
        ArchiveProvider,
        ETFProvider,
        SensitivityProvider,
        SnapshotProvider,
        YahooETFProvider,
    )

    window = calendar_window(schedule, START, END)
    registry, providers, histories, stock_memos = {}, {}, {}, {}
    forecast_ids = {r["arm_id"] for r in freeze.get("forecast_arms", [])}
    for declaration in freeze["portfolio_arms"] + freeze.get("forecast_arms", []):
        name = declaration["arm_id"]
        if name in registry:
            raise ValueError("Duplicate frozen arm")
        recipe_path = _bound(project, declaration["recipe_path"])
        if file_sha(recipe_path) != _sha(declaration["recipe_sha256"]):
            raise ValueError("Actual recipe bytes differ")
        recipe = json.loads(recipe_path.read_bytes())
        if recipe["schema"] != "shared_quant_concrete_recipe_v1":
            raise ValueError("Explicit concrete recipe schema required")
        for field in (
            "arm_id",
            "component",
            "fee_bps",
            "refit_policy",
            "source_identity",
            "input_id",
        ):
            if recipe[field] != declaration[field]:
                raise ValueError("Concrete recipe/declaration differs: " + field)
        if (
            recipe["signal_policy"] != "initial_then_actual_month_end"
            or recipe["support_policy"] != "preserve_incomplete_no_fill_no_name_substitution"
            or recipe["initial_signal"] != START.isoformat()
            or recipe["terminal_close"] != END.isoformat()
            or (
                name not in forecast_ids
                and (not math.isfinite(recipe["fee_bps"]) or recipe["fee_bps"] < 0)
            )
        ):
            raise ValueError("Explicit dates, support and costs required")
        evidence = freeze["development_evidence"][recipe["component"]]
        if (
            recipe["development_evidence_sha256"] != evidence["sha256"]
            or recipe["development_economic_support"] != evidence["economic_support"]
        ):
            raise ValueError("Recipe must preserve reviewed development support classification")
        identity = SourceIdentity(**recipe["source_identity"])
        history = recipe["history"]
        binding = freeze["input_files"][recipe["input_id"]]
        if history["kind"] == "original_archive":
            if (
                history["path"] != binding["path"]
                or history["sha256"] != binding["sha256"]
                or identity.archive_sha256 != history["sha256"]
            ):
                raise ValueError("Archive loader/source identity must match actual input")
            assets = tuple(tuple(pair) for pair in history["assets"])
            if history["population"] == "all_original_7195_stocks":
                if len(assets) != 7195 or any(kind != "stock" for kind, _ in assets):
                    raise ValueError("Full original stock population required")
            elif history["population"] != "fixed_prescribed_etf_collection":
                raise ValueError("Explicit original source population required")
            loader = ArchiveProvider(
                _bound(inputs, history["path"]), history["sha256"], assets, identity
            )
        elif history["kind"] == "yahoo_etf_release":
            root = _directory(inputs, history["root"])
            if (
                root / "manifest.json" != _bound(inputs, binding["path"])
                or binding["sha256"] != history["manifest_sha256"]
                or identity.manifest_sha256 != history["manifest_sha256"]
            ):
                raise ValueError("Actual Yahoo manifest/source binding differs")
            loader = YahooETFProvider(root, history["manifest_sha256"], schedule)
        elif history["kind"] == "market_snapshot":
            root = _directory(inputs, history["root"])
            if (
                root / "manifest.json" != _bound(inputs, binding["path"])
                or binding["sha256"] != history["manifest_sha256"]
                or identity.manifest_sha256 != history["manifest_sha256"]
            ):
                raise ValueError("Actual BIN manifest/source binding differs")
            loader = SnapshotProvider(
                root,
                history["manifest_sha256"],
                tuple(tuple(p) for p in history["assets"]),
                schedule,
                identity,
            )
        else:
            raise ValueError("Unsupported concrete historical source")
        spec = recipe["decision"]
        if (name in forecast_ids) != (spec["kind"] == "risk_forecast"):
            raise ValueError("Explicit forecast-only role must match the actual recipe")
        if name in forecast_ids and (
            recipe["fee_bps"] is not None or recipe["refit_policy"] != "none"
        ):
            raise ValueError("Forecast-only recipes cannot request economics or supervised fits")
        if spec["kind"] == "etf_control":
            provider = ETFProvider(
                schedule,
                identity,
                window.rebalance_signals,
                policy=spec["policy"],
                covariance_method=spec["covariance_method"],
            )
        elif spec["kind"] == "etf_sensitivity":
            (scenario,) = (s for s in SCENARIOS if s.scenario_id == spec["scenario_id"])
            provider = SensitivityProvider(
                schedule, identity, window.rebalance_signals, scenario=scenario
            )
        elif spec["kind"] == "stock":
            if "final_preparation" not in spec:
                raise ValueError(
                    "Released final stock preparation is required, not guessed final partitions"
                )
            provider = _stock_signal_provider(
                stock_memos,
                spec,
                refit_policy=recipe["refit_policy"],
                history=history,
                identity=identity,
                release_sha=expected_sha256,
                inputs=inputs,
                schedule=schedule,
                signals=window.rebalance_signals,
                release_context=(freeze, expected_sha256),
            )
        elif spec["kind"] in {"risk_portfolio", "risk_forecast"}:
            provider = construct_risk_provider(
                spec,
                freeze=freeze,
                inputs=inputs,
                schedule=schedule,
                identity=identity,
                signals=window.rebalance_signals,
            )
        elif spec["kind"] == "stock_risk_integration":
            if "final_preparation" not in spec["stock"]:
                raise ValueError("Released final stock preparation is required for integration")
            stock = _stock_signal_provider(
                stock_memos,
                spec["stock"],
                refit_policy=recipe["refit_policy"],
                history=history,
                identity=identity,
                release_sha=expected_sha256,
                inputs=inputs,
                schedule=schedule,
                signals=window.rebalance_signals,
                release_context=(freeze, expected_sha256),
            )
            provider = construct_risk_provider(
                spec["risk"],
                freeze=freeze,
                inputs=inputs,
                schedule=schedule,
                identity=identity,
                signals=window.rebalance_signals,
                stock=stock,
            )
        else:
            raise ValueError("Unsupported recipe; no placeholder provider can complete")
        cache_key = json.dumps({"history": history, "identity": identity.as_dict()}, sort_keys=True)
        if cache_key not in histories:
            histories[cache_key] = SharedHistory(loader)
        wrapper = RegisteredProvider(
            recipe["input_id"],
            histories[cache_key],
            provider,
            arm_id=name,
            schedule=schedule,
            refit_policy=recipe["refit_policy"],
        )
        refit = recipe["refit_policy"] != "none"
        if refit and spec["kind"] not in {"stock", "stock_risk_integration"}:
            raise ValueError("Only a concrete stock learner may request supervised refits")
        registry[name] = Arm(
            name,
            recipe["component"],
            declaration["recipe_sha256"],
            wrapper.load,
            wrapper.decide,
            wrapper.clocks if refit else None,
            wrapper.fit if refit else None,
        )
        providers[name] = wrapper
    return freeze, registry, providers
