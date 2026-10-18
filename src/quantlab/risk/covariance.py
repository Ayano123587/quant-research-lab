"""Daily covariance estimates with explicit asset order and input availability.

These estimators consume complete, aligned decimal simple returns. They never
fill missing returns, annualise a matrix, or add an undocumented diagonal ridge.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

import numpy as np

UNITS = "decimal_return_squared_per_session"


def _aware(value):
    return (
        isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None
    )


@dataclass(frozen=True)
class ReturnWindow:
    values: np.ndarray
    asset_ids: tuple[str, ...]
    interval_start: tuple[datetime, ...]
    interval_end: tuple[datetime, ...]
    available_at: tuple[datetime, ...]
    signal_at: datetime
    source_snapshot_id: str
    return_convention: str
    eligibility_rule_id: str
    coverage_report_id: str
    source_provenance: Mapping[str, str]
    input_units: str = "decimal_simple_return_per_session"

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64, order="C", copy=True)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("At least two observations and one asset required")
        t, p = values.shape
        if not np.isfinite(values).all() or np.any(values <= -1):
            raise ValueError("Complete finite positive-price decimal simple returns required")
        if len(self.asset_ids) != p or len(set(self.asset_ids)) != p or not all(self.asset_ids):
            raise ValueError("Unique ordered asset IDs must match columns")
        if self.input_units != "decimal_simple_return_per_session":
            raise ValueError("Explicit daily decimal simple returns required")
        if not _aware(self.signal_at):
            raise ValueError("Timezone-aware decision cutoff required")
        for name in ("interval_start", "interval_end", "available_at"):
            times = tuple(getattr(self, name))
            if len(times) != t or not all(_aware(x) for x in times):
                raise ValueError("One timezone-aware timestamp per observation required")
            object.__setattr__(self, name, times)
        for i, (start, end, available) in enumerate(
            zip(self.interval_start, self.interval_end, self.available_at, strict=True)
        ):
            if not start < end <= available <= self.signal_at:
                raise ValueError("Input interval/availability violates decision cutoff")
            if i and start != self.interval_end[i - 1]:
                raise ValueError("Consecutive aligned intervals required; no gap collapse")
        if not all(
            (
                self.source_snapshot_id,
                self.return_convention,
                self.eligibility_rule_id,
                self.coverage_report_id,
                self.source_provenance,
            )
        ):
            raise ValueError("Source, convention, eligibility and coverage provenance required")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "asset_ids", tuple(self.asset_ids))
        object.__setattr__(
            self, "source_provenance", MappingProxyType(dict(self.source_provenance))
        )

    @property
    def window_hash(self):
        metadata = {
            "asset_ids": self.asset_ids,
            "starts": [x.isoformat() for x in self.interval_start],
            "ends": [x.isoformat() for x in self.interval_end],
            "available": [x.isoformat() for x in self.available_at],
            "signal": self.signal_at.isoformat(),
            "source": self.source_snapshot_id,
            "convention": self.return_convention,
            "eligibility": self.eligibility_rule_id,
            "coverage": self.coverage_report_id,
            "source_provenance": dict(self.source_provenance),
            "units": self.input_units,
            "shape": self.values.shape,
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(self.values.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: np.ndarray
    location: np.ndarray
    asset_ids: tuple[str, ...]
    method: str
    parameters: dict
    diagnostics: dict
    window_hash: str
    signal_at: datetime
    training_start: datetime
    training_end: datetime
    max_available_at: datetime
    fitted_at: datetime
    runtime: dict
    configuration_hash: str
    source_provenance: dict
    units: str = UNITS
    annualisation_factor: int = 252
    status: str = "valid"

    def portfolio_variance(self, weights, asset_ids: tuple[str, ...]):
        if tuple(asset_ids) != self.asset_ids:
            raise ValueError("Asset order mismatch; explicit recorded reindex required")
        if self.status != "valid":
            raise ValueError("Invalid covariance cannot be consumed")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(self.asset_ids),) or not np.isfinite(weights).all():
            raise ValueError("Finite weights with exact asset count required")
        return float(weights @ self.matrix @ weights)


def estimate_covariance(
    window: ReturnWindow,
    method: str = "sample_ddof1",
    *,
    min_observations: int = 2,
    ewma_lambda: float = 2 ** (-1 / 63),
    n_factors: int = 1,
    residual_floor: float = 0.0,
    constant_correlation_intensity: float | None = None,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-15,
):
    """Fit only this window; default PCA residual floor is zero, without a ridge.

    EWMA is weighted-centered with denominator 1-sum(w²), not a zero-mean
    recursive RiskMetrics update. Constant-correlation shrinkage has a configured
    intensity; it does not estimate the optimal Ledoit--Wolf intensity.
    """
    x = window.values
    t, p = x.shape
    if not isinstance(min_observations, int) or min_observations < 2 or t < min_observations:
        raise ValueError("Minimum observation count not satisfied")
    if not (
        np.isfinite(relative_tolerance)
        and relative_tolerance >= 0
        and np.isfinite(absolute_tolerance)
        and absolute_tolerance >= 0
    ):
        raise ValueError("Finite nonnegative numerical tolerances required")
    location = x.mean(axis=0)
    centered = x - location
    sample = centered.T @ centered / (t - 1)
    params = {
        "min_observations": min_observations,
        "centering": "sample_mean",
        "denominator": t - 1,
        "repair": "none",
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }
    effective = float(t)
    regularisation_amount = 0.0
    if method in ("sample_ddof1", "sample_mle"):
        matrix = sample if method == "sample_ddof1" else centered.T @ centered / t
        params["denominator"] = t - 1 if method == "sample_ddof1" else t
    elif method == "ewma":
        if not np.isfinite(ewma_lambda) or not 0 < ewma_lambda <= 1:
            raise ValueError("EWMA lambda must lie in (0,1]")
        weights = ewma_lambda ** np.arange(t - 1, -1, -1, dtype=float)
        weights /= weights.sum()
        denominator = float(1 - weights @ weights)
        if denominator <= np.finfo(float).eps:
            raise ValueError("Insufficient effective observations")
        location = weights @ x
        centered = x - location
        matrix = (centered.T * weights) @ centered / denominator
        effective = float(1 / (weights @ weights))
        params.update(
            lambda_=ewma_lambda,
            centering="weighted_mean",
            denominator=denominator,
            effective_count_definition="weight_concentration",
        )
    elif method in ("ledoit_wolf", "oas"):
        from sklearn.covariance import OAS, LedoitWolf

        estimator = (LedoitWolf if method == "ledoit_wolf" else OAS)(
            assume_centered=False, store_precision=False
        )
        estimator.fit(x)
        matrix, location = estimator.covariance_, estimator.location_
        params.update(
            denominator=t,
            shrinkage=float(estimator.shrinkage_),
            target="scaled_identity",
            target_variance=float(np.trace(sample) * (t - 1) / t / p),
            assume_centered=False,
            store_precision=False,
            implementation="sklearn.covariance." + type(estimator).__name__,
        )
    elif method == "pca_factor":
        if not isinstance(n_factors, int) or not 1 <= n_factors <= min(p, t - 1):
            raise ValueError("Factor count outside available sample dimension")
        if not np.isfinite(residual_floor) or residual_floor < 0:
            raise ValueError("Finite nonnegative daily residual variance floor required")
        eigenvalues, vectors = np.linalg.eigh(sample)
        tolerance = max(absolute_tolerance, relative_tolerance * np.linalg.norm(sample, ord=2))
        if n_factors > int(np.count_nonzero(eigenvalues > tolerance)):
            raise ValueError("Factor count exceeds numerical training-window rank")
        factor = (vectors[:, -n_factors:] * eigenvalues[-n_factors:]) @ vectors[:, -n_factors:].T
        residual = np.diag(sample - factor)
        if np.min(residual) < -tolerance:
            raise ValueError("Material negative PCA residual; no automatic repair")
        floored = np.maximum(residual, residual_floor)
        regularisation_amount = float(np.linalg.norm(floored - residual))
        matrix = factor + np.diag(floored)
        total = float(eigenvalues.sum())
        params.update(
            n_factors=n_factors,
            residual_floor=residual_floor,
            floored_residuals=int(np.count_nonzero(residual < residual_floor)),
            residual_variances=floored.tolist(),
            factor_eigenvalues=eigenvalues[-n_factors:].tolist(),
            explained_variance=float(eigenvalues[-n_factors:].sum() / total) if total > 0 else None,
            repair="explicit_residual_floor"
            if residual_floor > 0
            else "roundoff_residual_clip_at_zero",
        )
    elif method == "constant_correlation":
        delta = constant_correlation_intensity
        if delta is None or not np.isfinite(delta) or not 0 <= delta <= 1:
            raise ValueError("Explicit fixed shrinkage intensity in [0,1] required")
        scales = np.sqrt(np.diag(sample))
        if np.any(scales <= 0) or p < 2:
            raise ValueError("Constant correlation requires two nonconstant assets")
        correlation = sample / np.outer(scales, scales)
        # A PSD correlation matrix has mean off-diagonal correlation in
        # [-1/(p-1), 1]. The equicorrelation target therefore has nonnegative
        # eigenvalues 1-r (p-1 times) and 1+(p-1)r; diagonal rescaling preserves PSD.
        average = float((correlation.sum() - np.trace(correlation)) / (p * (p - 1)))
        target = np.outer(scales, scales) * average
        np.fill_diagonal(target, np.diag(sample))
        matrix = (1 - delta) * sample + delta * target
        params.update(
            shrinkage=delta,
            target="constant_correlation",
            average_correlation=average,
            intensity_rule="fixed_configuration_not_estimated_optimum",
        )
    else:
        raise ValueError("Unknown covariance estimator")
    asymmetry = float(np.linalg.norm(matrix - matrix.T, ord=2))
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = max(absolute_tolerance, relative_tolerance * float(np.linalg.norm(matrix, ord=2)))
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    valid = bool(
        np.isfinite(matrix).all()
        and asymmetry <= tolerance
        and eigenvalues[0] >= -tolerance
        and np.min(np.diag(matrix)) >= -tolerance
    )
    diagnostics = {
        "n_observations": t,
        "n_assets": p,
        "effective_weight_count": effective,
        "eigenvalues": eigenvalues.tolist(),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular": rank < p,
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]) if rank == p else None,
        "asymmetry_norm_before_symmetrising": asymmetry,
        "repair_amount": regularisation_amount,
        "p_over_T": p / t,
        "p_over_effective_count": p / effective,
    }
    runtime = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    config_hash = hashlib.sha256(
        json.dumps({"method": method, "parameters": params}, sort_keys=True).encode()
    ).hexdigest()
    matrix.setflags(write=False)
    location = np.array(location, copy=True)
    location.setflags(write=False)
    return CovarianceEstimate(
        matrix,
        location,
        window.asset_ids,
        method,
        params,
        diagnostics,
        window.window_hash,
        window.signal_at,
        window.interval_start[0],
        window.interval_end[-1],
        max(window.available_at),
        datetime.now(UTC),
        runtime,
        config_hash,
        {
            **dict(window.source_provenance),
            "source_snapshot_id": window.source_snapshot_id,
            "return_convention": window.return_convention,
            "eligibility_rule_id": window.eligibility_rule_id,
            "coverage_report_id": window.coverage_report_id,
        },
        status="valid" if valid else "invalid_matrix",
    )
