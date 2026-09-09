"""Gaussian-process models for Action-to-Metric F and Metrics-to-TPS G."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal
from warnings import catch_warnings, simplefilter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

from dibo.schemas import METRIC_IDS, ActionBundle, Graph, TrialResult, TrialStatus


@dataclass(frozen=True)
class Posterior:
    mean: np.ndarray
    variance: np.ndarray


@dataclass
class GPModel:
    model_id: str
    feature_order: tuple[str, ...]
    output_unit: str
    fit_status: Literal["ready", "not_ready", "unstable"]
    failure_reason: str | None
    training_trial_ids: tuple[str, ...]
    n_samples: int
    n_unique_configs: int
    observation_noise: tuple[float, ...]
    input_scaler: StandardScaler | None = None
    output_scaler: StandardScaler | None = None
    regressor: GaussianProcessRegressor | None = None

    def posterior(self, values: np.ndarray | list[list[float]]) -> Posterior:
        if self.fit_status != "ready" or self.regressor is None:
            raise RuntimeError(f"model {self.model_id} is not ready: {self.failure_reason}")
        inputs = np.asarray(values, dtype=np.float64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.ndim != 2 or inputs.shape[1] != len(self.feature_order):
            raise ValueError(f"{self.model_id} expects {len(self.feature_order)} features")
        if not np.all(np.isfinite(inputs)):
            raise ValueError("posterior inputs must be finite")
        assert self.input_scaler is not None
        assert self.output_scaler is not None
        mean_scaled, std_scaled = self.regressor.predict(
            self.input_scaler.transform(inputs),
            return_std=True,
        )
        output_scale = float(self.output_scaler.scale_[0])
        output_mean = float(self.output_scaler.mean_[0])
        return Posterior(
            mean=np.asarray(mean_scaled * output_scale + output_mean, dtype=np.float64),
            variance=np.asarray(np.square(std_scaled * output_scale), dtype=np.float64),
        )


def posterior(model: GPModel, values: np.ndarray | list[list[float]]) -> Posterior:
    """Return a named model's posterior in original output units."""
    return model.posterior(values)


def _eligible(result: TrialResult) -> bool:
    return (
        result.status == TrialStatus.SUCCESS
        and result.request_count > 0
        and result.successful_requests == result.request_count
    )


def _fit_model(
    model_id: str,
    feature_order: tuple[str, ...],
    output_unit: str,
    rows: list[tuple[str, str, list[float], float, float]],
    *,
    min_train_samples: int,
    min_unique_configs: int,
    seed: int,
) -> GPModel:
    trial_ids = tuple(row[0] for row in rows)
    unique_configs = len({row[1] for row in rows})
    metadata = {
        "model_id": model_id,
        "feature_order": feature_order,
        "output_unit": output_unit,
        "training_trial_ids": trial_ids,
        "n_samples": len(rows),
        "n_unique_configs": unique_configs,
        "observation_noise": tuple(row[4] for row in rows),
    }
    if len(rows) < min_train_samples:
        return GPModel(fit_status="not_ready", failure_reason="insufficient_samples", **metadata)
    if unique_configs < min_unique_configs:
        return GPModel(fit_status="not_ready", failure_reason="insufficient_unique_configs", **metadata)

    inputs = np.asarray([row[2] for row in rows], dtype=np.float64)
    outputs = np.asarray([row[3] for row in rows], dtype=np.float64).reshape(-1, 1)
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(outputs)):
        return GPModel(fit_status="unstable", failure_reason="non_finite_training_data", **metadata)
    if float(np.ptp(outputs)) <= 1e-12:
        return GPModel(fit_status="unstable", failure_reason="constant_output", **metadata)

    input_scaler = StandardScaler().fit(inputs)
    output_scaler = StandardScaler().fit(outputs)
    scaled_inputs = input_scaler.transform(inputs)
    scaled_outputs = output_scaler.transform(outputs).ravel()
    alpha = np.asarray([row[4] for row in rows], dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(inputs.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    )
    regressor = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        random_state=seed,
        n_restarts_optimizer=0,
    )
    try:
        with catch_warnings():
            simplefilter("ignore", ConvergenceWarning)
            regressor.fit(scaled_inputs, scaled_outputs)
    except (ValueError, np.linalg.LinAlgError) as error:
        return GPModel(fit_status="unstable", failure_reason=f"fit_failed:{error}", **metadata)
    return GPModel(
        fit_status="ready",
        failure_reason=None,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        regressor=regressor,
        **metadata,
    )


def fit_f(
    trials: list[TrialResult],
    graph: Graph,
    current_action_bundle: ActionBundle,
    *,
    min_train_samples: int = 5,
    min_unique_configs: int = 3,
    base_noise: float = 1e-6,
    historical_noise_multiplier: float = 2.0,
    seed: int = 42,
) -> dict[str, GPModel]:
    """Fit exactly six F models from each Metric's neighboring saved z values."""
    models: dict[str, GPModel] = {}
    for metric_id in METRIC_IDS:
        neighbors = graph.neighbors(metric_id)
        rows: list[tuple[str, str, list[float], float, float]] = []
        for result in trials:
            metric_value = result.metrics.get(metric_id)
            if not _eligible(result) or metric_value is None or not isfinite(metric_value):
                continue
            if result.trace.base_trial_id is None and result.trace.phase == "bo":
                continue
            noise_multiplier = (
                1.0
                if result.trace.action_version == current_action_bundle.action_version
                else historical_noise_multiplier
            )
            rows.append(
                (
                    result.trial_id,
                    result.trace.config_hash,
                    [result.trace.coefficients[action_id] for action_id in neighbors],
                    float(metric_value),
                    base_noise * noise_multiplier,
                )
            )
        models[metric_id] = _fit_model(
            f"F_{metric_id}",
            neighbors,
            metric_id,
            rows,
            min_train_samples=min_train_samples,
            min_unique_configs=min_unique_configs,
            seed=seed,
        )
    return models


def fit_g(
    trials: list[TrialResult],
    *,
    min_train_samples: int = 5,
    min_unique_configs: int = 3,
    base_noise: float = 1e-6,
    seed: int = 42,
) -> GPModel:
    """Fit G only from complete measured Metrics and measured TPS."""
    rows: list[tuple[str, str, list[float], float, float]] = []
    for result in trials:
        values = [result.metrics.get(metric_id) for metric_id in METRIC_IDS]
        if (
            not _eligible(result)
            or result.throughput_tps is None
            or not isfinite(result.throughput_tps)
            or any(value is None or not isfinite(value) for value in values)
        ):
            continue
        rows.append(
            (
                result.trial_id,
                result.trace.config_hash,
                [float(value) for value in values if value is not None],
                float(result.throughput_tps),
                base_noise,
            )
        )
    return _fit_model(
        "G",
        METRIC_IDS,
        "output_token_throughput_tps",
        rows,
        min_train_samples=min_train_samples,
        min_unique_configs=min_unique_configs,
        seed=seed,
    )