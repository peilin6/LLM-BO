"""Dynamic-dimensional Metric-Loss Bayesian optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.stats import qmc

from dibo.compiler import CompileEnvironment, compile_config
from dibo.models import GPModel
from dibo.schemas import (
    ACTION_IDS,
    METRIC_IDS,
    ActionBundle,
    BOSelection,
    MetricSelection,
    MetricTarget,
    TrialResult,
    TrialStatus,
)

_RATIO_METRICS = frozenset({"m01", "m03", "m05"})


@dataclass(frozen=True)
class OptimizerConfig:
    environment: CompileEnvironment
    bo_round: int = 1
    base_trial_id: str = "x_base"
    candidate_pool_size: int = 2048
    max_candidate_pools: int = 3
    mc_samples: int = 128
    seed: int = 42
    coefficient_low: float = -1.0
    coefficient_high: float = 1.0
    acquisition_tie_tolerance: float = 1e-6
    loss_tie_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.bo_round < 1:
            raise ValueError("bo_round must be positive")
        if self.candidate_pool_size < 1 or self.max_candidate_pools < 1:
            raise ValueError("candidate pool limits must be positive")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be positive")
        if not -1 <= self.coefficient_low < self.coefficient_high <= 1:
            raise ValueError("coefficient bounds must be ordered within [-1, 1]")
        if self.acquisition_tie_tolerance < 0 or self.loss_tie_tolerance < 0:
            raise ValueError("tie tolerances must be non-negative")


@dataclass(frozen=True)
class _Candidate:
    coefficients: dict[str, float]
    config_hash: str
    f_means: dict[str, float]
    f_variances: dict[str, float]
    expected_loss: float | None
    acquisition: float | None
    uncertainty: float
    predicted_tps: float | None = None


def metric_distance(values: np.ndarray | Sequence[float] | float, target: MetricTarget) -> np.ndarray:
    """Return normalized distance to an upper, lower, interval, or point target."""
    samples = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("metric samples must be finite")
    if not isfinite(target.scale):
        raise ValueError("target scale must be finite")
    if target.kind in {"upper", "lower", "point"} and target.value is None:
        raise ValueError(f"{target.kind} target requires value")
    if target.value is not None and not isfinite(target.value):
        raise ValueError("target value must be finite")
    if target.kind == "upper":
        return np.maximum(0.0, (samples - float(target.value)) / target.scale)
    if target.kind == "lower":
        return np.maximum(0.0, (float(target.value) - samples) / target.scale)
    if target.kind == "point":
        return np.abs(samples - float(target.value)) / target.scale
    if (
        target.lower is None
        or target.upper is None
        or not isfinite(target.lower)
        or not isfinite(target.upper)
        or target.lower > target.upper
    ):
        raise ValueError("interval target requires ordered lower and upper bounds")
    return np.maximum(
        np.maximum(float(target.lower) - samples, samples - float(target.upper)),
        0.0,
    ) / target.scale


def metric_loss(
    samples: Mapping[str, np.ndarray | Sequence[float] | float],
    targets: Mapping[str, MetricTarget],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combine per-Metric distances in real output units."""
    if not targets or set(targets) != set(weights) or set(samples) != set(targets):
        raise ValueError("samples, targets, and weights must contain the same Metrics")
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Metric weights must be finite and non-negative")
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        raise ValueError("Metric weights must sum to one")
    total: np.ndarray | None = None
    for metric_id in targets:
        contribution = weights[metric_id] * metric_distance(samples[metric_id], targets[metric_id])
        total = contribution if total is None else total + contribution
    assert total is not None
    return np.asarray(total, dtype=np.float64)


def monte_carlo_ei(loss_samples: np.ndarray | Sequence[float], incumbent_loss: float) -> float:
    """Compute expected improvement from sampled nonlinear Loss values."""
    losses = np.asarray(loss_samples, dtype=np.float64)
    if losses.size == 0 or not np.all(np.isfinite(losses)):
        raise ValueError("loss_samples must be non-empty and finite")
    if not isfinite(incumbent_loss) or incumbent_loss < 0:
        raise ValueError("incumbent_loss must be finite and non-negative")
    return float(np.mean(np.maximum(0.0, incumbent_loss - losses)))


def _measured_incumbent(
    history: Sequence[TrialResult],
    selection: MetricSelection,
) -> float | None:
    losses: list[float] = []
    for result in history:
        values = {metric_id: result.metrics.get(metric_id) for metric_id in selection.selected_metrics}
        if (
            result.status != TrialStatus.SUCCESS
            or any(value is None or not isfinite(value) for value in values.values())
            or not selection.targets
        ):
            continue
        loss = metric_loss(
            {metric_id: float(value) for metric_id, value in values.items() if value is not None},
            selection.targets,
            selection.weights,
        )
        losses.append(float(loss))
    return min(losses) if losses else None


def _full_coefficients(selected_actions: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    selected_values = dict(zip(selected_actions, values, strict=True))
    return {action_id: float(selected_values.get(action_id, 0.0)) for action_id in ACTION_IDS}


def _project_samples(metric_id: str, samples: np.ndarray) -> np.ndarray:
    if metric_id in _RATIO_METRICS:
        return np.clip(samples, 0.0, 1.0)
    return np.maximum(samples, 0.0)


def _predict_candidate(
    coefficients: Mapping[str, float],
    selection: MetricSelection,
    f_models: Mapping[str, GPModel],
    incumbent: float | None,
    config: OptimizerConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float], float | None, float | None, float]:
    means: dict[str, float] = {}
    variances: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    normalized_variances: list[float] = []
    selected_ready = bool(selection.targets)
    for metric_id in METRIC_IDS:
        model = f_models.get(metric_id)
        if model is None or model.fit_status != "ready":
            if metric_id in selection.selected_metrics:
                selected_ready = False
            continue
        features = np.asarray([[coefficients[action_id] for action_id in model.feature_order]])
        posterior = model.posterior(features)
        mean = float(posterior.mean[0])
        variance = max(float(posterior.variance[0]), 0.0)
        means[metric_id] = mean
        variances[metric_id] = variance
        if metric_id in selection.selected_metrics:
            output_scale = float(model.output_scaler.scale_[0]) if model.output_scaler is not None else 1.0
            normalized_variances.append(variance / max(output_scale * output_scale, 1e-12))
            draws[metric_id] = _project_samples(
                metric_id,
                rng.normal(mean, np.sqrt(variance), config.mc_samples),
            )
    uncertainty = float(sum(normalized_variances))
    if not selected_ready or incumbent is None:
        return means, variances, None, None, uncertainty
    losses = metric_loss(draws, selection.targets, selection.weights)
    return means, variances, float(np.mean(losses)), monte_carlo_ei(losses, incumbent), uncertainty


def _eligible_ties(candidates: Sequence[_Candidate], config: OptimizerConfig) -> list[_Candidate]:
    best_acquisition = max(candidate.acquisition or 0.0 for candidate in candidates)
    acquisition_ties = [
        candidate
        for candidate in candidates
        if best_acquisition - (candidate.acquisition or 0.0) <= config.acquisition_tie_tolerance
    ]
    best_loss = min(
        candidate.expected_loss for candidate in acquisition_ties if candidate.expected_loss is not None
    )
    return [
        candidate
        for candidate in acquisition_ties
        if candidate.expected_loss is not None
        and candidate.expected_loss - best_loss <= config.loss_tie_tolerance
    ]


def _g_tie_break(
    candidates: Sequence[_Candidate],
    g_model: GPModel,
    config: OptimizerConfig,
) -> tuple[_Candidate, bool]:
    ties = _eligible_ties(candidates, config)
    if g_model.fit_status != "ready" or any(set(candidate.f_means) != set(METRIC_IDS) for candidate in ties):
        return min(
            ties,
            key=lambda candidate: (
                -(candidate.acquisition or 0.0),
                candidate.expected_loss
                if candidate.expected_loss is not None
                else float("inf"),
                tuple(candidate.coefficients.values()),
            ),
        ), False
    ranked: list[_Candidate] = []
    for candidate in ties:
        metrics = np.asarray([[candidate.f_means[metric_id] for metric_id in METRIC_IDS]])
        predicted_tps = float(g_model.posterior(metrics).mean[0])
        if not isfinite(predicted_tps):
            return min(
                ties,
                key=lambda item: (
                    -(item.acquisition or 0.0),
                    item.expected_loss if item.expected_loss is not None else float("inf"),
                    tuple(item.coefficients.values()),
                ),
            ), False
        ranked.append(
            _Candidate(
                **{**candidate.__dict__, "predicted_tps": predicted_tps},
            )
        )
    return max(ranked, key=lambda candidate: (candidate.predicted_tps, tuple(candidate.coefficients.values()))), True


def suggest(
    metric_selection: MetricSelection,
    selected_actions: Sequence[str],
    f_models: Mapping[str, GPModel],
    g_model: GPModel,
    base_config: Mapping[str, object],
    action_bundle: ActionBundle,
    history: Sequence[TrialResult],
    *,
    config: OptimizerConfig,
) -> BOSelection | None:
    """Suggest one legal unique candidate in exactly the selected Action union."""
    if not 1 <= len(selected_actions) <= len(ACTION_IDS):
        raise ValueError("selected_actions must contain 1 through 8 Actions")
    if len(set(selected_actions)) != len(selected_actions):
        raise ValueError("selected_actions contains duplicates")
    canonical = tuple(action_id for action_id in ACTION_IDS if action_id in selected_actions)
    if tuple(selected_actions) != canonical:
        raise ValueError("selected_actions must use canonical Action order")

    incumbent = _measured_incumbent(history, metric_selection)
    existing_hashes = {result.trace.config_hash for result in history}
    seen_hashes: set[str] = set()
    candidates: list[_Candidate] = []
    invalid_count = 0
    duplicate_count = 0
    rng = np.random.default_rng(config.seed)
    for pool_index in range(config.max_candidate_pools):
        sampler = qmc.Sobol(d=len(selected_actions), scramble=True, seed=config.seed + pool_index)
        unit_points = sampler.random(config.candidate_pool_size)
        points = qmc.scale(unit_points, config.coefficient_low, config.coefficient_high)
        for point in points:
            coefficients = _full_coefficients(selected_actions, point)
            trace = compile_config(
                base_config,
                action_bundle,
                selected_actions,
                coefficients,
                config.environment,
                trial_id="candidate",
                phase="bo",
                bo_round=config.bo_round,
                compile_base_id=config.base_trial_id,
                base_trial_id=config.base_trial_id,
                selected_metrics=metric_selection.selected_metrics,
            )
            if not trace.valid:
                invalid_count += 1
                continue
            if trace.config_hash in existing_hashes or trace.config_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(trace.config_hash)
            means, variances, expected_loss, acquisition, uncertainty = _predict_candidate(
                coefficients,
                metric_selection,
                f_models,
                incumbent,
                config,
                rng,
            )
            candidates.append(
                _Candidate(
                    coefficients,
                    trace.config_hash,
                    means,
                    variances,
                    expected_loss,
                    acquisition,
                    uncertainty,
                )
            )
        if candidates:
            break
    if not candidates:
        return None

    ready_for_ei = incumbent is not None and all(
        f_models.get(metric_id) is not None and f_models[metric_id].fit_status == "ready"
        for metric_id in metric_selection.selected_metrics
    )
    positive_ei = ready_for_ei and any((candidate.acquisition or 0.0) > 0 for candidate in candidates)
    g_tiebreak_used = False
    if positive_ei:
        chosen, g_tiebreak_used = _g_tie_break(candidates, g_model, config)
        source = "metric_ei"
    elif ready_for_ei:
        chosen = max(candidates, key=lambda candidate: (candidate.uncertainty, tuple(candidate.coefficients.values())))
        source = "uncertainty_exploration"
    else:
        chosen = candidates[0]
        source = "sobol_exploration"

    return BOSelection(
        bo_round=config.bo_round,
        action_version=action_bundle.action_version,
        base_trial_id=config.base_trial_id,
        selected_metrics=metric_selection.selected_metrics,
        selected_actions=tuple(selected_actions),
        bo_dimension=len(selected_actions),
        coefficients=chosen.coefficients,
        predictions={
            "targets": {
                metric_id: target.model_dump(mode="json")
                for metric_id, target in metric_selection.targets.items()
            },
            "weights": metric_selection.weights,
            "credible_ranges": metric_selection.credible_ranges,
            "candidate_count": len(candidates),
            "invalid_count": invalid_count,
            "duplicate_count": duplicate_count,
            "requested_z": {
                action_id: chosen.coefficients[action_id] for action_id in selected_actions
            },
            "full_z": chosen.coefficients,
            "final_config_hash": chosen.config_hash,
            "f_mean": chosen.f_means,
            "f_variance": chosen.f_variances,
            "predicted_loss": chosen.expected_loss,
            "g_tiebreak_used": g_tiebreak_used,
            "predicted_tps": chosen.predicted_tps,
        },
        acquisition_name="metric_loss_ei",
        acquisition_value=chosen.acquisition,
        source=source,
    )