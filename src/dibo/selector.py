"""Metric selection and complete graph-neighbor Action unions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from dibo.models import GPModel
from dibo.schemas import METRIC_IDS, Graph, MetricSelection, MetricTarget, Threshold, TrialResult

_PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "m01": (0.0, 1.0),
    "m02": (0.0, float("inf")),
    "m03": (0.0, 1.0),
    "m04": (0.0, float("inf")),
    "m05": (0.0, 1.0),
    "m06": (0.0, float("inf")),
}


@dataclass(frozen=True)
class SelectorConfig:
    metric_top_k: int = 2
    metric_top_k_max: int = 3
    third_metric_ratio: float = 0.8
    beta: float = 1.0
    grid_size: int = 64
    rotation_index: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.metric_top_k <= self.metric_top_k_max <= 3:
            raise ValueError("metric top-k bounds are inconsistent")
        if not 0 < self.third_metric_ratio <= 1:
            raise ValueError("third_metric_ratio must be in (0, 1]")
        if not isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.rotation_index < 0:
            raise ValueError("rotation_index must be non-negative")


DEFAULT_SELECTOR_CONFIG = SelectorConfig()


def union_neighbor_actions(metric_ids: Sequence[str], graph: Graph) -> tuple[str, ...]:
    """Return the full de-duplicated neighbor union in canonical Action order."""
    if not metric_ids:
        raise ValueError("at least one Metric is required")
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError("metric_ids must not contain duplicates")
    unknown = set(metric_ids) - set(graph.metric_order)
    if unknown:
        raise ValueError(f"unknown Metrics: {sorted(unknown)}")
    selected = {action for metric_id in metric_ids for action in graph.neighbors(metric_id)}
    return tuple(action_id for action_id in graph.action_order if action_id in selected)


def _limit(scores: Mapping[str, float], config: SelectorConfig) -> tuple[str, ...]:
    ordered = sorted(scores, key=lambda metric_id: (-scores[metric_id], metric_id))
    selected = ordered[: min(config.metric_top_k, 2)]
    if (
        config.metric_top_k >= 2
        and config.metric_top_k_max >= 3
        and len(ordered) >= 3
        and scores[ordered[2]] > 0
        and scores[ordered[2]] >= config.third_metric_ratio * scores[ordered[1]]
    ):
        selected.append(ordered[2])
    return tuple(selected)


def _weights(selected: Sequence[str], scores: Mapping[str, float]) -> dict[str, float]:
    positive = {metric_id: max(float(scores[metric_id]), 0.0) for metric_id in selected}
    total = sum(positive.values())
    if total == 0:
        return {metric_id: 1.0 / len(selected) for metric_id in selected}
    floor = max(total * 1e-9, 1e-12)
    adjusted = {metric_id: max(score, floor) for metric_id, score in positive.items()}
    adjusted_total = sum(adjusted.values())
    return {metric_id: score / adjusted_total for metric_id, score in adjusted.items()}


def _hard_thresholds(
    reference: TrialResult,
    thresholds: Mapping[str, Threshold],
) -> tuple[dict[str, float], dict[str, MetricTarget]]:
    scores: dict[str, float] = {}
    targets: dict[str, MetricTarget] = {}
    for metric_id in ("m01", "m02", "m04", "m06"):
        threshold = thresholds.get(metric_id)
        value = reference.metrics.get(metric_id)
        if threshold is None or value is None or not isfinite(value):
            continue
        if not isinstance(threshold.trigger, (int, float)) or not isinstance(threshold.target, (int, float)):
            continue
        if not isinstance(threshold.scale, (int, float)) or not isfinite(threshold.scale) or threshold.scale <= 0:
            continue
        if threshold.kind == "interval":
            continue
        violated = value >= threshold.trigger if threshold.kind == "upper" else value <= threshold.trigger
        if not violated:
            continue
        difference = value - threshold.trigger if threshold.kind == "upper" else threshold.trigger - value
        scores[metric_id] = max(0.0, difference / threshold.scale)
        targets[metric_id] = MetricTarget(
            kind=threshold.kind,
            value=float(threshold.target),
            scale=float(threshold.scale),
        )
    return scores, targets


def _credible_range(metric_id: str, values: list[float]) -> tuple[float, float] | None:
    unique = np.unique([value for value in values if isfinite(value)])
    if len(unique) < 2:
        return None
    if len(unique) >= 5:
        low, high = np.quantile(unique, (0.05, 0.95), method="linear")
    else:
        low, high = unique[0], unique[-1]
    physical_low, physical_high = _PHYSICAL_RANGES[metric_id]
    low = max(float(low), physical_low)
    high = min(float(high), physical_high)
    return (low, high) if low < high else None


def _counterfactual(
    reference: TrialResult,
    history: Sequence[TrialResult],
    g_model: GPModel,
    config: SelectorConfig,
) -> tuple[
    dict[str, float],
    dict[str, MetricTarget],
    dict[str, tuple[float, float]],
    dict[str, dict[str, float]],
]:
    reference_values = [reference.metrics.get(metric_id) for metric_id in METRIC_IDS]
    if g_model.fit_status != "ready" or any(value is None or not isfinite(value) for value in reference_values):
        return {}, {}, {}, {}
    reference_vector = np.asarray(reference_values, dtype=float)
    reference_mean = float(g_model.posterior(reference_vector).mean[0])
    scores: dict[str, float] = {}
    targets: dict[str, MetricTarget] = {}
    ranges: dict[str, tuple[float, float]] = {}
    posterior_details: dict[str, dict[str, float]] = {}
    for index, metric_id in enumerate(METRIC_IDS):
        observed = [
            float(value)
            for result in history
            if (value := result.metrics.get(metric_id)) is not None and isfinite(value)
        ]
        credible = _credible_range(metric_id, observed)
        if credible is None:
            continue
        observed_in_range = [value for value in observed if credible[0] <= value <= credible[1]]
        grid = np.unique(
            np.concatenate((np.linspace(*credible, config.grid_size), np.asarray(observed_in_range)))
        )
        candidates = np.repeat(reference_vector.reshape(1, -1), len(grid), axis=0)
        candidates[:, index] = grid
        prediction = g_model.posterior(candidates)
        sigma = np.sqrt(np.maximum(prediction.variance, 0.0))
        lcb = prediction.mean - config.beta * sigma
        distances = np.abs(grid - reference_vector[index])
        best = min(range(len(grid)), key=lambda item: (-lcb[item], distances[item], grid[item]))
        importance = max(0.0, float(lcb[best] - reference_mean))
        scale = float(np.subtract(*np.quantile(grid, (0.75, 0.25), method="linear")))
        if scale <= 0:
            scale = max((credible[1] - credible[0]) / 2, 1e-12)
        scores[metric_id] = importance
        targets[metric_id] = MetricTarget(kind="point", value=float(grid[best]), scale=scale)
        ranges[metric_id] = credible
        posterior_details[metric_id] = {
            "mean": float(prediction.mean[best]),
            "sigma": float(sigma[best]),
            "lcb": float(lcb[best]),
            "beta": config.beta,
            "importance": importance,
        }
    return scores, targets, ranges, posterior_details


def select_metrics(
    reference: TrialResult,
    history: Sequence[TrialResult],
    f_models: Mapping[str, GPModel],
    g_model: GPModel,
    thresholds: Mapping[str, Threshold],
    *,
    config: SelectorConfig = DEFAULT_SELECTOR_CONFIG,
) -> MetricSelection:
    """Select Metrics by threshold, G counterfactual, uncertainty, then rotation."""
    scores, targets = _hard_thresholds(reference, thresholds)
    if scores:
        selected = _limit(scores, config)
        return MetricSelection(
            reference_trial_id=reference.trial_id,
            mode="hard_threshold",
            selected_metrics=selected,
            scores={metric: scores[metric] for metric in selected},
            weights=_weights(selected, scores),
            targets={metric: targets[metric] for metric in selected},
        )

    scores, targets, ranges, details = _counterfactual(reference, history, g_model, config)
    positive_scores = {metric: score for metric, score in scores.items() if score > 1e-12}
    if positive_scores:
        selected = _limit(positive_scores, config)
        return MetricSelection(
            reference_trial_id=reference.trial_id,
            mode="g_counterfactual",
            selected_metrics=selected,
            scores={metric: scores[metric] for metric in selected},
            weights=_weights(selected, scores),
            targets={metric: targets[metric] for metric in selected},
            credible_ranges={metric: ranges[metric] for metric in selected},
            g_posterior={metric: details[metric] for metric in selected},
        )

    uncertainties: dict[str, float] = {}
    for metric_id in METRIC_IDS:
        model = f_models.get(metric_id)
        if model is None or model.fit_status != "ready" or model.output_scaler is None:
            continue
        variance = float(model.posterior(np.zeros((1, len(model.feature_order)))).variance[0])
        output_scale = float(model.output_scaler.scale_[0])
        uncertainties[metric_id] = variance / max(output_scale * output_scale, 1e-12)
    if uncertainties:
        selected = _limit(uncertainties, config)
        return MetricSelection(
            reference_trial_id=reference.trial_id,
            mode="uncertainty",
            selected_metrics=selected,
            scores={metric: uncertainties[metric] for metric in selected},
            weights=_weights(selected, uncertainties),
            targets={},
        )

    metric_id = METRIC_IDS[config.rotation_index % len(METRIC_IDS)]
    return MetricSelection(
        reference_trial_id=reference.trial_id,
        mode="rotation",
        selected_metrics=(metric_id,),
        scores={metric_id: 0.0},
        weights={metric_id: 1.0},
        targets={},
    )