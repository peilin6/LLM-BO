from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dibo.models import Posterior
from dibo.schemas import METRIC_IDS, Threshold
from dibo.selector import SelectorConfig, select_metrics
from tests.unit.test_initial_design import make_result


class LinearG:
    fit_status = "ready"

    def __init__(self, reference: np.ndarray, slopes: np.ndarray, variance: float = 0.0) -> None:
        self.reference = reference
        self.slopes = slopes
        self.variance = variance

    def posterior(self, values):
        inputs = np.asarray(values, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        mean = 100.0 + (inputs - self.reference) @ self.slopes
        return Posterior(mean=mean, variance=np.full(len(inputs), self.variance))


class FixedF:
    fit_status = "ready"
    feature_order = ("A01",)

    def __init__(self, variance: float, output_scale: float) -> None:
        self.variance = variance
        self.output_scaler = SimpleNamespace(scale_=np.asarray([output_scale]))

    def posterior(self, values):
        return Posterior(mean=np.asarray([0.0]), variance=np.asarray([self.variance]))


def no_breaches() -> dict[str, Threshold]:
    return {
        "m01": Threshold(kind="upper", trigger=0.99, target=0.85, scale=0.05),
        "m02": Threshold(kind="upper", trigger=1.0, target=0.01, scale=0.01),
        "m04": Threshold(kind="upper", trigger=100.0, target=4.0, scale=2.0),
        "m06": Threshold(kind="upper", trigger=10.0, target=1.0, scale=0.1),
    }


def varied_history():
    results = []
    for index in range(6):
        result = make_result(f"trial_{index:03}")
        metrics = {
            "m01": 0.2 + index * 0.1,
            "m02": 0.01 + index * 0.01,
            "m03": 0.2 + index * 0.1,
            "m04": 2.0 + index,
            "m05": 0.1 + index * 0.1,
            "m06": 0.2 + index * 0.1,
        }
        results.append(result.model_copy(update={"metrics": metrics}))
    return results


def test_g_searches_lcb_range_and_applies_third_metric_ratio() -> None:
    history = varied_history()
    reference = history[2]
    reference_vector = np.asarray([reference.metrics[metric] for metric in METRIC_IDS], dtype=float)
    model = LinearG(reference_vector, np.asarray([0.0, 0.0, 3.0, 0.0, 2.5, 2.4]))

    selection = select_metrics(
        reference,
        history,
        {},
        model,
        no_breaches(),
        config=SelectorConfig(grid_size=8),
    )

    assert selection.mode == "g_counterfactual"
    assert selection.selected_metrics == ("m03", "m05", "m06")
    assert all(target.kind == "point" for target in selection.targets.values())
    assert selection.targets["m03"].value == selection.credible_ranges["m03"][1]
    assert selection.g_posterior["m03"]["sigma"] == 0.0
    assert selection.g_posterior["m03"]["beta"] == 1.0
    assert sum(selection.weights.values()) == 1.0


def test_lcb_penalizes_uncertainty() -> None:
    history = varied_history()
    reference = history[2]
    reference_vector = np.asarray([reference.metrics[metric] for metric in METRIC_IDS], dtype=float)
    certain = LinearG(reference_vector, np.asarray([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]))
    uncertain = LinearG(reference_vector, np.asarray([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]), variance=4.0)

    assert select_metrics(reference, history, {}, certain, no_breaches()).mode == "g_counterfactual"
    assert select_metrics(reference, history, {}, uncertain, no_breaches()).mode == "rotation"


def test_incomplete_reference_and_unique_ranges_fall_back_to_rotation() -> None:
    history = varied_history()
    reference = history[0].model_copy(update={"metrics": {**history[0].metrics, "m05": None}})
    selection = select_metrics(
        reference,
        [reference],
        {},
        SimpleNamespace(fit_status="ready"),
        no_breaches(),
        config=SelectorConfig(rotation_index=4),
    )

    assert selection.mode == "rotation"
    assert selection.selected_metrics == ("m05",)
    assert selection.targets == {}


def test_range_outside_physical_domain_and_normalized_f_uncertainty_fallback() -> None:
    history = []
    for index in range(6):
        result = make_result(f"trial_domain_{index}")
        metrics = {metric_id: 0.1 for metric_id in METRIC_IDS}
        metrics["m03"] = 1.2 + index * 0.1
        history.append(result.model_copy(update={"metrics": metrics}))
    reference = history[0]
    reference_vector = np.asarray([reference.metrics[metric] for metric in METRIC_IDS], dtype=float)
    selection = select_metrics(
        reference,
        history,
        {
            "m01": FixedF(variance=4.0, output_scale=10.0),
            "m02": FixedF(variance=1.0, output_scale=1.0),
        },
        LinearG(reference_vector, np.ones(6)),
        no_breaches(),
        config=SelectorConfig(metric_top_k=1),
    )

    assert selection.mode == "uncertainty"
    assert selection.selected_metrics == ("m02",)
    assert selection.scores["m02"] == 1.0
