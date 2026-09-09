from __future__ import annotations

from pathlib import Path

import numpy as np

from dibo.models import fit_f, posterior
from dibo.schemas import METRIC_IDS, load_actions, load_graph
from tests.unit.test_initial_design import make_result

CONFIGS = Path(__file__).parents[2] / "configs"


def training_results():
    results = []
    for index in range(6):
        result = make_result(f"trial_{index:03}", tps=100.0 + index, ttft=0.2 + index * 0.01)
        coefficients = {action_id: 0.0 for action_id in result.trace.coefficients}
        coefficients[f"A{index % 8 + 1:02}"] = -0.8 + index * 0.3
        metrics = {metric_id: 0.1 + index * 0.02 for metric_id in METRIC_IDS}
        trace = result.trace.model_copy(
            update={
                "coefficients": coefficients,
                "config_hash": f"hash_{index}",
                "action_version": 1 if index == 0 else 2,
            }
        )
        results.append(result.model_copy(update={"trace": trace, "metrics": metrics}))
    return results


def test_fit_exactly_six_f_with_canonical_neighbor_dimensions() -> None:
    bundle = load_actions(CONFIGS / "actions_v1.yaml").model_copy(
        update={"action_version": 2, "parent_action_version": 1}
    )
    models = fit_f(training_results(), load_graph(CONFIGS / "graph.yaml"), bundle)

    assert tuple(models) == METRIC_IDS
    assert [len(models[metric].feature_order) for metric in METRIC_IDS] == [4, 3, 5, 4, 3, 5]
    assert all(model.fit_status == "ready" for model in models.values())
    assert models["m01"].observation_noise[0] == 2e-6
    assert models["m01"].observation_noise[1] == 1e-6
    prediction = posterior(models["m01"], np.zeros((2, 4)))
    assert prediction.mean.shape == (2,)
    assert prediction.variance.shape == (2,)
    assert np.all(prediction.variance >= 0)


def test_each_f_filters_only_its_missing_metric_rows() -> None:
    results = training_results()
    metrics = dict(results[0].metrics)
    metrics["m01"] = None
    results[0] = results[0].model_copy(update={"metrics": metrics})
    models = fit_f(
        results,
        load_graph(CONFIGS / "graph.yaml"),
        load_actions(CONFIGS / "actions_v1.yaml"),
    )

    assert models["m01"].n_samples == 5
    assert models["m02"].n_samples == 6


def test_f_reports_not_ready_and_constant_output() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")
    bundle = load_actions(CONFIGS / "actions_v1.yaml")
    assert fit_f(training_results()[:2], graph, bundle)["m01"].fit_status == "not_ready"

    constant = [result.model_copy(update={"metrics": {metric: 0.1 for metric in METRIC_IDS}}) for result in training_results()]
    model = fit_f(constant, graph, bundle)["m01"]
    assert model.fit_status == "unstable"
    assert model.failure_reason == "constant_output"