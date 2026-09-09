from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dibo.optimizer as optimizer_module
from dibo.compiler import CompileEnvironment
from dibo.models import Posterior
from dibo.optimizer import OptimizerConfig, suggest
from dibo.schemas import ACTION_IDS, MetricSelection, MetricTarget, load_actions, load_parameters
from tests.unit.test_initial_design import BASE_CONFIG, make_result

CONFIGS = Path(__file__).parents[2] / "configs"


class ConstantModel:
    def __init__(
        self,
        feature_order: tuple[str, ...],
        mean: float,
        variance: float,
        *,
        fit_status: str = "ready",
    ) -> None:
        self.feature_order = feature_order
        self.mean = mean
        self.variance = variance
        self.fit_status = fit_status
        self.output_scaler = SimpleNamespace(scale_=np.asarray([1.0]))

    def posterior(self, values: np.ndarray) -> Posterior:
        return Posterior(
            mean=np.full(values.shape[0], self.mean),
            variance=np.full(values.shape[0], self.variance),
        )


def optimizer_config(**updates: object) -> OptimizerConfig:
    environment = CompileEnvironment(
        load_parameters(CONFIGS / "parameters.yaml"),
        8,
        32,
        32,
        8192,
        (8, 16, 32),
        model_context="synthetic-model",
        workload_context="synthetic-workload",
    )
    values = {
        "environment": environment,
        "candidate_pool_size": 16,
        "max_candidate_pools": 2,
        "mc_samples": 16,
        **updates,
    }
    return OptimizerConfig(**values)


def selection(metrics: tuple[str, ...] = ("m01",)) -> MetricSelection:
    weight = 1.0 / len(metrics)
    return MetricSelection(
        reference_trial_id="trial_base",
        mode="hard_threshold",
        selected_metrics=metrics,
        scores={metric_id: 1.0 for metric_id in metrics},
        weights={metric_id: weight for metric_id in metrics},
        targets={
            metric_id: MetricTarget(kind="upper", value=0.1, scale=0.1)
            for metric_id in metrics
        },
    )


@pytest.mark.parametrize("dimension", range(1, 9))
def test_suggest_supports_one_through_eight_dimensions_and_zero_fills(dimension: int) -> None:
    selected_actions = ACTION_IDS[:dimension]
    result = suggest(
        selection(),
        selected_actions,
        {},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [],
        config=optimizer_config(seed=dimension),
    )

    assert result is not None
    assert result.bo_dimension == dimension
    assert tuple(result.coefficients) == ACTION_IDS
    assert all(result.coefficients[action_id] == 0.0 for action_id in ACTION_IDS[dimension:])
    assert result.source == "sobol_exploration"


def test_final_compiled_hashes_are_filtered_against_history(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = make_result("trial_existing")
    monkeypatch.setattr(optimizer_module, "compile_config", lambda *args, **kwargs: existing.trace)

    result = suggest(
        selection(),
        ("A01",),
        {},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [existing],
        config=optimizer_config(candidate_pool_size=4),
    )

    assert result is None


def test_all_invalid_candidate_pools_return_none(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = make_result("invalid").trace.model_copy(update={"constraint_errors": ("invalid",)})
    monkeypatch.setattr(optimizer_module, "compile_config", lambda *args, **kwargs: invalid)

    result = suggest(
        selection(),
        ("A01",),
        {},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [],
        config=optimizer_config(candidate_pool_size=2, max_candidate_pools=2),
    )

    assert result is None


def test_zero_ei_uses_uncertainty_fallback_inside_selected_actions() -> None:
    reference = make_result("trial_base").model_copy(
        update={"metrics": {metric_id: 0.1 for metric_id in ("m01", "m02", "m03", "m04", "m05", "m06")}}
    )
    result = suggest(
        selection(),
        ("A01", "A02"),
        {"m01": ConstantModel(("A01", "A02"), 0.1, 0.0)},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [reference],
        config=optimizer_config(),
    )

    assert result is not None
    assert result.source == "uncertainty_exploration"
    assert all(result.coefficients[action_id] == 0.0 for action_id in ACTION_IDS[2:])


def test_fixed_posterior_samples_produce_hand_computed_mc_ei() -> None:
    metrics = {metric_id: 0.1 for metric_id in ("m01", "m02", "m03", "m04", "m05", "m06")}
    metrics["m01"] = 0.3
    reference = make_result("trial_base").model_copy(update={"metrics": metrics})
    config = optimizer_config(candidate_pool_size=1, max_candidate_pools=1, mc_samples=4, seed=11)
    result = suggest(
        selection(),
        ("A06",),
        {"m01": ConstantModel(("A06",), 0.2, 0.01)},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [reference],
        config=config,
    )

    assert result is not None
    samples = np.clip(np.random.default_rng(11).normal(0.2, 0.1, 4), 0.0, 1.0)
    losses = np.maximum(0.0, (samples - 0.1) / 0.1)
    assert result.acquisition_value == pytest.approx(np.mean(np.maximum(0.0, 2.0 - losses)))
    assert result.predictions["predicted_loss"] == pytest.approx(np.mean(losses))
    assert result.predictions["f_variance"] == {"m01": pytest.approx(0.01)}
    assert result.predictions["targets"]["m01"]["kind"] == "upper"
    assert result.predictions["weights"] == {"m01": 1.0}
    assert result.predictions["requested_z"] == {"A06": result.coefficients["A06"]}
    assert result.predictions["full_z"] == result.coefficients


def test_g_cannot_override_a_clearly_better_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    template = make_result("template").trace

    def fake_compile(*args: object, **kwargs: object):
        coefficients = args[3]
        config_hash = f"hash-{float(coefficients['A01']):.12f}"
        return template.model_copy(update={"config_hash": config_hash})

    def fake_prediction(coefficients, *args, **kwargs):
        positive = coefficients["A01"] > 0
        means = {metric_id: (0.0 if positive else 1.0) for metric_id in ("m01", "m02", "m03", "m04", "m05", "m06")}
        return means, {"m01": 0.1}, 0.1, (0.9 if positive else 0.5), 0.1

    class PreferLargeMetrics:
        fit_status = "ready"

        @staticmethod
        def posterior(values: np.ndarray) -> Posterior:
            return Posterior(mean=np.sum(values, axis=1), variance=np.zeros(values.shape[0]))

    monkeypatch.setattr(optimizer_module, "compile_config", fake_compile)
    monkeypatch.setattr(optimizer_module, "_predict_candidate", fake_prediction)
    result = suggest(
        selection(),
        ("A01",),
        {"m01": ConstantModel(("A01",), 0.2, 0.1)},
        PreferLargeMetrics(),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [make_result("trial_base")],
        config=optimizer_config(),
    )

    assert result is not None
    assert result.coefficients["A01"] > 0
    assert result.acquisition_value == 0.9


def test_g_is_skipped_when_any_f_prediction_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    template = make_result("template").trace

    def fake_compile(*args: object, **kwargs: object):
        coefficients = args[3]
        return template.model_copy(update={"config_hash": f"hash-{float(coefficients['A01']):.12f}"})

    monkeypatch.setattr(optimizer_module, "compile_config", fake_compile)
    monkeypatch.setattr(
        optimizer_module,
        "_predict_candidate",
        lambda *args, **kwargs: ({"m01": 0.1}, {"m01": 0.1}, 0.1, 0.5, 0.1),
    )
    g_model = SimpleNamespace(
        fit_status="ready",
        posterior=lambda values: pytest.fail("G must not run without all six F predictions"),
    )
    result = suggest(
        selection(),
        ("A01",),
        {"m01": ConstantModel(("A01",), 0.2, 0.1)},
        g_model,
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [make_result("trial_base")],
        config=optimizer_config(candidate_pool_size=2),
    )

    assert result is not None
    assert result.predictions["g_tiebreak_used"] is False
    assert result.predictions["predicted_tps"] is None


def test_each_f_reads_its_neighbor_slice_from_the_same_full_candidate() -> None:
    class RecordingModel(ConstantModel):
        def __init__(self, feature_order: tuple[str, ...]) -> None:
            super().__init__(feature_order, 0.2, 0.01)
            self.inputs: list[np.ndarray] = []

        def posterior(self, values: np.ndarray) -> Posterior:
            self.inputs.append(values.copy())
            return super().posterior(values)

    first = RecordingModel(("A01", "A02"))
    second = RecordingModel(("A02", "A03"))
    metrics = {metric_id: 0.1 for metric_id in ("m01", "m02", "m03", "m04", "m05", "m06")}
    metrics.update({"m01": 0.3, "m02": 0.3})
    result = suggest(
        selection(("m01", "m02")),
        ("A01", "A02", "A03"),
        {"m01": first, "m02": second},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [make_result("trial_base").model_copy(update={"metrics": metrics})],
        config=optimizer_config(candidate_pool_size=1, max_candidate_pools=1),
    )

    assert result is not None
    np.testing.assert_allclose(first.inputs[0], [[result.coefficients["A01"], result.coefficients["A02"]]])
    np.testing.assert_allclose(second.inputs[0], [[result.coefficients["A02"], result.coefficients["A03"]]])
