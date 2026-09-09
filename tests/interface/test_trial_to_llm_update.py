from pathlib import Path

import pytest

from dibo.compiler import compile_config
from dibo.llm_update import UpdateValidationError, apply_update, build_evidence, validate_update
from dibo.models import fit_f
from dibo.schemas import (
    ActionUpdate,
    MetricSelection,
    MetricTarget,
    TrialStatus,
    load_actions,
    load_graph,
)
from tests.unit.test_initial_design import BASE_CONFIG
from tests.unit.test_llm_update import proposal, result
from tests.unit.test_optimizer import optimizer_config

CONFIGS = Path(__file__).parents[2] / "configs"


def selection(metric_id: str) -> MetricSelection:
    return MetricSelection(
        reference_trial_id="trial_early_1",
        mode="hard_threshold",
        selected_metrics=(metric_id,),
        scores={metric_id: 1.0},
        weights={metric_id: 1.0},
        targets={metric_id: MetricTarget(kind="upper", value=0.1, scale=0.1)},
    )


def test_if04_history_review_next_bundle_and_model_handoff() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")
    current_bundle = load_actions(CONFIGS / "actions_v1.yaml")
    history = [
        result("trial_early_1", 0.2, 0.35),
        result("trial_early_2", 0.4, 0.30),
        result("trial_round_1", 0.6, 0.22),
        result("trial_round_2", 0.8, 0.16),
        result("trial_round_failed", 0.9, 0.0, status=TrialStatus.BENCHMARK_FAILED),
    ]
    history = [
        item.model_copy(
            update={
                "trace": item.trace.model_copy(
                    update={
                        "base_trial_id": "trial_base",
                        "compile_base_id": "trial_base",
                        "config_hash": f"hash-{item.trial_id}",
                    }
                )
            }
        )
        for item in history
    ]
    round_results = history[-3:]
    evidence = build_evidence(
        current_bundle,
        graph,
        selection("m01"),
        history,
        round_results,
        bo_round=2,
    )

    assert len(evidence.trials) == 5
    assert sum(trial.current_round for trial in evidence.trials) == 3
    failed = next(trial for trial in evidence.trials if trial.trial_id == "trial_round_failed")
    assert failed.status == TrialStatus.BENCHMARK_FAILED
    assert failed.throughput_tps is None

    update = proposal().model_copy(
        update={
            "updates": (
                proposal().updates[0].model_copy(
                    update={"evidence_trial_ids": ("trial_early_1", "trial_round_1", "trial_round_2")}
                ),
            )
        }
    )
    next_bundle = apply_update(current_bundle, update, evidence, graph)

    assert current_bundle.action_version == 1
    assert current_bundle.actions[0].direction_vector[2] == -0.85
    assert next_bundle.action_version == 2
    assert next_bundle.actions[0].direction_vector[2] == -0.92

    coefficients = {action_id: 0.0 for action_id in current_bundle.action_order}
    coefficients["A01"] = 0.8
    environment = optimizer_config().environment
    current_trace = compile_config(
        BASE_CONFIG,
        current_bundle,
        ("A01",),
        coefficients,
        environment,
        trial_id="current_round_candidate",
        phase="bo",
        bo_round=2,
        compile_base_id="trial_base",
        base_trial_id="trial_base",
        selected_metrics=("m01",),
    )
    next_trace = compile_config(
        BASE_CONFIG,
        next_bundle,
        ("A01",),
        coefficients,
        environment,
        trial_id="next_round_candidate",
        phase="bo",
        bo_round=3,
        compile_base_id="trial_base",
        base_trial_id="trial_base",
        selected_metrics=("m02",),
    )

    assert current_trace.base_trial_id == next_trace.base_trial_id == "trial_base"
    assert current_trace.action_version == 1
    assert next_trace.action_version == 2
    assert current_trace.final_config["p03"] != next_trace.final_config["p03"]

    models = fit_f(
        history,
        graph,
        next_bundle,
        min_train_samples=2,
        min_unique_configs=2,
    )
    assert models["m01"].training_trial_ids == (
        "trial_early_1",
        "trial_early_2",
        "trial_round_1",
        "trial_round_2",
    )
    assert models["m01"].observation_noise == (2e-6, 2e-6, 2e-6, 2e-6)


def test_if04_conflicting_or_invalid_response_keeps_current_bundle() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")
    bundle = load_actions(CONFIGS / "actions_v1.yaml")
    first = result("trial_1", 0.2, 0.30)
    second = result("trial_2", 0.5, 0.10)
    third = result("trial_3", 0.8, 0.40)
    evidence = build_evidence(
        bundle,
        graph,
        selection("m01"),
        [first, second, third],
        [first, second, third],
        bo_round=2,
    )
    conflicting = proposal().model_copy(
        update={
            "updates": (
                proposal().updates[0].model_copy(
                    update={"evidence_trial_ids": ("trial_1", "trial_2", "trial_3")}
                ),
            )
        }
    )

    with pytest.raises(UpdateValidationError, match="conflicting observations"):
        validate_update(bundle, conflicting, evidence, graph)

    keep = ActionUpdate(
        decision="keep",
        parent_action_version=1,
        effective_from_bo_round=3,
        updates=(),
    )
    assert apply_update(bundle, keep, evidence, graph) is bundle