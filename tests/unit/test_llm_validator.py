from pathlib import Path

import pytest
from pydantic import ValidationError

from dibo.llm_update import ReviewRequest, UpdateValidationError, validate_update
from dibo.schemas import load_graph
from tests.unit.test_llm_update import evidence_fixture, proposal, weight_update

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize(
    "item",
    [
        weight_update(new_weight=-0.95),
        weight_update(
            parameter_id="p14",
            desired_parameter_effect="increase",
            weight_operation="weaken",
            old_weight=-0.05,
            new_weight=-0.02,
        ),
        weight_update(
            parameter_id="p14",
            desired_parameter_effect="increase",
            weight_operation="reverse",
            old_weight=-0.05,
            new_weight=0.05,
        ),
        weight_update(
            parameter_id="p14",
            desired_parameter_effect="increase",
            weight_operation="set_zero",
            old_weight=-0.05,
            new_weight=0.0,
        ),
    ],
)
def test_exact_delta_non_anchor_and_operation_boundaries_are_accepted(item) -> None:
    evidence = evidence_fixture()

    validate_update(
        evidence.action_bundle,
        proposal(item),
        evidence,
        load_graph(CONFIGS / "graph.yaml"),
    )


def test_anchor_exact_point_eight_is_accepted_for_a_future_parent() -> None:
    evidence = evidence_fixture()
    action = evidence.action_bundle.actions[0]
    vector = action.direction_vector[:5] + (0.90,) + action.direction_vector[6:]
    bundle = evidence.action_bundle.model_copy(
        update={
            "actions": (
                action.model_copy(update={"direction_vector": vector}),
                *evidence.action_bundle.actions[1:],
            )
        }
    )
    evidence = evidence.model_copy(update={"action_bundle": bundle})
    item = weight_update(
        parameter_id="p06",
        desired_parameter_effect="decrease",
        weight_operation="weaken",
        old_weight=0.90,
        new_weight=0.80,
    )

    validate_update(bundle, proposal(item), evidence, load_graph(CONFIGS / "graph.yaml"))


def test_more_than_five_cells_is_rejected_before_cell_validation() -> None:
    evidence = evidence_fixture()
    updates = tuple(weight_update(parameter_id=f"p{index:02}") for index in range(1, 7))

    with pytest.raises(UpdateValidationError, match="at most five cells"):
        validate_update(
            evidence.action_bundle,
            proposal(*updates),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )


def test_legacy_parent_version_alias_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewRequest.model_validate(
            {
                "request_id": "request_001",
                "schema_version": 1,
                "parent_version": 1,
                "payload": {},
                "status": "pending",
            }
        )


@pytest.mark.parametrize("operation,new", [("weaken", -0.02), ("set_zero", 0.0)])
def test_neutral_reduces_parameter_influence(operation, new) -> None:
    evidence = evidence_fixture()
    item = weight_update(
        parameter_id="p14",
        old_weight=-0.05,
        new_weight=new,
        weight_operation=operation,
        desired_parameter_effect="neutral",
    )
    validate_update(
        evidence.action_bundle, proposal(item), evidence, load_graph(CONFIGS / "graph.yaml")
    )


def test_missing_metric_cannot_support_a_performance_update() -> None:
    evidence = evidence_fixture()
    trials = tuple(
        item.model_copy(update={"metrics": {**item.metrics, "m01": None}})
        for item in evidence.trials
    )
    evidence = evidence.model_copy(update={"trials": trials})
    with pytest.raises(UpdateValidationError, match="observed finite"):
        validate_update(
            evidence.action_bundle, proposal(), evidence, load_graph(CONFIGS / "graph.yaml")
        )
