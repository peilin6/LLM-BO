from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from dibo.llm_update import (
    UpdateValidationError,
    apply_update,
    build_evidence,
    call_llm_api,
    process_review_request,
    read_review_response,
    review_round,
    save_action_bundle,
    validate_update,
    write_review_request,
)
from dibo.schemas import (
    ACTION_IDS,
    ActionUpdate,
    MetricSelection,
    MetricTarget,
    TrialStatus,
    WeightUpdate,
    load_actions,
    load_graph,
)
from tests.unit.test_initial_design import make_result

CONFIGS = Path(__file__).parents[2] / "configs"


def result(
    trial_id: str,
    coefficient: float,
    metric: float,
    *,
    status: TrialStatus = TrialStatus.SUCCESS,
    action_version: int = 1,
):
    base = make_result(trial_id, status=status, phase="bo")
    coefficients = {action_id: 0.0 for action_id in ACTION_IDS}
    coefficients["A01"] = coefficient
    metrics = dict(base.metrics)
    metrics["m01"] = metric if status == TrialStatus.SUCCESS else None
    trace = base.trace.model_copy(
        update={
            "action_version": action_version,
            "selected_metrics": ("m01",),
            "selected_actions": ("A01",),
            "requested_coefficients": {"A01": coefficient},
            "coefficients": coefficients,
            "effective_parameter_delta": {"p03": round(-100 * coefficient)},
        }
    )
    return base.model_copy(
        update={
            "status": status,
            "trace": trace,
            "metrics": metrics,
            "metric_missing_reasons": ({"m01": "benchmark failed"} if status != TrialStatus.SUCCESS else {}),
            "throughput_tps": (base.throughput_tps if status == TrialStatus.SUCCESS else None),
        }
    )


def evidence_fixture(*, conflicting: bool = False):
    first = result("trial_old", 0.4, 0.30)
    second = result("trial_new", 0.8, 0.40 if conflicting else 0.20)
    failed = result("trial_failed", 0.9, 0.0, status=TrialStatus.BENCHMARK_FAILED)
    selection = MetricSelection(
        reference_trial_id="trial_old",
        mode="hard_threshold",
        selected_metrics=("m01",),
        scores={"m01": 1.0},
        weights={"m01": 1.0},
        targets={"m01": MetricTarget(kind="upper", value=0.1, scale=0.1)},
    )
    return build_evidence(
        load_actions(CONFIGS / "actions_v1.yaml"),
        load_graph(CONFIGS / "graph.yaml"),
        selection,
        [first, second, failed],
        [second, failed],
        bo_round=2,
        model_evidence={"F": {"m01": {"fit_status": "ready"}}, "G": {"fit_status": "not_ready"}},
    )


def weight_update(**changes: object) -> WeightUpdate:
    values = {
        "action_id": "A01",
        "parameter_id": "p03",
        "desired_parameter_effect": "decrease",
        "weight_operation": "strengthen",
        "old_weight": -0.85,
        "new_weight": -0.92,
        "target_metrics": ("m01",),
        "evidence_trial_ids": ("trial_old", "trial_new"),
        "reason": "m01 fell in two successful trials with distinct A01 coefficients",
        **changes,
    }
    return WeightUpdate(**values)


def proposal(*items: WeightUpdate, **changes: object) -> ActionUpdate:
    values = {
        "decision": "modify",
        "parent_action_version": 1,
        "effective_from_bo_round": 3,
        "updates": items or (weight_update(),),
        **changes,
    }
    return ActionUpdate(**values)


def test_evidence_contains_all_history_and_marks_actual_round_failures() -> None:
    evidence = evidence_fixture()

    assert {trial.trial_id for trial in evidence.trials} == {
        "trial_old",
        "trial_new",
        "trial_failed",
    }
    assert [trial.trial_id for trial in evidence.trials if trial.current_round] == [
        "trial_failed",
        "trial_new",
    ]
    failed = next(trial for trial in evidence.trials if trial.trial_id == "trial_failed")
    assert failed.status == TrialStatus.BENCHMARK_FAILED
    assert "benchmark failed" in failed.failure_reason
    assert evidence.action_evidence["A01"].adjacent_metrics == ("m01", "m02")
    assert evidence.model_evidence["G"]["fit_status"] == "not_ready"


def test_empty_round_does_not_build_evidence() -> None:
    evidence = evidence_fixture()
    selection = MetricSelection(
        reference_trial_id="trial_old",
        mode="hard_threshold",
        selected_metrics=("m01",),
        scores={"m01": 1.0},
        weights={"m01": 1.0},
        targets={"m01": MetricTarget(kind="upper", value=0.1, scale=0.1)},
    )

    with pytest.raises(ValueError, match="empty round"):
        build_evidence(
            evidence.action_bundle,
            load_graph(CONFIGS / "graph.yaml"),
            selection,
            [],
            [],
            bo_round=2,
        )


def test_legal_modify_creates_next_bundle_without_mutating_parent() -> None:
    bundle = load_actions(CONFIGS / "actions_v1.yaml")
    evidence = evidence_fixture()

    updated = apply_update(bundle, proposal(), evidence, load_graph(CONFIGS / "graph.yaml"))

    assert updated.action_version == 2
    assert updated.parent_action_version == 1
    assert updated.actions[0].direction_vector[2] == -0.92
    assert bundle.actions[0].direction_vector[2] == -0.85


def test_keep_returns_same_bundle() -> None:
    bundle = load_actions(CONFIGS / "actions_v1.yaml")
    keep = ActionUpdate(
        decision="keep",
        parent_action_version=1,
        effective_from_bo_round=3,
        updates=(),
    )

    assert apply_update(bundle, keep, evidence_fixture(), load_graph(CONFIGS / "graph.yaml")) is bundle


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"new_weight": -0.96}, "delta exceeds"),
        ({"old_weight": -0.80}, "old_weight"),
        ({"weight_operation": "weaken"}, "does not match"),
        ({"desired_parameter_effect": "increase"}, "desired_parameter_effect"),
        ({"target_metrics": ("m03",)}, "adjacent"),
        ({"evidence_trial_ids": ("trial_old", "missing")}, "unknown Trial"),
    ],
)
def test_invalid_cell_contracts_are_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(UpdateValidationError, match=message):
        validate_update(
            load_actions(CONFIGS / "actions_v1.yaml"),
            proposal(weight_update(**changes)),
            evidence_fixture(),
            load_graph(CONFIGS / "graph.yaml"),
        )


def test_two_successes_must_have_distinct_action_or_parameter_effects() -> None:
    first = result("trial_old", 0.4, 0.30)
    duplicate = result("trial_new", 0.4, 0.20)
    selection = MetricSelection(
        reference_trial_id="trial_old",
        mode="hard_threshold",
        selected_metrics=("m01",),
        scores={"m01": 1.0},
        weights={"m01": 1.0},
        targets={"m01": MetricTarget(kind="upper", value=0.1, scale=0.1)},
    )
    evidence = build_evidence(
        load_actions(CONFIGS / "actions_v1.yaml"),
        load_graph(CONFIGS / "graph.yaml"),
        selection,
        [first, duplicate],
        [duplicate],
        bo_round=2,
    )

    with pytest.raises(UpdateValidationError, match="distinct coefficients or effects"):
        validate_update(
            load_actions(CONFIGS / "actions_v1.yaml"),
            proposal(),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )


def test_conflicting_observations_require_keep() -> None:
    third = result("trial_third", 0.6, 0.10)
    base = evidence_fixture(conflicting=True)
    successful = next(trial for trial in base.trials if trial.status == TrialStatus.SUCCESS)
    third_evidence = successful.model_copy(
        update={
            "trial_id": third.trial_id,
            "coefficients": third.trace.coefficients,
            "effective_parameter_delta": third.trace.effective_parameter_delta,
            "metrics": third.metrics,
        }
    )
    evidence = base.model_copy(update={"trials": base.trials + (third_evidence,)})
    item = weight_update(evidence_trial_ids=("trial_old", "trial_new", "trial_third"))

    with pytest.raises(UpdateValidationError, match="conflicting observations"):
        validate_update(
            load_actions(CONFIGS / "actions_v1.yaml"),
            proposal(item),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )


def test_failed_trial_cannot_support_strengthen_but_can_support_weaken() -> None:
    evidence = evidence_fixture()
    strengthen = weight_update(
        evidence_trial_ids=("trial_old", "trial_new", "trial_failed")
    )
    with pytest.raises(UpdateValidationError, match="failed Trials"):
        validate_update(
            evidence.action_bundle,
            proposal(strengthen),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )

    weaken = weight_update(
        desired_parameter_effect="increase",
        weight_operation="weaken",
        new_weight=-0.80,
        evidence_trial_ids=("trial_old", "trial_new", "trial_failed"),
    )
    validate_update(
        evidence.action_bundle,
        proposal(weaken),
        evidence,
        load_graph(CONFIGS / "graph.yaml"),
    )


def test_review_limits_and_version_boundaries_are_rejected() -> None:
    evidence = evidence_fixture()
    repeated_action = tuple(
        weight_update(parameter_id=parameter_id, old_weight=old, new_weight=new)
        for parameter_id, old, new in (("p01", 0.30, 0.35), ("p02", 0.15, 0.20), ("p03", -0.85, -0.92))
    )
    with pytest.raises(UpdateValidationError, match="two parameters per Action"):
        validate_update(
            evidence.action_bundle,
            proposal(*repeated_action),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )
    with pytest.raises(UpdateValidationError, match="next BO round"):
        validate_update(
            evidence.action_bundle,
            proposal(effective_from_bo_round=4),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )
    with pytest.raises(UpdateValidationError, match="parent_action_version"):
        validate_update(
            evidence.action_bundle,
            proposal(parent_action_version=2),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )


def test_anchor_non_anchor_and_single_gpu_constraints() -> None:
    evidence = evidence_fixture()
    anchor_bundle = evidence.action_bundle.model_copy(
        update={
            "actions": (
                evidence.action_bundle.actions[0].model_copy(
                    update={
                        "direction_vector": evidence.action_bundle.actions[0].direction_vector[:5]
                        + (0.80,)
                        + evidence.action_bundle.actions[0].direction_vector[6:]
                    }
                ),
                *evidence.action_bundle.actions[1:],
            )
        }
    )
    anchor_evidence = evidence.model_copy(update={"action_bundle": anchor_bundle})
    anchor = weight_update(
        parameter_id="p06",
        desired_parameter_effect="decrease",
        weight_operation="weaken",
        old_weight=0.80,
        new_weight=0.79,
    )
    with pytest.raises(UpdateValidationError, match="anchor magnitude"):
        validate_update(
            anchor_bundle,
            proposal(anchor),
            anchor_evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )

    non_anchor = weight_update(
        parameter_id="p14",
        desired_parameter_effect="increase",
        weight_operation="weaken",
        old_weight=-0.05,
        new_weight=-0.01,
    )
    with pytest.raises(UpdateValidationError, match="at least 0.02"):
        validate_update(
            evidence.action_bundle,
            proposal(non_anchor),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )

    a08 = weight_update(
        action_id="A08",
        parameter_id="p02",
        desired_parameter_effect="increase",
        weight_operation="strengthen",
        old_weight=0.45,
        new_weight=0.50,
        target_metrics=("m03",),
    )
    with pytest.raises(UpdateValidationError, match="single-GPU"):
        validate_update(
            evidence.action_bundle,
            proposal(a08),
            evidence,
            load_graph(CONFIGS / "graph.yaml"),
        )


@pytest.mark.parametrize("failure", [TimeoutError(), "not-json"])
def test_review_failure_calls_once_and_returns_keep(failure: Exception | str) -> None:
    calls = 0

    def reviewer(payload: dict[str, object]):
        nonlocal calls
        calls += 1
        if isinstance(failure, Exception):
            raise failure
        return failure

    update = review_round(evidence_fixture(), reviewer)

    assert calls == 1
    assert update.decision == "keep"
    assert update.updates == ()


def test_atomic_request_response_roundtrip_and_parent_identity(tmp_path: Path) -> None:
    evidence = evidence_fixture()
    request_path = write_review_request(tmp_path, "request_001", evidence)
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))

    assert set(request_payload) == {
        "request_id",
        "schema_version",
        "parent_action_version",
        "payload",
        "status",
    }
    assert request_payload["parent_action_version"] == 1
    response_path = process_review_request(
        request_path,
        tmp_path / "responses",
        lambda payload: proposal().model_dump(mode="json"),
    )
    update = read_review_response(response_path, evidence, request_id="request_001")

    assert update.decision == "modify"
    assert update.parent_action_version == evidence.parent_action_version
    assert not list(tmp_path.rglob("*.tmp"))


def test_response_id_or_parent_mismatch_returns_keep(tmp_path: Path) -> None:
    evidence = evidence_fixture()
    request_path = write_review_request(tmp_path, "request_001", evidence)
    response_path = process_review_request(
        request_path,
        tmp_path / "responses",
        lambda payload: proposal().model_dump(mode="json"),
    )

    assert read_review_response(response_path, evidence, request_id="other").decision == "keep"
    stale = evidence.model_copy(update={"parent_action_version": 2})
    assert read_review_response(response_path, stale, request_id="request_001").decision == "keep"


def test_bad_worker_result_is_one_error_response_without_retry(tmp_path: Path) -> None:
    calls = 0
    request_path = write_review_request(tmp_path, "request_001", evidence_fixture())

    def bad_reviewer(payload: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        return "bad-json"

    response_path = process_review_request(
        request_path,
        tmp_path / "responses",
        bad_reviewer,
    )

    assert calls == 1
    assert json.loads(response_path.read_text(encoding="utf-8"))["status"] == "error"


def test_new_action_yaml_is_loadable_and_never_overwrites_parent(tmp_path: Path) -> None:
    bundle = load_actions(CONFIGS / "actions_v1.yaml")
    updated = apply_update(
        bundle,
        proposal(),
        evidence_fixture(),
        load_graph(CONFIGS / "graph.yaml"),
    )
    parent_path = save_action_bundle(bundle, tmp_path)
    updated_path = save_action_bundle(updated, tmp_path)

    assert parent_path.name == "actions_v1.yaml"
    assert updated_path.name == "actions_v2.yaml"
    assert load_actions(parent_path).actions[0].direction_vector[2] == -0.85
    assert load_actions(updated_path).actions[0].direction_vector[2] == -0.92
    with pytest.raises(FileExistsError):
        save_action_bundle(updated, tmp_path)


def test_fake_http_is_called_once_with_timeout_and_does_not_embed_api_key() -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": proposal().model_dump_json()}}]},
        )

    result = call_llm_api(
        evidence_fixture().model_dump(mode="json"),
        endpoint="https://llm.invalid/review",
        model="fake-model",
        api_key="secret-test-key",
        timeout_s=7.0,
        post=fake_post,
    )

    assert len(calls) == 1
    assert calls[0]["timeout"] == 7.0
    assert calls[0]["headers"] == {"Authorization": "Bearer secret-test-key"}
    assert "secret-test-key" not in json.dumps(calls[0]["json"])
    assert ActionUpdate.model_validate_json(result).decision == "modify"