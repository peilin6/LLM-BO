from dibo.schemas import TrialStatus
from tests.unit.test_llm_update import evidence_fixture


def test_evidence_has_per_action_execution_slices_and_relative_outcomes() -> None:
    evidence = evidence_fixture()
    action = evidence.action_evidence["A01"]

    assert action.adjacent_metrics == ("m01", "m02")
    assert {trial.trial_id for trial in action.trials} == {
        "trial_old",
        "trial_new",
        "trial_failed",
    }
    assert all(trial.coefficient != 0.0 for trial in action.trials)
    assert all(set(trial.adjacent_metric_values) == {"m01", "m02"} for trial in action.trials)
    failed = next(trial for trial in action.trials if trial.status != TrialStatus.SUCCESS)
    assert failed.failure_reason is not None
    assert evidence.relative_outcomes["m01"] == {
        "reference": 0.3,
        "minimum_delta": -0.09999999999999998,
        "maximum_delta": 0.0,
    }


def test_evidence_contains_current_matrix_semantics_anchors_and_full_trace() -> None:
    evidence = evidence_fixture()

    assert len(evidence.action_bundle.actions) == 8
    assert evidence.action_bundle.actions[0].anchor_parameter_ids == ("p06",)
    assert evidence.action_bundle.actions[0].positive_semantics == "expand KV headroom"
    assert all(len(trial.final_config) == 15 for trial in evidence.trials)
    assert all(trial.compile_trace["action_version"] == 1 for trial in evidence.trials)
