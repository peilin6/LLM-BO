import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from dibo import controller
from dibo.controller import FakeTrialRunner, load_controller_config, run_experiment
from dibo.llm_update import ReviewAbortedError
from dibo.schemas import ExperimentConfig, TrialStatus, load_experiment

CONFIGS = Path(__file__).parents[2] / "configs"


def test_schedule_allows_budget_truncation() -> None:
    payload = load_experiment(CONFIGS / "experiment_smoke.yaml").model_dump(mode="json")
    payload["tuning"]["max_total_trials"] = 14
    assert ExperimentConfig.model_validate(payload).tuning.max_total_trials == 14
    payload["tuning"]["max_total_trials"] = 23
    with pytest.raises(ValidationError, match="must not exceed"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.asyncio
async def test_initial_run_persists_once_and_refuses_existing_run(tmp_path) -> None:
    config = load_controller_config(CONFIGS / "experiment_smoke.yaml", run_root=tmp_path)
    summary = await run_experiment(config, initial_only=True, runner=FakeTrialRunner())
    assert summary.attempted_trials == 10
    assert len(list(config.run_dir.glob("trials/*/result.json"))) == 10
    assert summary.base_trial.throughput_tps == max(item.throughput_tps for item in summary.history)
    assert summary.selections == []
    events = [json.loads(line) for line in (config.run_dir / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "trial_completed" for event in events) == 10
    assert events[-1]["event"] == "experiment_finished"
    with pytest.raises(FileExistsError):
        await run_experiment(config)


def cpu_config(tmp_path, *, budget=14, initial=10, name="cpu_test"):
    config = load_controller_config(CONFIGS / "experiment_smoke.yaml", run_root=tmp_path)
    payload = config.experiment.model_dump(mode="json")
    payload["experiment_id"] = name
    payload["tuning"].update(
        max_total_trials=budget, initial_trials=initial, candidate_pool_size=8, mc_samples=8
    )
    return replace(
        config, experiment=ExperimentConfig.model_validate(payload), run_dir=tmp_path / name
    )


def improving_review(payload):
    update = {
        "decision": "keep",
        "parent_action_version": payload["parent_action_version"],
        "effective_from_bo_round": payload["bo_round"] + 1,
        "updates": [],
    }
    for action in payload["action_bundle"]["actions"]:
        action_id = action["action_id"]
        trials = [
            item
            for item in payload["trials"]
            if item["status"] == "success" and item["coefficients"][action_id] > 0
        ]
        trials.sort(key=lambda item: item["coefficients"][action_id])
        old = action["direction_vector"][2]
        if (
            len(trials) < 2
            or trials[0]["coefficients"][action_id] == trials[-1]["coefficients"][action_id]
        ):
            continue
        if "p03" in action["anchor_parameter_ids"] or abs(old) > 0.97:
            continue
        update.update(
            decision="modify",
            updates=[
                {
                    "action_id": action_id,
                    "parameter_id": "p03",
                    "old_weight": old,
                    "new_weight": round(old + (0.02 if old > 0 else -0.02), 2),
                    "desired_parameter_effect": "increase" if old > 0 else "decrease",
                    "weight_operation": "strengthen",
                    "target_metrics": [
                        payload["action_evidence"][action_id]["adjacent_metrics"][0]
                    ],
                    "evidence_trial_ids": [trials[0]["trial_id"], trials[-1]["trial_id"]],
                    "reason": "Synthetic fixture with distinct positive coefficients",
                }
            ],
        )
        break
    return update


@pytest.mark.asyncio
async def test_budget_tail_fixed_base_and_next_round_update(tmp_path) -> None:
    config = cpu_config(tmp_path)
    summary = await run_experiment(config, reviewer=improving_review)
    assert summary.attempted_trials == 14
    assert [len(item["suggestions"]) for item in summary.selections] == [3, 1]
    assert [item["action_version"] for item in summary.selections] == [1, 2]
    assert summary.reviews[0]["update"]["decision"] == "modify"
    assert len(summary.reviews[1]["evidence"]["trials"]) == 14
    base = summary.base_trial
    for result in summary.history[10:]:
        assert result.trace.base_trial_id == base.trial_id
        for parameter, decision in result.trace.numeric_decisions.items():
            assert decision["base_value"] == base.trace.final_config[parameter]
    assert (config.run_dir / "action_versions/actions_v1.yaml").exists()
    assert summary.selections[1]["models"]["m01"]["observation_noise"][0] == 2e-6


class FailingEngine:
    async def start(self, config, run_dir):
        raise RuntimeError("synthetic startup failure")


@pytest.mark.asyncio
async def test_no_success_stops_after_counted_initial_failures(tmp_path) -> None:
    config = cpu_config(tmp_path, budget=5, initial=5)
    summary = await run_experiment(config, runner=FakeTrialRunner(engine=FailingEngine()))
    assert summary.attempted_trials == 5
    assert summary.base_trial is None
    assert summary.stop_reason == "no_successful_initial_trial"
    assert summary.reviews == []
    assert all(item.status == TrialStatus.STARTUP_FAILED for item in summary.history)
    assert len(list(config.run_dir.glob("trials/*/result.json"))) == 5


@pytest.mark.asyncio
async def test_empty_candidate_stops_without_review_or_extra_attempt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(controller, "suggest", lambda *args, **kwargs: None)
    summary = await run_experiment(cpu_config(tmp_path))
    assert summary.attempted_trials == 10
    assert summary.stop_reason == "no_candidate"
    assert not summary.reviews


@pytest.mark.asyncio
async def test_partial_failure_counts_once_and_invalid_review_keeps(tmp_path) -> None:
    normal = FakeTrialRunner()
    failing = FakeTrialRunner(engine=FailingEngine())

    async def runner(spec):
        return await (failing(spec) if spec.trace.trial_id == "trial_012" else normal(spec))

    summary = await run_experiment(
        cpu_config(tmp_path), runner=runner, reviewer=lambda payload: "bad json"
    )
    assert summary.attempted_trials == 14
    assert sum(item.status != TrialStatus.SUCCESS for item in summary.history) == 1
    assert all(item["update"]["decision"] == "keep" for item in summary.reviews)
    assert all(item["action_version"] == 1 for item in summary.selections)


def test_unresolved_real_profile_cannot_start(tmp_path) -> None:
    with pytest.raises(ValueError, match="placeholders"):
        load_controller_config(CONFIGS / "experiment.yaml", run_root=tmp_path)


@pytest.mark.asyncio
async def test_cleanup_failure_prevents_next_trial(tmp_path) -> None:
    class CleanupFailure(controller._FakeEngine):
        async def stop(self, handle):
            raise RuntimeError("synthetic cleanup failure")

    summary = await run_experiment(
        cpu_config(tmp_path), runner=FakeTrialRunner(engine=CleanupFailure())
    )
    assert summary.attempted_trials == 1
    assert summary.stop_reason == "cleanup_failed"
    assert not summary.reviews


@pytest.mark.asyncio
async def test_budget_can_stop_inside_initial_design(tmp_path) -> None:
    summary = await run_experiment(cpu_config(tmp_path, budget=3))
    assert summary.attempted_trials == 3
    assert len(summary.history) == 3
    assert summary.base_trial in summary.history
    assert not summary.selections


@pytest.mark.asyncio
async def test_stop_file_ends_after_safe_trial_boundary(tmp_path) -> None:
    config = cpu_config(tmp_path, budget=14)
    runner = FakeTrialRunner()

    async def stopping_runner(spec):
        result = await runner(spec)
        if spec.trace.trial_id == "trial_001":
            (config.run_dir / "STOP").write_text("debug stop\n")
        return result

    summary = await run_experiment(config, runner=stopping_runner)
    events = [json.loads(line) for line in (config.run_dir / "events.jsonl").read_text().splitlines()]
    assert summary.attempted_trials == 1
    assert summary.stop_reason == "stop_requested"
    assert any(event["event"] == "stop_requested" for event in events)


@pytest.mark.asyncio
async def test_llm_stop_ends_after_current_round_without_next_round(tmp_path) -> None:
    config = cpu_config(tmp_path, budget=14)

    def abort_reviewer(payload):
        raise ReviewAbortedError("operator requested stop while waiting for LLM response")

    summary = await run_experiment(config, reviewer=abort_reviewer)
    events = [json.loads(line) for line in (config.run_dir / "events.jsonl").read_text().splitlines()]

    assert summary.attempted_trials == 13
    assert summary.stop_reason == "stop_requested"
    assert len(summary.selections) == 1
    assert any(event["event"] == "experiment_stop_during_llm_wait" for event in events)
