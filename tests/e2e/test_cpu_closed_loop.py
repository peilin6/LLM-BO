from dataclasses import replace

import pytest

from dibo.compiler import compile_config
from dibo.controller import FakeTrialRunner, run_experiment, synthetic_metrics
from dibo.optimizer import metric_loss
from dibo.report import write_report
from dibo.schemas import ExperimentConfig, MetricSelection, RunMode, TrialStatus, load_actions
from dibo.selector import union_neighbor_actions
from dibo.store import load_trials
from tests.unit.test_controller import FailingEngine, cpu_config, improving_review


@pytest.mark.asyncio
async def test_cpu_22_trial_closed_loop_with_real_algorithms(tmp_path) -> None:
    config = cpu_config(tmp_path, budget=22, name="complete")
    payload = config.experiment.model_dump(mode="json")
    payload["tuning"].update(candidate_pool_size=64, mc_samples=128)
    config = replace(config, experiment=ExperimentConfig.model_validate(payload))
    summary = await run_experiment(config, reviewer=improving_review)
    assert summary.attempted_trials == 22
    assert len(summary.selections) == len(summary.reviews) == 4
    assert len(load_trials(config.run_dir)) == 22
    assert all(item.run_mode == RunMode.SYNTHETIC for item in summary.history)
    assert len({item.trace.config_hash for item in summary.history}) == 22
    assert summary.base_trial == max(summary.history[:10], key=lambda item: item.throughput_tps)
    assert summary.best_trial.throughput_tps >= summary.base_trial.throughput_tps
    modes = {item["selection"]["mode"] for item in summary.selections}
    assert modes & {"hard_threshold", "g_counterfactual"}
    dimensions = set()
    sources = set()
    improved_metric_distance = False
    for record in summary.selections:
        selection = MetricSelection.model_validate(record["selection"])
        reference = next(
            item for item in summary.history if item.trial_id == selection.reference_trial_id
        )

        def distance(item, selection=selection):
            return float(
                metric_loss(
                    {metric: item.metrics[metric] for metric in selection.selected_metrics},
                    selection.targets,
                    selection.weights,
                )
            )

        neighbors = union_neighbor_actions(selection.selected_metrics, config.graph)
        bundle = load_actions(
            config.run_dir / "action_versions" / f"actions_v{record['action_version']}.yaml"
        )
        assert tuple(record["selected_actions"]) == neighbors
        for suggested in record["suggestions"]:
            sources.add(suggested["source"])
            dimensions.add(suggested["bo_dimension"])
            assert all(
                value == 0
                for action, value in suggested["coefficients"].items()
                if action not in neighbors
            )
            replay = compile_config(
                summary.base_trial.trace.final_config,
                bundle,
                neighbors,
                suggested["coefficients"],
                config.environment,
            )
            executed = next(
                item for item in summary.history if item.trial_id == suggested["trial_id"]
            )
            assert executed.trace.config_hash == replay.config_hash
            assert executed.metrics == synthetic_metrics(replay.final_config)
            assert executed.trace.base_trial_id == summary.base_trial.trial_id
            for metric, error in suggested["predictions"]["f_error"].items():
                assert error == pytest.approx(
                    executed.metrics[metric] - suggested["predictions"]["f_mean"][metric]
                )
            if selection.targets and distance(executed) < distance(reference):
                improved_metric_distance = True
        assert [
            len(record["models"][metric]["feature_order"]) for metric in config.graph.metric_order
        ] == [4, 3, 5, 4, 3, 5]
    assert "metric_ei" in sources
    assert len(dimensions) >= 2
    assert improved_metric_distance
    assert any(item["update"]["decision"] == "modify" for item in summary.reviews)
    report = write_report(config.run_dir).read_text()
    assert "Attempts: 22" in report and "not vLLM measurements" in report


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["g", "hard", "not_ready", "zero_ei", "failure"])
async def test_controlled_cpu_fallback_and_failure_scenarios(tmp_path, case) -> None:
    config = cpu_config(tmp_path, budget=14, name=case)
    payload = config.experiment.model_dump(mode="json")
    if case == "g":
        payload["thresholds"] = {}
    if case == "hard":
        payload["thresholds"] = {
            "m01": {"kind": "upper", "trigger": 0.1, "target": 0.05, "scale": 0.1}
        }
    if case == "not_ready":
        payload["models"]["min_train_samples"] = 100
    if case == "zero_ei":
        payload["thresholds"] = {"m01": {"kind": "upper", "trigger": 0, "target": 1, "scale": 0.1}}
    config = replace(config, experiment=ExperimentConfig.model_validate(payload))
    normal = FakeTrialRunner()
    failing = FakeTrialRunner(engine=FailingEngine())

    async def runner(spec):
        if case == "failure" and spec.trace.trial_id == "trial_012":
            return await failing(spec)
        return await normal(spec)

    summary = await run_experiment(config, runner=runner)
    assert summary.attempted_trials == 14
    sources = {point["source"] for record in summary.selections for point in record["suggestions"]}
    if case == "g":
        assert any(
            record["selection"]["mode"] == "g_counterfactual" for record in summary.selections
        )
    elif case == "hard":
        assert all(record["selection"]["mode"] == "hard_threshold" for record in summary.selections)
    elif case == "not_ready":
        assert sources == {"sobol_exploration"}
    elif case == "zero_ei":
        assert sources == {"uncertainty_exploration"}
    else:
        assert sum(item.status != TrialStatus.SUCCESS for item in summary.history) == 1
        assert len(load_trials(config.run_dir)) == 14
