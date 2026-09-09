from __future__ import annotations

from pathlib import Path

from dibo.models import fit_f, fit_g
from dibo.schemas import METRIC_IDS, load_actions, load_graph, load_parameters
from dibo.store import load_trials, save_trial
from dibo.trace_encoder import encode_trace
from tests.unit.test_f_models import training_results
from tests.unit.test_initial_design import BASE_CONFIG

CONFIGS = Path(__file__).parents[2] / "configs"


def test_saved_history_encodes_for_evidence_but_f_uses_neighbor_z(tmp_path: Path) -> None:
    results = training_results()
    for result in results:
        save_trial(result, tmp_path)

    loaded = load_trials(tmp_path)
    encoded = [
        encode_trace(
            result.trace,
            BASE_CONFIG,
            load_parameters(CONFIGS / "parameters.yaml"),
            base_trial_id="trial_base",
        )
        for result in loaded
    ]
    current_bundle = load_actions(CONFIGS / "actions_v1.yaml").model_copy(
        update={"action_version": 2, "parent_action_version": 1}
    )
    f_models = fit_f(loaded, load_graph(CONFIGS / "graph.yaml"), current_bundle)

    assert len(encoded[0].features) == 15
    assert f_models["m01"].feature_order == ("A01", "A02", "A05", "A07")
    assert len(f_models["m01"].feature_order) == 4
    assert f_models["m01"].observation_noise[0] == 2e-6
    assert "features" not in vars(f_models["m01"])


def test_missing_metric_only_removes_relevant_f_and_complete_g_row(tmp_path: Path) -> None:
    results = training_results()
    metrics = dict(results[0].metrics)
    metrics["m05"] = None
    results[0] = results[0].model_copy(update={"metrics": metrics})
    for result in results:
        save_trial(result, tmp_path)

    loaded = load_trials(tmp_path)
    f_models = fit_f(loaded, load_graph(CONFIGS / "graph.yaml"), load_actions(CONFIGS / "actions_v1.yaml"))
    g_model = fit_g(loaded)

    assert f_models["m05"].n_samples == 5
    assert f_models["m01"].n_samples == 6
    assert g_model.n_samples == 5
    assert g_model.feature_order == METRIC_IDS