from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dibo.schemas import (
    ACTION_IDS,
    METRIC_IDS,
    PARAMETER_IDS,
    ActionBundle,
    ExperimentConfig,
    RunMode,
    load_actions,
    load_experiment,
    load_graph,
    load_parameters,
)

CONFIGS = Path(__file__).parents[2] / "configs"


def test_load_all_configuration_models() -> None:
    parameters = load_parameters(CONFIGS / "parameters.yaml")
    actions = load_actions(CONFIGS / "actions_v1.yaml")
    graph = load_graph(CONFIGS / "graph.yaml")
    real = load_experiment(CONFIGS / "experiment.yaml")
    smoke = load_experiment(CONFIGS / "experiment_smoke.yaml")

    assert parameters.parameter_order == PARAMETER_IDS
    assert actions.action_order == ACTION_IDS
    assert graph.metric_order == METRIC_IDS
    assert real.run_mode == RunMode.REAL
    assert smoke.run_mode == RunMode.SYNTHETIC


def test_unknown_fields_are_rejected() -> None:
    payload = yaml.safe_load((CONFIGS / "actions_v1.yaml").read_text())
    payload["active"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ActionBundle.model_validate(payload)


def test_duplicate_or_reordered_ids_are_rejected() -> None:
    payload = yaml.safe_load((CONFIGS / "actions_v1.yaml").read_text())
    payload["action_order"][1] = "A01"

    with pytest.raises(ValidationError, match="canonical order"):
        ActionBundle.model_validate(payload)


def test_synthetic_mode_rejects_real_adapters() -> None:
    payload = yaml.safe_load((CONFIGS / "experiment_smoke.yaml").read_text())
    payload["engine"]["adapter"] = "vllm_subprocess"

    with pytest.raises(ValidationError, match="synthetic mode requires the fake engine"):
        ExperimentConfig.model_validate(payload)


def test_real_placeholders_are_rejected_only_for_execution() -> None:
    config = load_experiment(CONFIGS / "experiment.yaml")

    with pytest.raises(ValueError, match="contains placeholders"):
        config.validate_for_execution()


def test_synthetic_execution_validation_has_no_external_side_effects() -> None:
    config = load_experiment(CONFIGS / "experiment_smoke.yaml", for_execution=True)

    assert config.engine.adapter.value == "fake"
    assert config.metrics.adapter.value == "fake"
    assert config.llm.transport.value == "fake"
