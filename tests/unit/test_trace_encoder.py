from __future__ import annotations

from pathlib import Path

import pytest

from dibo.compiler import CompileEnvironment, compile_config
from dibo.schemas import load_actions, load_parameters
from dibo.trace_encoder import encode_trace

CONFIGS = Path(__file__).parents[2] / "configs"


def base_config() -> dict[str, object]:
    return {
        "p01": 1,
        "p02": 1,
        "p03": 128,
        "p04": 8192,
        "p05": 16,
        "p06": 0.90,
        "p07": 4.0,
        "p08": 2.0,
        "p09": 2,
        "p10": 1,
        "p11": 2048,
        "p12": True,
        "p13": True,
        "p14": False,
        "p15": False,
    }


def environment() -> CompileEnvironment:
    return CompileEnvironment(
        parameter_catalog=load_parameters(CONFIGS / "parameters.yaml"),
        allocated_gpu_count=1,
        num_attention_heads=28,
        num_hidden_layers=28,
        max_model_len=8192,
        supported_block_sizes=(8, 16, 32),
        model_context="synthetic-model",
        workload_context="synthetic-workload",
    )


def test_encoding_is_15_dimensional_and_not_clipped() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A02",),
        {"A02": 1.0},
        environment(),
        trial_id="trial_001",
    )
    fixed_base = base_config()
    fixed_base["p03"] = 32
    encoded = encode_trace(
        trace,
        fixed_base,
        environment().parameter_catalog,
        base_trial_id="trial_base",
    )

    assert len(encoded.features) == 15
    assert encoded.features[2] > 1.0
    assert encoded.base_trial_id == "trial_base"


def test_same_final_config_across_versions_has_same_features() -> None:
    actions = load_actions(CONFIGS / "actions_v1.yaml")
    first = compile_config(base_config(), actions, (), {}, environment(), trial_id="trial_001")
    second = first.model_copy(
        update={"trial_id": "trial_002", "action_version": 2, "coefficients": dict(first.coefficients)}
    )

    first_encoded = encode_trace(
        first, base_config(), environment().parameter_catalog, base_trial_id="trial_base"
    )
    second_encoded = encode_trace(
        second, base_config(), environment().parameter_catalog, base_trial_id="trial_base"
    )

    assert first_encoded.features == second_encoded.features
    assert first_encoded.effective_parameter_delta == second_encoded.effective_parameter_delta


def test_same_z_with_different_final_config_has_different_features() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A02",),
        {"A02": 0.8},
        environment(),
        trial_id="trial_001",
    )
    changed = dict(trace.final_config)
    changed["p03"] += 10
    other = trace.model_copy(update={"trial_id": "trial_002", "final_config": changed})

    first_encoded = encode_trace(
        trace, base_config(), environment().parameter_catalog, base_trial_id="trial_base"
    )
    other_encoded = encode_trace(
        other, base_config(), environment().parameter_catalog, base_trial_id="trial_base"
    )

    assert first_encoded.features != other_encoded.features


def test_invalid_trace_is_not_reproducible_evidence() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A01",),
        {"A02": 0.5},
        environment(),
    )

    with pytest.raises(ValueError, match="invalid compile traces"):
        encode_trace(trace, base_config(), environment().parameter_catalog, base_trial_id="base")
