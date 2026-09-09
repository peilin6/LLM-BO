from __future__ import annotations

from pathlib import Path

import pytest

from dibo.compiler import CompileEnvironment, compile_config
from dibo.schemas import load_actions, load_parameters

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.fixture
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


@pytest.fixture
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


def test_zero_vector_restores_base_config(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    trace = compile_config(
        base_config,
        load_actions(CONFIGS / "actions_v1.yaml"),
        (),
        {},
        environment,
    )

    assert trace.final_config == base_config
    assert tuple(trace.coefficients.values()) == (0.0,) * 8


def test_a06_numeric_hand_calculation(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    trace = compile_config(
        base_config,
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A06",),
        {"A06": 0.8},
        environment,
    )

    assert trace.numeric_decisions["p03"]["raw_value"] == pytest.approx(143.36)
    assert trace.final_config["p03"] == 143
    assert trace.numeric_decisions["p04"]["raw_value"] == pytest.approx(8519.68)
    assert trace.numeric_decisions["p04"]["rounded_value"] == 8448
    assert trace.final_config["p04"] == 8448


def test_all_contributions_are_summed_before_rounding(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    actions = load_actions(CONFIGS / "actions_v1.yaml")
    combined = compile_config(
        base_config,
        actions,
        ("A01", "A02"),
        {"A01": 0.5, "A02": 0.5},
        environment,
    )

    assert combined.numeric_decisions["p03"]["raw_value"] == pytest.approx(132.8)
    assert combined.final_config["p03"] == 133


def test_candidates_do_not_accumulate(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    actions = load_actions(CONFIGS / "actions_v1.yaml")
    first = compile_config(base_config, actions, ("A02",), {"A02": 0.8}, environment)
    second = compile_config(base_config, actions, ("A02",), {"A02": 0.8}, environment)

    assert first.final_config == second.final_config
    assert first.config_hash == second.config_hash
    assert base_config["p03"] == 128


def test_numeric_values_are_clipped_to_bounds(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    near_limit = dict(base_config)
    near_limit["p06"] = 0.98
    trace = compile_config(
        near_limit,
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A01",),
        {"A01": 1.0},
        environment,
    )

    assert trace.final_config["p06"] == 0.98
    assert trace.numeric_decisions["p06"]["raw_value"] == pytest.approx(1.02)
    assert trace.numeric_decisions["p06"]["clipped_value"] == pytest.approx(0.98)


def test_unselected_coefficients_are_zeroed_and_reported(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    trace = compile_config(
        base_config,
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A01",),
        {"A01": 0.2, "A02": 0.4},
        environment,
    )

    assert trace.coefficients["A01"] == 0.2
    assert trace.coefficients["A02"] == 0.0
    assert trace.constraint_errors == ("unselected action A02 requested non-zero coefficient",)


def test_invalid_coefficient_is_rejected(
    environment: CompileEnvironment, base_config: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="finite in"):
        compile_config(
            base_config,
            load_actions(CONFIGS / "actions_v1.yaml"),
            ("A01",),
            {"A01": float("nan")},
            environment,
        )
