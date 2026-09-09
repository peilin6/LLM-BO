from __future__ import annotations

from pathlib import Path

from dibo.compiler import CompileEnvironment, compile_config
from dibo.schemas import load_actions, load_parameters

CONFIGS = Path(__file__).parents[2] / "configs"


def environment(*, allocated_gpu_count: int = 1) -> CompileEnvironment:
    return CompileEnvironment(
        parameter_catalog=load_parameters(CONFIGS / "parameters.yaml"),
        allocated_gpu_count=allocated_gpu_count,
        num_attention_heads=28,
        num_hidden_layers=28,
        max_model_len=8192,
        supported_block_sizes=(8, 16, 32),
        model_context="synthetic-model",
        workload_context="synthetic-workload",
    )


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


def test_a06_votes_enforce_eager_in_both_directions() -> None:
    actions = load_actions(CONFIGS / "actions_v1.yaml")
    positive = compile_config(base_config(), actions, ("A06",), {"A06": 0.8}, environment())
    negative = compile_config(base_config(), actions, ("A06",), {"A06": -0.8}, environment())

    assert positive.categorical_decisions["p15"]["total_vote"] == -1.15
    assert positive.final_config["p15"] is False
    assert negative.categorical_decisions["p15"]["total_vote"] == 0.45
    assert negative.final_config["p15"] is True


def test_vote_equal_to_threshold_keeps_base_value() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A06",),
        {"A06": -0.7},
        environment(),
    )

    assert trace.categorical_decisions["p15"]["total_vote"] == 0.35
    assert trace.final_config["p15"] is False


def test_a07_positive_moves_to_smaller_block() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A07",),
        {"A07": 0.8},
        environment(),
    )

    assert trace.final_config["p05"] == 8
    assert trace.categorical_decisions["p05"]["anchor"] == 16


def test_ordered_vote_equal_to_threshold_keeps_anchor() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A07",),
        {"A07": -0.35},
        environment(),
    )

    assert trace.categorical_decisions["p05"]["total_vote"] == 0.35
    assert trace.final_config["p05"] == 16


def test_single_gpu_keeps_a08_but_tp_at_boundary() -> None:
    trace = compile_config(
        base_config(),
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A08",),
        {"A08": 0.8},
        environment(),
    )

    assert trace.coefficients["A08"] == 0.8
    assert trace.final_config["p01"] == 1
    assert trace.categorical_decisions["p01"]["at_boundary"] is True
