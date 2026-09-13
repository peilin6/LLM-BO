from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dibo.compiler import CompileEnvironment, compile_config
from dibo.schemas import load_actions, load_parameters

CONFIGS = Path(__file__).parents[2] / "configs"


def environment() -> CompileEnvironment:
    return CompileEnvironment(
        parameter_catalog=load_parameters(CONFIGS / "parameters.yaml"),
        allocated_gpu_count=4,
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


def compile_zero(config: dict[str, object], env: CompileEnvironment | None = None):
    return compile_config(
        config,
        load_actions(CONFIGS / "actions_v1.yaml"),
        (),
        {},
        env or environment(),
    )


def test_joint_parallel_and_model_constraints_are_reported() -> None:
    config = base_config()
    config["p01"] = 4
    config["p02"] = 2
    env = replace(environment(), num_attention_heads=30, num_hidden_layers=27)
    trace = compile_zero(config, env)

    assert "p01 * p02 exceeds allocated_gpu_count" in trace.constraint_errors
    assert "num_attention_heads is not divisible by p01" in trace.constraint_errors
    assert "num_hidden_layers is not divisible by p02" in trace.constraint_errors


def test_batch_and_partial_prefill_constraints_are_reported() -> None:
    config = base_config()
    config["p03"] = 900
    config["p04"] = 512
    config["p09"] = 1
    config["p10"] = 4
    trace = compile_zero(config)

    assert "p04 must be greater than or equal to p03" in trace.constraint_errors
    assert "p10 must be between 1 and p09" in trace.constraint_errors


def test_vllm_0112_rejects_concurrent_partial_prefill() -> None:
    trace = compile_zero(base_config(), replace(environment(), engine_version="0.11.2"))

    assert (
        "p09 must equal 1 in vLLM 0.11.2 because concurrent partial prefill is unsupported"
        in trace.constraint_errors
    )


def test_unsupported_parameter_and_block_size_are_reported() -> None:
    config = base_config()
    config["p05"] = 16
    env = replace(
        environment(),
        supported_block_sizes=(8, 32),
        supported_parameter_ids=tuple(item for item in environment().supported_parameter_ids if item != "p13"),
    )
    trace = compile_zero(config, env)

    assert "base value for p05 is not a legal ordered choice" in trace.constraint_errors
    assert "p05 is not supported by the current backend" in trace.constraint_errors
    assert "unsupported parameter: p13" in trace.constraint_errors


def test_config_hash_uses_final_config_and_fixed_context() -> None:
    config = base_config()
    first = compile_zero(config)
    same = compile_zero(config)
    other_context = compile_zero(config, replace(environment(), workload_context="other"))

    assert first.config_hash == same.config_hash
    assert first.config_hash != other_context.config_hash


def test_valid_trace_derives_validity_from_constraint_errors() -> None:
    assert compile_zero(base_config()).valid is True
