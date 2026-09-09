from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dibo.compiler import CompileEnvironment, compile_config
from dibo.initial_design import (
    InitialDesignConfig,
    InitialDesignError,
    choose_base_trial,
    generate_initial_candidates,
)
from dibo.schemas import (
    ACTION_IDS,
    RunMode,
    TrialResult,
    TrialStatus,
    load_actions,
    load_parameters,
)

CONFIGS = Path(__file__).parents[2] / "configs"
BASE_CONFIG = {
    "p01": 1,
    "p02": 1,
    "p03": 128,
    "p04": 8192,
    "p05": 16,
    "p06": 0.9,
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


def real_compiler() -> Callable[[dict[str, float], int], object]:
    actions = load_actions(CONFIGS / "actions_v1.yaml")
    environment = CompileEnvironment(
        load_parameters(CONFIGS / "parameters.yaml"),
        8,
        32,
        32,
        8192,
        (8, 16, 32),
        model_context="synthetic-model",
        workload_context="synthetic-workload",
    )

    def compile_point(coefficients: dict[str, float], sequence_number: int):
        selected = tuple(action_id for action_id, value in coefficients.items() if value != 0.0)
        return compile_config(
            BASE_CONFIG,
            actions,
            selected,
            coefficients,
            environment,
            trial_id=f"trial_{sequence_number:03}",
            phase="initial",
            bo_round=None,
            compile_base_id="x_init",
        )

    return compile_point


def make_result(
    trial_id: str,
    *,
    tps: float = 100.0,
    ttft: float | None = 0.5,
    status: TrialStatus = TrialStatus.SUCCESS,
    successful_requests: int = 10,
    phase: str = "initial",
) -> TrialResult:
    trace = real_compiler()({action_id: 0.0 for action_id in ACTION_IDS}, 1).model_copy(
        update={"trial_id": trial_id, "phase": phase}
    )
    metrics = {metric_id: 0.1 for metric_id in ("m01", "m02", "m03", "m04", "m05")}
    metrics["m06"] = ttft
    return TrialResult(
        trial_id=trial_id,
        run_mode=RunMode.SYNTHETIC,
        status=status,
        trace=trace,
        metrics=metrics,
        metric_missing_reasons=({"m06": "unavailable"} if ttft is None else {}),
        throughput_tps=(tps if status == TrialStatus.SUCCESS else None),
        request_count=10,
        completed_requests=successful_requests,
        successful_requests=successful_requests,
        duration_s=1.0,
        sampling_interval_s=1.0,
        valid_sample_counts={},
        log_paths={},
        cleanup_result="stopped",
    )


def test_default_order_is_zero_eight_positive_probes_and_sobol() -> None:
    candidates = generate_initial_candidates(InitialDesignConfig(10, 0.8, 42), real_compiler())

    assert candidates[0].source == "zero"
    assert tuple(candidates[0].coefficients.values()) == (0.0,) * 8
    for index, action_id in enumerate(ACTION_IDS, start=1):
        assert candidates[index].source == f"probe_{action_id}"
        assert candidates[index].coefficients[action_id] == 0.8
        assert sum(value != 0.0 for value in candidates[index].coefficients.values()) == 1
    assert candidates[9].source == "sobol"
    assert any(value < 0 for value in candidates[9].coefficients.values())


def test_short_design_keeps_zero_and_uses_seeded_sobol() -> None:
    first = generate_initial_candidates(InitialDesignConfig(5, 0.8, 7), real_compiler())
    second = generate_initial_candidates(InitialDesignConfig(5, 0.8, 7), real_compiler())

    assert first[0].source == "zero"
    assert all(candidate.source == "sobol" for candidate in first[1:])
    assert [candidate.coefficients for candidate in first] == [candidate.coefficients for candidate in second]


def test_duplicate_final_hash_and_invalid_trace_get_bounded_replacements() -> None:
    compile_real = real_compiler()
    calls = 0

    def compiler(coefficients: dict[str, float], sequence_number: int):
        nonlocal calls
        calls += 1
        trace = compile_real(coefficients, sequence_number)
        if calls == 2:
            return trace.model_copy(update={"config_hash": "forced-duplicate"})
        if calls == 3:
            return trace.model_copy(update={"config_hash": "forced-duplicate"})
        if calls == 4:
            return trace.model_copy(update={"constraint_errors": ("invalid",)})
        return trace

    candidates = generate_initial_candidates(InitialDesignConfig(5, 0.8, 42, 20), compiler)

    assert len(candidates) == 5
    assert len({candidate.trace.config_hash for candidate in candidates}) == 5
    assert calls > 5


def test_replacement_exhaustion_has_explicit_error() -> None:
    compile_real = real_compiler()

    def duplicate_compiler(coefficients: dict[str, float], sequence_number: int):
        return compile_real(coefficients, sequence_number).model_copy(update={"config_hash": "same"})

    with pytest.raises(InitialDesignError, match="exhausted after 5 attempts"):
        generate_initial_candidates(InitialDesignConfig(5, 0.8, 42, 5), duplicate_compiler)


def test_base_selection_filters_failures_and_low_success_rate() -> None:
    results = [
        make_result("trial_failed", status=TrialStatus.BENCHMARK_FAILED),
        make_result("trial_partial", tps=300.0, successful_requests=9),
        make_result("trial_valid", tps=100.0),
    ]

    assert choose_base_trial(results).trial_id == "trial_valid"


def test_base_selection_uses_tps_then_ttft_then_trial_id() -> None:
    results = [
        make_result("trial_003", tps=120.0, ttft=0.4),
        make_result("trial_002", tps=120.0, ttft=0.3),
        make_result("trial_001", tps=120.0, ttft=0.3),
        make_result("trial_004", tps=110.0, ttft=0.1),
    ]

    assert choose_base_trial(results).trial_id == "trial_001"


def test_no_eligible_initial_trial_returns_explicit_termination() -> None:
    with pytest.raises(InitialDesignError, match="no successful initial Trial"):
        choose_base_trial([make_result("trial_bo", phase="bo")])