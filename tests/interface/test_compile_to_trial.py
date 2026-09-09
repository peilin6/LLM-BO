from __future__ import annotations

from pathlib import Path

import pytest

from dibo.benchmark import BenchmarkResult
from dibo.compiler import CompileEnvironment, compile_config
from dibo.engine import EngineLaunchSpec, build_argv
from dibo.metrics import MetricsResult, MetricValue
from dibo.schemas import (
    METRIC_IDS,
    EngineAdapter,
    RunMode,
    load_actions,
    load_experiment,
    load_parameters,
)
from dibo.trial import TrialSpec, run_trial

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


def make_trace(*, invalid: bool = False):
    environment = CompileEnvironment(
        load_parameters(CONFIGS / "parameters.yaml"),
        1,
        28,
        28,
        8192,
        (8, 16, 32),
        supported_parameter_ids=(() if invalid else tuple(BASE_CONFIG)),
    )
    return compile_config(
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        ("A06",),
        {"A06": 0.8},
        environment,
        trial_id="trial_008",
    )


class EngineSpy:
    def __init__(self) -> None:
        self.launches: list[EngineLaunchSpec] = []

    async def start(self, config, run_dir):
        self.launches.append(config)
        return object()

    async def wait_ready(self, handle, timeout_s):
        return "V1"

    async def stop(self, handle):
        return None


class BenchmarkFake:
    async def run_warmup(self, handle, workload):
        return None

    async def run_measurement(self, handle, workload):
        return BenchmarkResult(1, 1, 0, 100, 1.0, 100.0, 0.2, 1.0, ())


class MetricsFake:
    def start_sampling(self, handle, gpu_uuid, interval_s, *, min_valid_samples):
        return object()

    def stop_and_aggregate(self, sampler, completed_requests, *, ttft_p95_s):
        return MetricsResult({metric_id: MetricValue(0.1, valid_samples=3) for metric_id in METRIC_IDS})


class StoreFake:
    def save_json_atomic(self, path, payload, *, overwrite=False):
        return None

    def save_trial(self, result, run_dir):
        return run_dir / "trials" / result.trial_id / "result.json"


def make_spec(tmp_path: Path, trace) -> TrialSpec:
    experiment = load_experiment(CONFIGS / "experiment_smoke.yaml")
    return TrialSpec(
        RunMode.SYNTHETIC,
        trace,
        EngineLaunchSpec(experiment.engine, trace.final_config),
        experiment.workload,
        tmp_path,
        1.0,
        1.0,
        "SYNTHETIC-GPU",
        1.0,
        3,
    )


@pytest.mark.asyncio
async def test_compiler_engine_argv_and_result_share_exact_final_config(tmp_path: Path) -> None:
    trace = make_trace()
    engine_spy = EngineSpy()

    result = await run_trial(
        make_spec(tmp_path, trace),
        engine_spy,
        BenchmarkFake(),
        MetricsFake(),
        StoreFake(),
    )

    launch = engine_spy.launches[0]
    real_engine = launch.engine.model_copy(
        update={"adapter": EngineAdapter.VLLM_SUBPROCESS, "execution_mode": "V1"}
    )
    argv = build_argv(real_engine, launch.final_config)
    assert dict(launch.final_config) == trace.final_config == result.trace.final_config
    assert argv[argv.index("--max-num-seqs") + 1] == str(trace.final_config["p03"])
    assert argv[argv.index("--max-num-batched-tokens") + 1] == str(trace.final_config["p04"])


@pytest.mark.asyncio
async def test_invalid_compiler_trace_never_calls_engine(tmp_path: Path) -> None:
    invalid_trace = make_trace(invalid=True)
    engine_spy = EngineSpy()

    with pytest.raises(ValueError, match="invalid CompileTrace"):
        await run_trial(
            make_spec(tmp_path, invalid_trace),
            engine_spy,
            BenchmarkFake(),
            MetricsFake(),
            StoreFake(),
        )

    assert engine_spy.launches == []