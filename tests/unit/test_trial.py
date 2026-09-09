from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dibo.benchmark import BenchmarkResult
from dibo.compiler import CompileEnvironment, compile_config
from dibo.engine import EngineLaunchSpec
from dibo.metrics import MetricsResult, MetricValue
from dibo.schemas import (
    METRIC_IDS,
    RunMode,
    TrialStatus,
    load_actions,
    load_experiment,
    load_parameters,
)
from dibo.trial import TrialSpec, run_trial

CONFIGS = Path(__file__).parents[2] / "configs"


def compiled_trace():
    catalog = load_parameters(CONFIGS / "parameters.yaml")
    base_config = {
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
    environment = CompileEnvironment(catalog, 1, 28, 28, 8192, (8, 16, 32))
    return compile_config(
        base_config,
        load_actions(CONFIGS / "actions_v1.yaml"),
        (),
        {},
        environment,
        trial_id="trial_001",
        phase="initial",
        bo_round=None,
        compile_base_id="x_init",
    )


class FakeEngine:
    def __init__(self, events: list[str], *, start_error: BaseException | None = None) -> None:
        self.events = events
        self.start_error = start_error

    async def start(self, config, run_dir):
        self.events.append("start")
        if self.start_error is not None:
            raise self.start_error
        return object()

    async def wait_ready(self, handle, timeout_s):
        self.events.append("ready")
        return "V1"

    async def stop(self, handle):
        self.events.append("engine_stop")


class FakeBenchmark:
    def __init__(self, events: list[str], *, error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    async def run_warmup(self, handle, workload):
        self.events.append("warmup")

    async def run_measurement(self, handle, workload):
        self.events.append("benchmark")
        if self.error is not None:
            raise self.error
        return BenchmarkResult(2, 2, 0, 200, 2.0, 100.0, 0.4, 1.0, ())


class FakeMetrics:
    def __init__(self, events: list[str], *, missing_metric: str | None = None) -> None:
        self.events = events
        self.missing_metric = missing_metric

    def start_sampling(self, handle, gpu_uuid, interval_s, *, min_valid_samples):
        self.events.append("sample_start")
        return object()

    def stop_and_aggregate(self, sampler, completed_requests, *, ttft_p95_s):
        self.events.append("sample_stop")
        values = {metric_id: MetricValue(0.1, valid_samples=3) for metric_id in METRIC_IDS}
        if self.missing_metric is not None:
            values[self.missing_metric] = MetricValue(None, "unavailable")
        return MetricsResult(values)


class FakeStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.results = []

    def save_json_atomic(self, path, payload, *, overwrite=False):
        self.events.append(path.name)

    def save_trial(self, result, run_dir):
        self.events.append("result.json")
        self.results.append(result)
        return run_dir / "trials" / result.trial_id / "result.json"


def trial_spec(tmp_path: Path, *, trace=None, timeout: float = 1.0) -> TrialSpec:
    experiment = load_experiment(CONFIGS / "experiment_smoke.yaml")
    candidate = trace or compiled_trace()
    return TrialSpec(
        run_mode=RunMode.SYNTHETIC,
        trace=candidate,
        engine_launch=EngineLaunchSpec(experiment.engine, candidate.final_config),
        workload=experiment.workload,
        run_dir=tmp_path,
        startup_timeout_s=1.0,
        benchmark_timeout_s=timeout,
        gpu_uuid="SYNTHETIC-GPU",
        sampling_interval_s=1.0,
        min_valid_samples=3,
    )


@pytest.mark.asyncio
async def test_success_uses_exact_lifecycle_order_and_saves_once(tmp_path: Path) -> None:
    events: list[str] = []
    store = FakeStore(events)

    result = await run_trial(
        trial_spec(tmp_path),
        FakeEngine(events),
        FakeBenchmark(events),
        FakeMetrics(events),
        store,
    )

    assert events == [
        "spec.json",
        "compile_trace.json",
        "start",
        "ready",
        "warmup",
        "sample_start",
        "benchmark",
        "sample_stop",
        "engine_stop",
        "benchmark.json",
        "metrics_summary.json",
        "result.json",
    ]
    assert result.status == TrialStatus.SUCCESS
    assert result.throughput_tps == 100.0
    assert result.trace.final_config == dict(trial_spec(tmp_path).engine_launch.final_config)
    assert len(store.results) == 1


@pytest.mark.asyncio
async def test_optional_metric_missing_preserves_successful_tps(tmp_path: Path) -> None:
    events: list[str] = []
    result = await run_trial(
        trial_spec(tmp_path),
        FakeEngine(events),
        FakeBenchmark(events),
        FakeMetrics(events, missing_metric="m05"),
        FakeStore(events),
    )

    assert result.status == TrialStatus.SUCCESS
    assert result.throughput_tps == 100.0
    assert result.metrics["m05"] is None
    assert result.metric_missing_reasons["m05"] == "unavailable"


@pytest.mark.asyncio
async def test_startup_failure_is_saved_once_without_retry(tmp_path: Path) -> None:
    events: list[str] = []
    store = FakeStore(events)
    result = await run_trial(
        trial_spec(tmp_path),
        FakeEngine(events, start_error=RuntimeError("start failed")),
        FakeBenchmark(events),
        FakeMetrics(events),
        store,
    )

    assert result.status == TrialStatus.STARTUP_FAILED
    assert result.throughput_tps is None
    assert events.count("start") == 1
    assert events.count("result.json") == 1


@pytest.mark.asyncio
async def test_benchmark_failure_cleans_sampler_and_engine(tmp_path: Path) -> None:
    events: list[str] = []
    result = await run_trial(
        trial_spec(tmp_path),
        FakeEngine(events),
        FakeBenchmark(events, error=RuntimeError("benchmark failed")),
        FakeMetrics(events),
        FakeStore(events),
    )

    assert result.status == TrialStatus.BENCHMARK_FAILED
    assert result.throughput_tps is None
    assert events[-4:] == ["sample_stop", "engine_stop", "metrics_summary.json", "result.json"]


@pytest.mark.asyncio
async def test_benchmark_timeout_is_distinct_and_cleans_up(tmp_path: Path) -> None:
    events: list[str] = []

    class SlowBenchmark(FakeBenchmark):
        async def run_measurement(self, handle, workload):
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    result = await run_trial(
        trial_spec(tmp_path, timeout=0.001),
        FakeEngine(events),
        SlowBenchmark(events),
        FakeMetrics(events),
        FakeStore(events),
    )

    assert result.status == TrialStatus.TIMEOUT
    assert "engine_stop" in events


@pytest.mark.asyncio
async def test_interrupted_and_internal_error_statuses(tmp_path: Path) -> None:
    for error, expected in (
        (asyncio.CancelledError(), TrialStatus.INTERRUPTED),
        (LookupError("unexpected"), TrialStatus.INTERNAL_ERROR),
    ):
        events: list[str] = []
        result = await run_trial(
            trial_spec(tmp_path / expected.value),
            FakeEngine(events),
            FakeBenchmark(events, error=error),
            FakeMetrics(events),
            FakeStore(events),
        )
        assert result.status == expected
        assert events.count("result.json") == 1


@pytest.mark.asyncio
async def test_invalid_trace_and_mismatched_engine_config_never_start(tmp_path: Path) -> None:
    invalid = compiled_trace().model_copy(update={"constraint_errors": ("invalid",)})
    events: list[str] = []
    with pytest.raises(ValueError, match="invalid CompileTrace"):
        await run_trial(
            trial_spec(tmp_path, trace=invalid),
            FakeEngine(events),
            FakeBenchmark(events),
            FakeMetrics(events),
            FakeStore(events),
        )

    valid_spec = trial_spec(tmp_path)
    changed = dict(valid_spec.trace.final_config)
    changed["p03"] = 999
    mismatched = TrialSpec(
        **{
            **valid_spec.__dict__,
            "engine_launch": EngineLaunchSpec(valid_spec.engine_launch.engine, changed),
        }
    )
    with pytest.raises(ValueError, match="must equal"):
        await run_trial(
            mismatched,
            FakeEngine(events),
            FakeBenchmark(events),
            FakeMetrics(events),
            FakeStore(events),
        )
    assert events == []