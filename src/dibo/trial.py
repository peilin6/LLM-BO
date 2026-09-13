"""Single-trial orchestration with injectable execution boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dibo.benchmark import BenchmarkResult
from dibo.engine import EngineLaunchSpec
from dibo.metrics import MetricsResult
from dibo.observability import emit_event
from dibo.schemas import METRIC_IDS, CompileTrace, RunMode, TrialResult, TrialStatus, WorkloadSpec


class EngineBoundary(Protocol):
    async def start(self, config: EngineLaunchSpec, run_dir: Path) -> Any: ...

    async def wait_ready(self, handle: Any, timeout_s: float) -> str: ...

    async def stop(self, handle: Any) -> None: ...


class BenchmarkBoundary(Protocol):
    async def run_warmup(self, handle: Any, workload: WorkloadSpec) -> None: ...

    async def run_measurement(self, handle: Any, workload: WorkloadSpec) -> BenchmarkResult: ...


class MetricsBoundary(Protocol):
    def start_sampling(
        self,
        handle: Any,
        gpu_uuid: str,
        interval_s: float,
        *,
        min_valid_samples: int,
    ) -> Any: ...

    def stop_and_aggregate(
        self,
        sampler: Any,
        completed_requests: int,
        *,
        ttft_p95_s: float | None,
    ) -> MetricsResult: ...


class StoreBoundary(Protocol):
    def save_json_atomic(self, path: Path, payload: Any, *, overwrite: bool = False) -> None: ...

    def save_trial(self, result: TrialResult, run_dir: Path) -> Path: ...


@dataclass(frozen=True)
class TrialSpec:
    run_mode: RunMode
    trace: CompileTrace
    engine_launch: EngineLaunchSpec
    workload: WorkloadSpec
    run_dir: Path
    startup_timeout_s: float
    benchmark_timeout_s: float
    gpu_uuid: str
    sampling_interval_s: float
    min_valid_samples: int


def _spec_payload(spec: TrialSpec) -> dict[str, Any]:
    return {
        "run_mode": spec.run_mode.value,
        "trial_id": spec.trace.trial_id,
        "engine": spec.engine_launch.engine.model_dump(mode="json"),
        "workload": spec.workload.model_dump(mode="json"),
        "startup_timeout_s": spec.startup_timeout_s,
        "benchmark_timeout_s": spec.benchmark_timeout_s,
        "gpu_uuid": spec.gpu_uuid,
        "sampling_interval_s": spec.sampling_interval_s,
        "min_valid_samples": spec.min_valid_samples,
    }


def _benchmark_observability(
    benchmark_result: BenchmarkResult,
    queue_backlog_p95: float | None,
) -> dict[str, Any]:
    configured_rate = benchmark_result.configured_request_rate_rps
    issued_rate = benchmark_result.issued_request_rate_rps
    completed_rate = benchmark_result.completed_request_rate_rps
    backlog_detected = queue_backlog_p95 is not None and queue_backlog_p95 > 0
    return {
        "total_requests": benchmark_result.total_requests,
        "completed_requests": benchmark_result.completed_requests,
        "failed_requests": benchmark_result.failed_requests,
        "success_rate": benchmark_result.success_rate,
        "configured_request_rate_rps": configured_rate,
        "issued_request_rate_rps": issued_rate,
        "completed_request_rate_rps": completed_rate,
        "launch_span_s": benchmark_result.launch_span_s,
        "duration_s": benchmark_result.duration_s,
        "queue_backlog_p95": queue_backlog_p95,
        "backlog_detected": backlog_detected,
    }


async def run_trial(
    spec: TrialSpec,
    engine: EngineBoundary,
    benchmark: BenchmarkBoundary,
    metrics: MetricsBoundary,
    store: StoreBoundary,
) -> TrialResult:
    """Execute one legal compiled candidate and persist exactly one result."""
    if not spec.trace.valid:
        raise ValueError("invalid CompileTrace cannot be executed")
    if dict(spec.engine_launch.final_config) != spec.trace.final_config:
        raise ValueError("Engine final_config must equal CompileTrace final_config")

    trial_dir = spec.run_dir / "trials" / spec.trace.trial_id
    emit_event(spec.run_dir, "trial_trace_saved", trial_id=spec.trace.trial_id)
    store.save_json_atomic(trial_dir / "spec.json", _spec_payload(spec))
    store.save_json_atomic(
        trial_dir / "compile_trace.json",
        spec.trace.model_dump(mode="json"),
    )

    handle: Any | None = None
    sampler: Any | None = None
    benchmark_result: BenchmarkResult | None = None
    metrics_result: MetricsResult | None = None
    execution_mode: str | None = None
    failure_reason: str | None = None
    status = TrialStatus.INTERNAL_ERROR
    stage = "startup"
    cleanup_parts: list[str] = []
    try:
        emit_event(spec.run_dir, "engine_starting", trial_id=spec.trace.trial_id)
        handle = await engine.start(spec.engine_launch, trial_dir)
        emit_event(spec.run_dir, "engine_started", trial_id=spec.trace.trial_id)
        execution_mode = await engine.wait_ready(handle, spec.startup_timeout_s)
        emit_event(spec.run_dir, "engine_ready", trial_id=spec.trace.trial_id, execution_mode=execution_mode)
        stage = "benchmark"
        emit_event(spec.run_dir, "warmup_started", trial_id=spec.trace.trial_id)
        await asyncio.wait_for(
            benchmark.run_warmup(handle, spec.workload),
            timeout=spec.benchmark_timeout_s,
        )
        emit_event(spec.run_dir, "warmup_completed", trial_id=spec.trace.trial_id)
        sampler = metrics.start_sampling(
            handle,
            spec.gpu_uuid,
            spec.sampling_interval_s,
            min_valid_samples=spec.min_valid_samples,
        )
        emit_event(spec.run_dir, "metrics_sampling_started", trial_id=spec.trace.trial_id,
               interval_s=spec.sampling_interval_s)
        emit_event(spec.run_dir, "benchmark_started", trial_id=spec.trace.trial_id)
        benchmark_result = await asyncio.wait_for(
            benchmark.run_measurement(handle, spec.workload),
            timeout=spec.benchmark_timeout_s,
        )
        emit_event(
            spec.run_dir,
            "benchmark_completed",
            trial_id=spec.trace.trial_id,
            throughput_tps=benchmark_result.throughput_tps,
            **_benchmark_observability(benchmark_result, None),
        )
        metrics_result = metrics.stop_and_aggregate(
            sampler,
            benchmark_result.completed_requests,
            ttft_p95_s=benchmark_result.ttft_p95_s,
        )
        queue_backlog_p95 = metrics_result.values["m04"].value
        emit_event(
            spec.run_dir,
            "metrics_aggregated",
            trial_id=spec.trace.trial_id,
            metrics={key: value.value for key, value in metrics_result.values.items()},
            queue_backlog_p95=queue_backlog_p95,
            backlog_detected=queue_backlog_p95 is not None and queue_backlog_p95 > 0,
        )
        sampler = None
        if benchmark_result.completed_requests == 0:
            raise RuntimeError("formal benchmark completed no requests")
        if benchmark_result.success_rate < spec.workload.required_success_rate:
            raise RuntimeError("formal benchmark success rate is below the required threshold")
        status = TrialStatus.SUCCESS
    except (TimeoutError, asyncio.TimeoutError) as error:
        status = TrialStatus.TIMEOUT
        failure_reason = str(error)
        emit_event(spec.run_dir, "trial_timeout", trial_id=spec.trace.trial_id, stage=stage,
               error=failure_reason[:500])
    except (KeyboardInterrupt, asyncio.CancelledError) as error:
        status = TrialStatus.INTERRUPTED
        failure_reason = str(error) or type(error).__name__
        emit_event(spec.run_dir, "trial_interrupted", trial_id=spec.trace.trial_id, stage=stage,
               error=failure_reason[:500])
    except (RuntimeError, OSError, ValueError) as error:
        status = TrialStatus.STARTUP_FAILED if stage == "startup" else TrialStatus.BENCHMARK_FAILED
        failure_reason = str(error)
        emit_event(spec.run_dir, "trial_failed", trial_id=spec.trace.trial_id, stage=stage,
                   failure_type=type(error).__name__, error=failure_reason[:500])
    except Exception as error:  # noqa: BLE001
        status = TrialStatus.INTERNAL_ERROR
        failure_reason = str(error)
        emit_event(spec.run_dir, "trial_internal_error", trial_id=spec.trace.trial_id, stage=stage,
                   failure_type=type(error).__name__, error=failure_reason[:500])
    finally:
        if sampler is not None:
            try:
                metrics.stop_and_aggregate(sampler, 0, ttft_p95_s=None)
                cleanup_parts.append("sampler_stopped")
            except Exception as error:  # noqa: BLE001
                cleanup_parts.append(f"sampler_cleanup_failed:{error}")
                emit_event(spec.run_dir, "sampler_cleanup_failed", trial_id=spec.trace.trial_id,
                           error=str(error)[:500])
        if handle is not None:
            try:
                await engine.stop(handle)
                cleanup_parts.append("engine_stopped")
                emit_event(spec.run_dir, "engine_stopped", trial_id=spec.trace.trial_id)
            except Exception as error:  # noqa: BLE001
                cleanup_parts.append(f"engine_cleanup_failed:{error}")
                emit_event(spec.run_dir, "engine_cleanup_failed", trial_id=spec.trace.trial_id,
                           error=str(error)[:500])
        if not cleanup_parts:
            cleanup_parts.append("nothing_started")

    metric_values = {metric_id: None for metric_id in METRIC_IDS}
    missing_reasons: dict[str, str] = {}
    valid_sample_counts: dict[str, int] = {}
    if metrics_result is not None:
        for metric_id, metric_value in metrics_result.values.items():
            metric_values[metric_id] = metric_value.value
            valid_sample_counts[metric_id] = metric_value.valid_samples
            if metric_value.missing_reason is not None:
                missing_reasons[metric_id] = metric_value.missing_reason
    if failure_reason is not None:
        missing_reasons["trial"] = failure_reason

    if benchmark_result is not None:
        store.save_json_atomic(
            trial_dir / "benchmark.json",
            {
                "total_requests": benchmark_result.total_requests,
                "completed_requests": benchmark_result.completed_requests,
                "failed_requests": benchmark_result.failed_requests,
                "output_tokens": benchmark_result.output_tokens,
                "duration_s": benchmark_result.duration_s,
                "throughput_tps": benchmark_result.throughput_tps,
                "ttft_p95_s": benchmark_result.ttft_p95_s,
                "success_rate": benchmark_result.success_rate,
                "configured_request_rate_rps": benchmark_result.configured_request_rate_rps,
                "issued_request_rate_rps": benchmark_result.issued_request_rate_rps,
                "completed_request_rate_rps": benchmark_result.completed_request_rate_rps,
                "launch_span_s": benchmark_result.launch_span_s,
                "requests": benchmark_result.request_results,
            },
        )
    store.save_json_atomic(
        trial_dir / "metrics_summary.json",
        {
            metric_id: {
                "value": metric_values[metric_id],
                "missing_reason": missing_reasons.get(metric_id),
                "valid_samples": valid_sample_counts.get(metric_id, 0),
            }
            for metric_id in METRIC_IDS
        },
    )

    result = TrialResult(
        trial_id=spec.trace.trial_id,
        run_mode=spec.run_mode,
        status=status,
        trace=spec.trace,
        vllm_execution_mode=execution_mode,
        metrics=metric_values,
        metric_missing_reasons=missing_reasons,
        throughput_tps=(benchmark_result.throughput_tps if status == TrialStatus.SUCCESS else None),
        request_count=(benchmark_result.total_requests if benchmark_result else 0),
        completed_requests=(benchmark_result.completed_requests if benchmark_result else 0),
        successful_requests=(benchmark_result.completed_requests if benchmark_result else 0),
        configured_request_rate_rps=(
            benchmark_result.configured_request_rate_rps if benchmark_result else None
        ),
        issued_request_rate_rps=(benchmark_result.issued_request_rate_rps if benchmark_result else None),
        completed_request_rate_rps=(
            benchmark_result.completed_request_rate_rps if benchmark_result else None
        ),
        queue_backlog_p95=metric_values["m04"],
        backlog_detected=metric_values["m04"] is not None and metric_values["m04"] > 0,
        duration_s=(benchmark_result.duration_s if benchmark_result else None),
        sampling_interval_s=spec.sampling_interval_s,
        valid_sample_counts=valid_sample_counts,
        log_paths={"trial_dir": str(trial_dir)},
        cleanup_result=",".join(cleanup_parts),
    )
    store.save_trial(result, spec.run_dir)
    emit_event(spec.run_dir, "trial_result_saved", trial_id=spec.trace.trial_id, status=result.status.value)
    return result
