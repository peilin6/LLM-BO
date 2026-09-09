"""Metric sampling boundary and six-metric aggregation."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from dibo.engine import EngineHandle


@dataclass(frozen=True)
class MetricSample:
    timestamp_s: float
    kv_cache_usage: float | None = None
    preemption_counter: float | None = None
    queue_length: float | None = None
    gpu_utilization: float | None = None
    memory_utilization: float | None = None


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    missing_reason: str | None = None
    valid_samples: int = 0


@dataclass(frozen=True)
class MetricsResult:
    values: dict[str, MetricValue]


class SampleProvider(Protocol):
    def sample(self, gpu_uuid: str) -> MetricSample: ...


@dataclass
class Sampler:
    provider: SampleProvider
    gpu_uuid: str
    interval_s: float
    min_valid_samples: int
    samples: list[MetricSample] = field(default_factory=list)

    def collect(self) -> MetricSample:
        sample = self.provider.sample(self.gpu_uuid)
        self.samples.append(sample)
        return sample


_PROMETHEUS_ALIASES = {
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "preemption_counter": ("vllm:num_preemptions", "vllm:num_preemptions_total"),
    "queue_length": ("vllm:num_requests_waiting",),
}


def _first_metric(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    return next((metrics[name] for name in names if name in metrics), None)


class PrometheusNvmlProvider:
    """Read one vLLM Prometheus snapshot and one assigned-GPU NVML snapshot."""

    def __init__(
        self,
        metrics_url: str,
        model_name: str,
        *,
        http_get=httpx.get,
        nvml_module=None,
    ) -> None:
        self.metrics_url = metrics_url
        self.model_name = model_name
        self.http_get = http_get
        if nvml_module is None:
            import pynvml

            nvml_module = pynvml
        self.nvml = nvml_module
        self.nvml.nvmlInit()

    def sample(self, gpu_uuid: str) -> MetricSample:
        response = self.http_get(self.metrics_url, timeout=5.0)
        response.raise_for_status()
        parsed = parse_prometheus(response.text, model_name=self.model_name)
        handle = self.nvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
        return MetricSample(
            timestamp_s=time.time(),
            kv_cache_usage=_first_metric(parsed, _PROMETHEUS_ALIASES["kv_cache_usage"]),
            preemption_counter=_first_metric(parsed, _PROMETHEUS_ALIASES["preemption_counter"]),
            queue_length=_first_metric(parsed, _PROMETHEUS_ALIASES["queue_length"]),
            gpu_utilization=float(utilization.gpu) / 100.0,
            memory_utilization=float(utilization.memory) / 100.0,
        )

    def close(self) -> None:
        self.nvml.nvmlShutdown()


@dataclass
class BackgroundSampler:
    sampler: Sampler
    output_path: Path
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    errors: list[str] = field(default_factory=list)


class PrometheusNvmlMetrics:
    """Trial metrics boundary that samples in a background thread during benchmark."""

    def __init__(self, *, provider_factory=PrometheusNvmlProvider) -> None:
        self.provider_factory = provider_factory

    def start_sampling(
        self,
        handle: EngineHandle,
        gpu_uuid: str,
        interval_s: float,
        *,
        min_valid_samples: int,
    ) -> BackgroundSampler:
        provider = self.provider_factory(
            f"http://{handle.config.host}:{handle.config.port}/metrics",
            handle.config.served_model_name,
        )
        state = BackgroundSampler(
            start_sampling(provider, gpu_uuid, interval_s, min_valid_samples=min_valid_samples),
            handle.log_path.parent / "metrics.jsonl",
        )

        def collect() -> None:
            with state.output_path.open("x", encoding="utf-8") as output:
                while not state.stop_event.is_set():
                    cycle_started = time.monotonic()
                    try:
                        sample = state.sampler.collect()
                        output.write(json.dumps({"sample": asdict(sample)}, ensure_ascii=True) + "\n")
                    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as error:
                        message = f"{type(error).__name__}: {error}"
                        state.errors.append(message)
                        output.write(json.dumps({"error": message, "timestamp_s": time.time()}) + "\n")
                    output.flush()
                    remaining = state.sampler.interval_s - (time.monotonic() - cycle_started)
                    if remaining > 0:
                        state.stop_event.wait(remaining)

        state.thread = threading.Thread(target=collect, name="dibo-metrics-sampler", daemon=True)
        state.thread.start()
        return state

    def stop_and_aggregate(
        self,
        sampler: BackgroundSampler,
        completed_requests: int,
        *,
        ttft_p95_s: float | None,
    ) -> MetricsResult:
        sampler.stop_event.set()
        if sampler.thread is not None:
            sampler.thread.join(timeout=max(5.0, sampler.sampler.interval_s * 2))
            if sampler.thread.is_alive():
                raise RuntimeError("metrics sampler thread did not stop")
        provider = sampler.sampler.provider
        if hasattr(provider, "close"):
            provider.close()
        return stop_and_aggregate(
            sampler.sampler,
            completed_requests,
            ttft_p95_s=ttft_p95_s,
        )


def start_sampling(
    handle: SampleProvider,
    gpu_uuid: str,
    interval_s: float,
    *,
    min_valid_samples: int = 3,
) -> Sampler:
    """Create an explicit sampler; callers control the formal-window collection loop."""
    if interval_s <= 0 or not isfinite(interval_s):
        raise ValueError("sampling interval must be positive and finite")
    if min_valid_samples < 2:
        raise ValueError("min_valid_samples must be at least 2")
    return Sampler(handle, gpu_uuid, interval_s, min_valid_samples)


def _aggregate(samples: list[MetricSample], field_name: str, method: str, minimum: int) -> MetricValue:
    values = [getattr(sample, field_name) for sample in samples]
    finite_values = [float(value) for value in values if value is not None and isfinite(value)]
    if len(finite_values) < minimum:
        return MetricValue(None, "insufficient_samples", len(finite_values))
    value = (
        float(np.quantile(finite_values, 0.95, method="linear"))
        if method == "p95"
        else float(np.mean(finite_values))
    )
    return MetricValue(value, valid_samples=len(finite_values))


def stop_and_aggregate(
    sampler: Sampler,
    completed_requests: int,
    *,
    ttft_p95_s: float | None = None,
) -> MetricsResult:
    """Aggregate samples captured strictly inside the formal measurement window."""
    minimum = sampler.min_valid_samples
    values = {
        "m01": _aggregate(sampler.samples, "kv_cache_usage", "p95", minimum),
        "m03": _aggregate(sampler.samples, "gpu_utilization", "mean", minimum),
        "m04": _aggregate(sampler.samples, "queue_length", "p95", minimum),
        "m05": _aggregate(sampler.samples, "memory_utilization", "mean", minimum),
    }

    counters = [
        float(sample.preemption_counter)
        for sample in sampler.samples
        if sample.preemption_counter is not None and isfinite(sample.preemption_counter)
    ]
    if len(counters) < 2:
        values["m02"] = MetricValue(None, "insufficient_counter_samples", len(counters))
    elif any(current < previous for previous, current in pairwise(counters)):
        values["m02"] = MetricValue(None, "counter_reset", len(counters))
    elif completed_requests <= 0:
        values["m02"] = MetricValue(None, "zero_completed_requests", len(counters))
    else:
        values["m02"] = MetricValue(
            (counters[-1] - counters[0]) / completed_requests,
            valid_samples=len(counters),
        )

    if ttft_p95_s is None or not isfinite(ttft_p95_s) or ttft_p95_s < 0:
        values["m06"] = MetricValue(None, "missing_ttft")
    else:
        values["m06"] = MetricValue(float(ttft_p95_s), valid_samples=completed_requests)
    return MetricsResult(values)


_PROMETHEUS_LINE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$'
)


def parse_prometheus(text: str, *, model_name: str) -> dict[str, float]:
    """Parse selected vLLM metrics and require matching model_name labels when present."""
    parsed: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_LINE.match(line)
        if match is None:
            continue
        labels = match.group("labels") or ""
        label_values = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        if "model_name" in label_values and label_values["model_name"] != model_name:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if isfinite(value):
            parsed[match.group("name")] = value
    return parsed