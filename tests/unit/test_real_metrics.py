import json
import time
from types import SimpleNamespace

import httpx
import pytest

from dibo.metrics import PrometheusNvmlMetrics, PrometheusNvmlProvider


class FakeNvml:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.uuids = []

    def nvmlInit(self) -> None:
        self.initialized = True

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.uuids.append(uuid)
        return "handle"

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=75, memory=25)

    def nvmlShutdown(self) -> None:
        self.closed = True


def prometheus_response(preemptions: int = 12) -> httpx.Response:
    text = f"""vllm:kv_cache_usage_perc{{model_name="dibo-model"}} 0.8
vllm:num_preemptions{{model_name="dibo-model"}} {preemptions}
vllm:num_requests_waiting{{model_name="dibo-model"}} 3
"""
    return httpx.Response(200, text=text, request=httpx.Request("GET", "http://local/metrics"))


def test_provider_combines_prometheus_and_exact_uuid_nvml() -> None:
    nvml = FakeNvml()
    provider = PrometheusNvmlProvider(
        "http://local/metrics", "dibo-model",
        http_get=lambda url, timeout: prometheus_response(), nvml_module=nvml,
    )
    sample = provider.sample("GPU-assigned")
    provider.close()

    assert (sample.kv_cache_usage, sample.preemption_counter, sample.queue_length) == (0.8, 12, 3)
    assert (sample.gpu_utilization, sample.memory_utilization) == (0.75, 0.25)
    assert nvml.uuids == ["GPU-assigned"]
    assert nvml.initialized and nvml.closed


def test_background_sampler_writes_raw_metrics_jsonl_and_aggregates(tmp_path) -> None:
    nvml = FakeNvml()
    counter = 0

    def factory(url, model):
        nonlocal counter
        def get(_url, timeout):
            nonlocal counter
            counter += 1
            return prometheus_response(10 + counter)
        return PrometheusNvmlProvider(url, model, http_get=get, nvml_module=nvml)

    adapter = PrometheusNvmlMetrics(provider_factory=factory)
    handle = SimpleNamespace(
        config=SimpleNamespace(host="127.0.0.1", port=8000, served_model_name="dibo-model"),
        log_path=tmp_path / "engine.log",
    )
    sampler = adapter.start_sampling(handle, "GPU-assigned", 0.01, min_valid_samples=2)
    deadline = time.time() + 1
    while counter < 3 and time.time() < deadline:
        time.sleep(0.01)
    result = adapter.stop_and_aggregate(sampler, 100, ttft_p95_s=0.4)

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 2 and all("sample" in row for row in rows)
    assert result.values["m01"].value == pytest.approx(0.8)
    assert result.values["m02"].value is not None and result.values["m02"].value > 0
    assert result.values["m03"].value == pytest.approx(0.75)
    assert result.values["m05"].value == pytest.approx(0.25)