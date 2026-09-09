from __future__ import annotations

from pathlib import Path

import pytest

from dibo.metrics import MetricSample, parse_prometheus, start_sampling, stop_and_aggregate

FIXTURES = Path(__file__).parents[1] / "fixtures"


class SequenceProvider:
    def __init__(self, samples: list[MetricSample]) -> None:
        self.samples = iter(samples)
        self.uuids: list[str] = []

    def sample(self, gpu_uuid: str) -> MetricSample:
        self.uuids.append(gpu_uuid)
        return next(self.samples)


def samples() -> list[MetricSample]:
    return [
        MetricSample(0, 0.70, 10, 1, 0.50, 0.20),
        MetricSample(1, 0.80, 11, 2, 0.60, 0.40),
        MetricSample(2, 0.90, 12, 3, 0.70, 0.60),
    ]


def test_prometheus_parser_filters_model_labels() -> None:
    text = (FIXTURES / "prometheus_vllm_0112.txt").read_text(encoding="utf-8")

    parsed = parse_prometheus(text, model_name="dibo-model")

    assert parsed["vllm:kv_cache_usage_perc"] == 0.75
    assert parsed["vllm:num_requests_waiting"] == 3
    assert parsed["vllm:num_preemptions_total"] == 12


def test_six_metrics_are_aggregated_from_formal_samples() -> None:
    provider = SequenceProvider(samples())
    sampler = start_sampling(provider, "GPU-allocated", 1.0, min_valid_samples=3)
    for _ in range(3):
        sampler.collect()

    result = stop_and_aggregate(sampler, 100, ttft_p95_s=0.45)

    assert result.values["m01"].value == pytest.approx(0.89)
    assert result.values["m02"].value == pytest.approx(0.02)
    assert result.values["m03"].value == pytest.approx(0.60)
    assert result.values["m04"].value == pytest.approx(2.9)
    assert result.values["m05"].value == pytest.approx(0.40)
    assert result.values["m06"].value == 0.45
    assert provider.uuids == ["GPU-allocated"] * 3


def test_gpu_and_memory_utilization_are_not_interchanged() -> None:
    provider = SequenceProvider(samples())
    sampler = start_sampling(provider, "GPU-allocated", 1.0, min_valid_samples=3)
    for _ in range(3):
        sampler.collect()

    result = stop_and_aggregate(sampler, 100, ttft_p95_s=0.45)

    assert result.values["m03"].value == pytest.approx(0.60)
    assert result.values["m05"].value == pytest.approx(0.40)


def test_insufficient_samples_and_missing_ttft_are_not_filled_with_zero() -> None:
    provider = SequenceProvider(samples()[:2])
    sampler = start_sampling(provider, "GPU-allocated", 1.0, min_valid_samples=3)
    sampler.collect()
    sampler.collect()

    result = stop_and_aggregate(sampler, 100)

    assert result.values["m01"].value is None
    assert result.values["m01"].missing_reason == "insufficient_samples"
    assert result.values["m06"].value is None
    assert result.values["m06"].missing_reason == "missing_ttft"


def test_counter_reset_and_zero_completed_requests_are_missing() -> None:
    reset_provider = SequenceProvider(
        [MetricSample(0, preemption_counter=10), MetricSample(1, preemption_counter=2)]
    )
    reset = start_sampling(reset_provider, "GPU", 1.0)
    reset.collect()
    reset.collect()
    assert stop_and_aggregate(reset, 100).values["m02"].missing_reason == "counter_reset"

    zero_provider = SequenceProvider(
        [MetricSample(0, preemption_counter=1), MetricSample(1, preemption_counter=3)]
    )
    zero = start_sampling(zero_provider, "GPU", 1.0)
    zero.collect()
    zero.collect()
    assert stop_and_aggregate(zero, 0).values["m02"].missing_reason == "zero_completed_requests"


def test_intermediate_counter_reset_is_detected_even_if_final_value_is_larger() -> None:
    provider = SequenceProvider([
        MetricSample(0, preemption_counter=10),
        MetricSample(1, preemption_counter=2),
        MetricSample(2, preemption_counter=15),
    ])
    sampler = start_sampling(provider, "GPU", 1.0, min_valid_samples=3)
    for _ in range(3):
        sampler.collect()
    assert stop_and_aggregate(sampler, 100).values["m02"].missing_reason == "counter_reset"


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan")])
def test_invalid_sampling_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="interval"):
        start_sampling(SequenceProvider([]), "GPU", interval)