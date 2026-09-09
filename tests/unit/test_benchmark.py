from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dibo.benchmark import aggregate_measurement, run_benchmark, run_measurement, run_warmup
from dibo.schemas import load_experiment

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeBenchmarkHandle:
    def __init__(self, measurement: dict[str, Any]) -> None:
        self.measurement = measurement
        self.calls: list[bool] = []

    async def run_requests(self, workload, *, warmup: bool) -> dict[str, Any]:
        self.calls.append(warmup)
        if warmup:
            return {
                "duration_s": 1.0,
                "requests": [
                    {"success": True, "output_tokens": 99999, "ttft_s": 99.0},
                ],
            }
        return self.measurement


def test_success_fixture_uses_actual_tokens_and_linear_p95() -> None:
    result = aggregate_measurement(load_fixture("benchmark_success.json"))

    assert result.completed_requests == 4
    assert result.output_tokens == 400
    assert result.throughput_tps == 100.0
    assert result.ttft_p95_s == pytest.approx(0.385)
    assert result.success_rate == 1.0


def test_partial_failure_keeps_total_as_success_rate_denominator() -> None:
    result = aggregate_measurement(load_fixture("benchmark_partial_failure.json"))

    assert result.completed_requests == 2
    assert result.failed_requests == 2
    assert result.throughput_tps == 60.0
    assert result.success_rate == 0.5
    assert result.ttft_p95_s == pytest.approx(0.39)


@pytest.mark.asyncio
async def test_warmup_is_excluded_from_formal_measurement() -> None:
    workload = load_experiment(ROOT / "configs" / "experiment_smoke.yaml").workload
    handle = FakeBenchmarkHandle(load_fixture("benchmark_success.json"))

    result = await run_benchmark(handle, workload)

    assert handle.calls == [True, False]
    assert result.output_tokens == 400
    assert result.ttft_p95_s is not None and result.ttft_p95_s < 1.0


@pytest.mark.asyncio
async def test_warmup_and_measurement_can_be_controlled_separately() -> None:
    workload = load_experiment(ROOT / "configs" / "experiment_smoke.yaml").workload
    handle = FakeBenchmarkHandle(load_fixture("benchmark_success.json"))

    await run_warmup(handle, workload)
    assert handle.calls == [True]

    result = await run_measurement(handle, workload)
    assert handle.calls == [True, False]
    assert result.throughput_tps == 100.0


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"duration_s": 0, "requests": [{"success": True}]}, "duration"),
        ({"duration_s": 1, "requests": []}, "non-empty"),
        (
            {"duration_s": 1, "requests": [{"success": True, "output_tokens": 1.5, "ttft_s": 0.1}]},
            "output_tokens",
        ),
        (
            {"duration_s": 1, "requests": [{"success": True, "output_tokens": 1, "ttft_s": float("nan")}]},
            "ttft_s",
        ),
    ],
)
def test_invalid_measurements_are_rejected(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_measurement(payload)