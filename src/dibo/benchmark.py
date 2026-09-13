"""Benchmark adapter boundary and deterministic result aggregation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from dibo.engine import EngineHandle
from dibo.schemas import WorkloadSpec


class BenchmarkHandle(Protocol):
    async def run_requests(
        self,
        workload: WorkloadSpec,
        *,
        warmup: bool,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BenchmarkResult:
    total_requests: int
    completed_requests: int
    failed_requests: int
    output_tokens: int
    duration_s: float
    throughput_tps: float
    ttft_p95_s: float | None
    success_rate: float
    request_results: tuple[dict[str, Any], ...]
    configured_request_rate_rps: float | None = None
    issued_request_rate_rps: float | None = None
    completed_request_rate_rps: float | None = None
    launch_span_s: float | None = None


def load_requests(path: Path) -> tuple[dict[str, Any], ...]:
    """Load a non-empty JSONL request manifest without silently repairing rows."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on request line {line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise TypeError(f"request line {line_number} requires a messages list")
            rows.append(row)
    if not rows:
        raise ValueError("request manifest is empty")
    return tuple(rows)


def _request_payload(row: dict[str, Any], handle: EngineHandle, workload: WorkloadSpec) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        model=handle.config.served_model_name,
        stream=workload.streaming,
        max_tokens=workload.output_len,
        ignore_eos=workload.ignore_eos,
    )
    if workload.streaming:
        payload["stream_options"] = {"include_usage": True}
    return payload


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    output_tokens: int | None = None
    try:
        if payload["stream"]:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content") and first_token_at is None:
                            first_token_at = time.perf_counter()
                    usage = chunk.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])
        else:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            first_token_at = time.perf_counter()
            usage = body.get("usage") or {}
            output_tokens = int(usage["completion_tokens"])
        if first_token_at is None:
            raise ValueError("response contained no generated token")
        if output_tokens is None:
            raise ValueError("response did not include completion token usage")
        return {
            "request_id": request_id,
            "success": True,
            "output_tokens": output_tokens,
            "ttft_s": first_token_at - started,
            "duration_s": time.perf_counter() - started,
        }
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return {
            "request_id": request_id,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "duration_s": time.perf_counter() - started,
        }


class VLLMHttpBenchmark:
    """Rate-limited OpenAI chat workload driver with per-request TTFT evidence."""

    def __init__(self, *, request_timeout_s: float = 600.0, seed: int = 42) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.request_timeout_s = request_timeout_s
        self.seed = seed

    async def run_requests(
        self,
        handle: EngineHandle,
        workload: WorkloadSpec,
        *,
        warmup: bool,
    ) -> dict[str, Any]:
        rows = load_requests(Path(workload.request_file))
        count = workload.warmup_prompts if warmup else workload.num_prompts
        selected: Sequence[dict[str, Any]] = tuple(rows[index % len(rows)] for index in range(count))
        if not selected:
            return {"duration_s": 0.0, "requests": []}
        semaphore = asyncio.Semaphore(workload.max_concurrency)
        url = f"http://{handle.config.host}:{handle.config.port}{workload.endpoint}"
        random = np.random.default_rng(self.seed + int(warmup))
        if workload.burstiness == float("inf"):
            delays = np.full(count, 1.0 / workload.request_rate)
        else:
            delays = random.gamma(
                shape=workload.burstiness,
                scale=1.0 / (workload.request_rate * workload.burstiness),
                size=count,
            )
        launch_offsets = np.concatenate(([0.0], np.cumsum(delays[:-1])))
        started = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            async def scheduled(index: int, row: dict[str, Any]) -> dict[str, Any]:
                target = started + float(launch_offsets[index])
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                async with semaphore:
                    return await _one_request(
                        client,
                        url,
                        _request_payload(row, handle, workload),
                        f"{'warmup' if warmup else 'request'}_{index + 1:04}",
                    )

            results = await asyncio.gather(
                *(scheduled(index, row) for index, row in enumerate(selected))
            )
        duration_s = time.perf_counter() - started
        launch_span_s = float(launch_offsets[-1]) if len(launch_offsets) else 0.0
        return {
            "duration_s": duration_s,
            "requests": results,
            "configured_request_rate_rps": workload.request_rate,
            "issued_request_rate_rps": (
                count / launch_span_s if launch_span_s > 0 else float(count) / duration_s
            ),
            "launch_span_s": launch_span_s,
            "max_concurrency": workload.max_concurrency,
            "burstiness": workload.burstiness,
        }


def aggregate_measurement(payload: dict[str, Any]) -> BenchmarkResult:
    """Aggregate only one formal measurement payload."""
    requests = payload.get("requests")
    duration_s = payload.get("duration_s")
    if not isinstance(requests, list) or not requests:
        raise ValueError("measurement requires non-empty request results")
    if not isinstance(duration_s, (int, float)) or not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("measurement duration must be positive and finite")

    completed = [item for item in requests if item.get("success") is True]
    output_tokens = 0
    ttft_values: list[float] = []
    for item in completed:
        tokens = item.get("output_tokens")
        ttft = item.get("ttft_s")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError("successful requests require non-negative integer output_tokens")
        if not isinstance(ttft, (int, float)) or not isfinite(ttft) or ttft < 0:
            raise ValueError("successful requests require finite non-negative ttft_s")
        output_tokens += tokens
        ttft_values.append(float(ttft))

    total = len(requests)
    completed_count = len(completed)
    completed_request_rate_rps = completed_count / float(duration_s)
    configured_request_rate = payload.get("configured_request_rate_rps")
    issued_request_rate = payload.get("issued_request_rate_rps")
    launch_span = payload.get("launch_span_s")
    return BenchmarkResult(
        total_requests=total,
        completed_requests=completed_count,
        failed_requests=total - completed_count,
        output_tokens=output_tokens,
        duration_s=float(duration_s),
        throughput_tps=output_tokens / float(duration_s),
        ttft_p95_s=(float(np.quantile(ttft_values, 0.95, method="linear")) if ttft_values else None),
        success_rate=completed_count / total,
        request_results=tuple(dict(item) for item in requests),
        configured_request_rate_rps=(
            float(configured_request_rate)
            if isinstance(configured_request_rate, (int, float)) and isfinite(configured_request_rate)
            else None
        ),
        issued_request_rate_rps=(
            float(issued_request_rate)
            if isinstance(issued_request_rate, (int, float)) and isfinite(issued_request_rate)
            else None
        ),
        completed_request_rate_rps=completed_request_rate_rps,
        launch_span_s=(
            float(launch_span)
            if isinstance(launch_span, (int, float)) and isfinite(launch_span)
            else None
        ),
    )


async def run_benchmark(handle: BenchmarkHandle, workload: WorkloadSpec) -> BenchmarkResult:
    """Run an isolated warmup followed by one formal measurement window."""
    await run_warmup(handle, workload)
    return await run_measurement(handle, workload)


async def run_warmup(handle: BenchmarkHandle, workload: WorkloadSpec) -> None:
    """Run warmup requests without returning model input data."""
    await handle.run_requests(workload, warmup=True)


async def run_measurement(handle: BenchmarkHandle, workload: WorkloadSpec) -> BenchmarkResult:
    """Run and aggregate only the formal measurement window."""
    measurement = await handle.run_requests(workload, warmup=False)
    return aggregate_measurement(measurement)
