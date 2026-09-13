"""Minimal vLLM subprocess lifecycle management."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

import httpx

from dibo.schemas import PARAMETER_IDS, EngineAdapter, EngineSpec


class ProcessHandle(Protocol):
    pid: int
    returncode: int | None

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass
class EngineHandle:
    process: ProcessHandle
    config: EngineSpec
    final_config: dict[str, Any]
    log_path: Path
    log_file: TextIO
    started_at: float
    stopped: bool = False


@dataclass(frozen=True)
class EngineLaunchSpec:
    engine: EngineSpec
    final_config: Mapping[str, Any]


PARAMETER_FLAGS = {
    "p01": "--tensor-parallel-size",
    "p02": "--pipeline-parallel-size",
    "p03": "--max-num-seqs",
    "p04": "--max-num-batched-tokens",
    "p05": "--block-size",
    "p06": "--gpu-memory-utilization",
    "p07": "--swap-space",
    "p08": "--cpu-offload-gb",
    "p09": "--max-num-partial-prefills",
    "p10": "--max-long-partial-prefills",
    "p11": "--long-prefill-token-threshold",
    "p12": "--enable-chunked-prefill",
    "p13": "--enable-prefix-caching",
    "p14": "--disable-custom-all-reduce",
    "p15": "--enforce-eager",
}


def build_argv(config: EngineSpec, final_config: Mapping[str, Any]) -> list[str]:
    """Build one argv list from the already compiled final configuration."""
    if config.adapter != EngineAdapter.VLLM_SUBPROCESS:
        raise ValueError("only the vllm_subprocess adapter can build a real argv")
    if tuple(final_config) != PARAMETER_IDS:
        raise ValueError("final_config must contain p01 through p15 in order")
    if config.execution_mode not in {"V0", "V1"}:
        raise ValueError("execution_mode must be resolved before engine start")

    argv = [
        "vllm",
        "serve",
        config.model,
        "--tokenizer",
        config.tokenizer,
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
    ]
    for parameter_id in PARAMETER_IDS:
        value = final_config[parameter_id]
        flag = PARAMETER_FLAGS[parameter_id]
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            argv.extend((flag, str(value)))
    return argv


def cuda_visible_devices(configured_devices: tuple[str, ...]) -> str:
    """Return a vLLM-compatible CUDA_VISIBLE_DEVICES value.

    NVML metrics keep using the configured GPU UUIDs, but vLLM 0.11.x on WSL
    expects CUDA visible device ids to parse as integer ordinals.
    """
    if all(device.startswith("GPU-") for device in configured_devices):
        return ",".join(str(index) for index, _ in enumerate(configured_devices))
    return ",".join(configured_devices)


async def start(config: EngineLaunchSpec, run_dir: Path) -> EngineHandle:
    """Start one owned vLLM process without invoking a shell."""
    engine = config.engine
    argv = build_argv(engine, config.final_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "engine.log"
    log_file = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(engine.allocated_gpu_uuids)
    environment["VLLM_USE_V1"] = "1" if engine.execution_mode == "V1" else "0"
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return EngineHandle(
        process=process,
        config=engine,
        final_config=dict(config.final_config),
        log_path=log_path,
        log_file=log_file,
        started_at=time.monotonic(),
    )


def detect_execution_mode(log_text: str) -> str | None:
    """Extract explicit V0/V1 evidence without guessing from package version."""
    normalized = log_text.lower()
    has_v1 = (
        "vllm v1" in normalized
        or "engine core initialization" in normalized
        or "initializing a v1 llm engine" in normalized
    )
    has_v0 = "vllm v0" in normalized or "legacy llm engine" in normalized
    if has_v1 == has_v0:
        return None
    return "V1" if has_v1 else "V0"


def engine_exit_detail(log_path: Path) -> str | None:
    """Return the final meaningful vLLM log line for a startup failure event."""
    if not log_path.exists():
        return None
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    lines = [
        ansi_escape.sub("", line).strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    for line in reversed(lines):
        normalized = line.lower()
        if (
            (
                "valueerror:" in normalized
                or "notimplementederror:" in normalized
                or "assertionerror:" in normalized
            )
            and "engine core initialization failed" not in normalized
        ):
            return line
    return next((line for line in reversed(lines) if line), None)


async def _default_get(url: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=2.0) as client:
        return await client.get(url)


async def wait_ready(
    handle: EngineHandle,
    timeout_s: float,
    *,
    get: Callable[[str], Awaitable[Any]] = _default_get,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Wait for health, model identity, and explicit execution-mode evidence."""
    deadline = time.monotonic() + timeout_s
    base_url = f"http://{handle.config.host}:{handle.config.port}"
    last_error = "engine did not become ready"
    while time.monotonic() < deadline:
        if handle.process.returncode is not None:
            detail = engine_exit_detail(handle.log_path)
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"vLLM exited before readiness with code {handle.process.returncode}{suffix}"
            )
        try:
            health = await get(f"{base_url}/health")
            if health.status_code == 200:
                models = await get(f"{base_url}/v1/models")
                model_ids = {item["id"] for item in models.json().get("data", [])}
                if handle.config.served_model_name not in model_ids:
                    raise RuntimeError("ready endpoint returned a different model identity")
                handle.log_file.flush()
                log_text = handle.log_path.read_text(encoding="utf-8")
                actual_mode = detect_execution_mode(log_text)
                if actual_mode is None:
                    raise RuntimeError("vLLM execution mode is not confirmed by engine evidence")
                if actual_mode != handle.config.execution_mode:
                    raise RuntimeError("vLLM execution mode differs from configuration")
                return actual_mode
        except (httpx.HTTPError, OSError, ValueError) as error:
            last_error = str(error)
        await sleep(0.1)
    raise TimeoutError(last_error)


async def stop(handle: EngineHandle, timeout_s: float = 10.0) -> None:
    """Idempotently stop only the process group created for this handle."""
    if handle.stopped:
        return
    try:
        if handle.process.returncode is None:
            os.killpg(handle.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=timeout_s)
            except TimeoutError:
                os.killpg(handle.process.pid, signal.SIGKILL)
                await handle.process.wait()
    finally:
        handle.log_file.close()
        handle.stopped = True
