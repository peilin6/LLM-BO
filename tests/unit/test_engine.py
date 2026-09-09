from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

import pytest

from dibo.engine import (
    EngineHandle,
    EngineLaunchSpec,
    build_argv,
    detect_execution_mode,
    start,
    stop,
    wait_ready,
)
from dibo.schemas import EngineAdapter, EngineSpec


class FakeProcess:
    def __init__(self, pid: int = 4321, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.wait_count = 0

    async def wait(self) -> int:
        self.wait_count += 1
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.returncode = -signal.SIGKILL


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def engine_spec(**updates: Any) -> EngineSpec:
    values: dict[str, Any] = {
        "adapter": EngineAdapter.VLLM_SUBPROCESS,
        "version": "0.11.2",
        "execution_mode": "V1",
        "model": "/models/model with spaces",
        "tokenizer": "/models/tokenizer with spaces",
        "model_revision": "revision",
        "dtype": "bfloat16",
        "served_model_name": "dibo-model",
        "host": "127.0.0.1",
        "port": 8000,
        "max_model_len": 8192,
        "allocated_gpu_count": 1,
        "allocated_gpu_uuids": ("GPU-TEST-UUID",),
        "startup_timeout_s": 10,
        "benchmark_timeout_s": 10,
    }
    values.update(updates)
    return EngineSpec.model_validate(values)


def final_config() -> dict[str, object]:
    return {
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
        "p13": False,
        "p14": False,
        "p15": False,
    }


def make_handle(tmp_path: Path, *, process: FakeProcess | None = None) -> EngineHandle:
    log_path = tmp_path / "engine.log"
    log_path.write_text("vLLM V1 engine core initialization\n", encoding="utf-8")
    log_file = log_path.open("a", encoding="utf-8")
    return EngineHandle(
        process=process or FakeProcess(),
        config=engine_spec(),
        final_config=final_config(),
        log_path=log_path,
        log_file=log_file,
        started_at=0.0,
    )


def test_build_argv_uses_final_values_without_shell_or_recompilation() -> None:
    argv = build_argv(engine_spec(), final_config())

    assert argv[:3] == ["vllm", "serve", "/models/model with spaces"]
    assert argv[argv.index("--tokenizer") + 1] == "/models/tokenizer with spaces"
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
    assert "--enable-chunked-prefill" in argv
    assert "--no-enable-prefix-caching" in argv
    assert "--no-disable-custom-all-reduce" in argv
    assert "--no-enforce-eager" in argv
    assert "shell=True" not in argv


def test_build_argv_rejects_fake_adapter_and_unresolved_mode() -> None:
    with pytest.raises(ValueError, match="vllm_subprocess"):
        build_argv(engine_spec(adapter=EngineAdapter.FAKE), final_config())
    with pytest.raises(ValueError, match="execution_mode"):
        build_argv(
            engine_spec(execution_mode="REPLACE_AFTER_SMOKE_WITH_V1_OR_V0"),
            final_config(),
        )


@pytest.mark.asyncio
async def test_start_uses_owned_session_and_only_assigned_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    process = FakeProcess()

    async def fake_create(*argv: str, **kwargs: Any) -> FakeProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    launch = EngineLaunchSpec(engine=engine_spec(), final_config=final_config())

    handle = await start(launch, tmp_path / "run with spaces")

    assert captured["argv"][2] == "/models/model with spaces"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-TEST-UUID"
    assert captured["kwargs"]["env"]["VLLM_USE_V1"] == "1"
    assert handle.process is process
    handle.log_file.close()


@pytest.mark.asyncio
async def test_wait_ready_checks_health_model_and_execution_mode(tmp_path: Path) -> None:
    handle = make_handle(tmp_path)

    async def fake_get(url: str) -> FakeResponse:
        if url.endswith("/health"):
            return FakeResponse(200)
        return FakeResponse(200, {"data": [{"id": "dibo-model"}]})

    try:
        assert await wait_ready(handle, 1.0, get=fake_get) == "V1"
    finally:
        handle.log_file.close()


@pytest.mark.asyncio
async def test_wait_ready_rejects_wrong_model_and_early_exit(tmp_path: Path) -> None:
    handle = make_handle(tmp_path)

    async def wrong_model(url: str) -> FakeResponse:
        if url.endswith("/health"):
            return FakeResponse(200)
        return FakeResponse(200, {"data": [{"id": "other-model"}]})

    try:
        with pytest.raises(RuntimeError, match="different model"):
            await wait_ready(handle, 1.0, get=wrong_model)
    finally:
        handle.log_file.close()

    exited = make_handle(tmp_path, process=FakeProcess(returncode=3))
    try:
        with pytest.raises(RuntimeError, match="code 3"):
            await wait_ready(exited, 1.0)
    finally:
        exited.log_file.close()


@pytest.mark.asyncio
async def test_wait_ready_rejects_mode_mismatch_and_times_out(tmp_path: Path) -> None:
    handle = make_handle(tmp_path)
    handle.config = engine_spec(execution_mode="V0")

    async def ready(url: str) -> FakeResponse:
        if url.endswith("/health"):
            return FakeResponse(200)
        return FakeResponse(200, {"data": [{"id": "dibo-model"}]})

    try:
        with pytest.raises(RuntimeError, match="differs from configuration"):
            await wait_ready(handle, 1.0, get=ready)
        with pytest.raises(TimeoutError):
            await wait_ready(handle, 0.0, get=ready)
    finally:
        handle.log_file.close()


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_signals_only_owned_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = make_handle(tmp_path)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("dibo.engine.os.killpg", lambda pid, sig: signals.append((pid, sig)))

    await stop(handle)
    await stop(handle)

    assert signals == [(4321, signal.SIGTERM)]
    assert handle.process.wait_count == 1
    assert handle.stopped is True
    assert handle.log_file.closed is True


def test_detect_execution_mode_requires_unambiguous_evidence() -> None:
    assert detect_execution_mode("vLLM V1 engine core initialization") == "V1"
    assert detect_execution_mode("legacy LLM engine uses vLLM V0") == "V0"
    assert detect_execution_mode("no mode evidence") is None
    assert detect_execution_mode("vLLM V0 then vLLM V1") is None