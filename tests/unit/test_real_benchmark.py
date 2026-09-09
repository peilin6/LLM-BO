import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from dibo.benchmark import VLLMHttpBenchmark, load_requests
from dibo.schemas import load_experiment

CONFIGS = Path(__file__).parents[2] / "configs"


def test_request_manifest_requires_messages(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    assert load_requests(path)[0]["messages"][0]["content"] == "hi"
    path.write_text('{"prompt": "bad"}\n')
    with pytest.raises(TypeError, match="messages"):
        load_requests(path)


@pytest.mark.asyncio
async def test_streaming_benchmark_captures_ttft_and_usage(tmp_path, monkeypatch) -> None:
    request_file = tmp_path / "requests.jsonl"
    request_file.write_text(json.dumps({"messages": [{"role": "user", "content": "hello"}]}) + "\n")
    workload = load_experiment(CONFIGS / "experiment_smoke.yaml").workload.model_copy(
        update={"request_file": str(request_file), "num_prompts": 2, "warmup_prompts": 1,
                "request_rate": 1000.0, "max_concurrency": 2, "output_len": 4}
    )
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        body = (
            'data: {"choices":[{"delta":{"content":"x"}}]}\n'
            'data: {"choices":[],"usage":{"completion_tokens":4}}\n'
            "data: [DONE]\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    handle = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=8000,
                                                    served_model_name="dibo-model"))
    result = await VLLMHttpBenchmark(request_timeout_s=2).run_requests(handle, workload, warmup=False)

    assert len(result["requests"]) == 2
    assert all(item["success"] and item["output_tokens"] == 4 and item["ttft_s"] >= 0
               for item in result["requests"])
    assert all(payload["stream_options"] == {"include_usage": True} for payload in seen)
    assert all(payload["model"] == "dibo-model" and payload["ignore_eos"] is True for payload in seen)


def test_burstiness_schedule_is_seeded() -> None:
    first = VLLMHttpBenchmark(seed=7)
    second = VLLMHttpBenchmark(seed=7)
    assert first.seed == second.seed == 7