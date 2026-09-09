import hashlib
import json
from pathlib import Path

import pytest
import yaml

from dibo.controller import RealTrialRunner, load_controller_config
from dibo.schemas import RunMode

CONFIGS = Path(__file__).parents[2] / "configs"


def real_fixture(tmp_path: Path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_attention_heads": 28,
                                                    "num_hidden_layers": 28}))
    workload = tmp_path / "requests.jsonl"
    workload.write_text('{"messages":[{"role":"user","content":"hello"}]}\n')
    payload = yaml.safe_load((CONFIGS / "experiment.yaml").read_text())
    payload["experiment_id"] = "real_fixture"
    payload["engine"].update(model=str(model), tokenizer=str(model), model_revision="commit123",
                             execution_mode="V1", allocated_gpu_uuids=["GPU-12345678"])
    payload["workload"].update(request_file=str(workload),
                               request_sha256=hashlib.sha256(workload.read_bytes()).hexdigest())
    payload["llm"]["model"] = "approved-model"
    payload["thresholds"]["m04"] = {"kind": "upper", "trigger": 8.0,
                                      "target": 4.0, "scale": 2.0}
    for name in ("parameters.yaml", "actions_v1.yaml", "graph.yaml"):
        (tmp_path / name).write_text((CONFIGS / name).read_text())
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            return [1, 2, 3, 4]

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *args, **kwargs: Tokenizer())
    return config_path, workload


def test_real_loader_resolves_local_model_workload_and_real_uuid(tmp_path, monkeypatch) -> None:
    path, _ = real_fixture(tmp_path, monkeypatch)
    config = load_controller_config(path, run_root=tmp_path / "runs")

    assert config.experiment.run_mode == RunMode.REAL
    assert config.experiment.engine.allocated_gpu_uuids == ("GPU-12345678",)
    assert config.environment.num_attention_heads == 28
    assert config.x_init["p01"] == 1
    assert config.x_init["p04"] == 8192
    assert config.x_init["p11"] == 256
    assert isinstance(RealTrialRunner(), RealTrialRunner)


def test_real_loader_rejects_changed_workload_hash(tmp_path, monkeypatch) -> None:
    path, workload = real_fixture(tmp_path, monkeypatch)
    workload.write_text('{"messages":[{"role":"user","content":"changed"}]}\n')
    with pytest.raises(ValueError, match="SHA256"):
        load_controller_config(path, run_root=tmp_path / "runs")