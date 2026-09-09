from __future__ import annotations

import json
from pathlib import Path

import pytest

from dibo.compiler import CompileEnvironment, compile_config
from dibo.schemas import RunMode, TrialResult, TrialStatus, load_actions, load_parameters
from dibo.store import load_trials, save_json_atomic, save_trial

CONFIGS = Path(__file__).parents[2] / "configs"


def make_result(trial_id: str = "trial_001") -> TrialResult:
    base = {
        "p01": 1,
        "p02": 1,
        "p03": 128,
        "p04": 8192,
        "p05": 16,
        "p06": 0.90,
        "p07": 4.0,
        "p08": 2.0,
        "p09": 2,
        "p10": 1,
        "p11": 2048,
        "p12": True,
        "p13": True,
        "p14": False,
        "p15": False,
    }
    environment = CompileEnvironment(
        parameter_catalog=load_parameters(CONFIGS / "parameters.yaml"),
        allocated_gpu_count=1,
        num_attention_heads=28,
        num_hidden_layers=28,
        max_model_len=8192,
        supported_block_sizes=(8, 16, 32),
    )
    trace = compile_config(
        base,
        load_actions(CONFIGS / "actions_v1.yaml"),
        (),
        {},
        environment,
        trial_id=trial_id,
        phase="initial",
    )
    return TrialResult(
        trial_id=trial_id,
        run_mode=RunMode.SYNTHETIC,
        status=TrialStatus.SUCCESS,
        trace=trace,
        metrics={metric: 0.1 for metric in ("m01", "m02", "m03", "m04", "m05", "m06")},
        metric_missing_reasons={},
        throughput_tps=100.0,
        request_count=10,
        completed_requests=10,
        successful_requests=10,
        duration_s=1.0,
        sampling_interval_s=1.0,
        valid_sample_counts={metric: 3 for metric in trace.selected_metrics},
        log_paths={},
        cleanup_result="not_required",
    )


def test_atomic_json_write(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "value.json"
    save_json_atomic(target, {"value": 1})

    assert json.loads(target.read_text()) == {"value": 1}
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_json_does_not_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "value.json"
    save_json_atomic(target, {"value": 1})

    with pytest.raises(FileExistsError):
        save_json_atomic(target, {"value": 2})

    assert json.loads(target.read_text()) == {"value": 1}


def test_save_and_load_trials_in_stable_order(tmp_path: Path) -> None:
    save_trial(make_result("trial_002"), tmp_path)
    save_trial(make_result("trial_001"), tmp_path)

    assert [result.trial_id for result in load_trials(tmp_path)] == ["trial_001", "trial_002"]


def test_duplicate_trial_id_is_not_overwritten(tmp_path: Path) -> None:
    result = make_result()
    path = save_trial(result, tmp_path)

    with pytest.raises(FileExistsError):
        save_trial(result, tmp_path)

    assert TrialResult.model_validate_json(path.read_text()).trial_id == "trial_001"
