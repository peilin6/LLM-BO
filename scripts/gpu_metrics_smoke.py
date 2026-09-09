"""Run one zero-coefficient real Trial and print the persisted metrics paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dibo.compiler import compile_config
from dibo.controller import RealTrialRunner, load_controller_config
from dibo.engine import EngineLaunchSpec
from dibo.schemas import ACTION_IDS, RunMode
from dibo.store import save_json_atomic
from dibo.trial import TrialSpec


async def run(config_path: Path, run_root: Path, prompts: int, warmup: int):
    config = load_controller_config(config_path, run_root=run_root)
    if config.experiment.run_mode != RunMode.REAL:
        raise ValueError("metrics smoke requires a real configuration")
    run_dir = run_root / f"metrics_smoke_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    experiment = config.experiment.model_copy(update={"workload": config.experiment.workload.model_copy(
        update={"num_prompts": prompts, "warmup_prompts": warmup}
    ), "metrics": config.experiment.metrics.model_copy(
        update={"sampling_interval_s": min(config.experiment.metrics.sampling_interval_s, 0.5),
                "min_valid_samples": min(config.experiment.metrics.min_valid_samples, 3)}
    )})
    trace = compile_config(
        config.x_init,
        config.actions,
        ACTION_IDS,
        {action_id: 0.0 for action_id in ACTION_IDS},
        config.environment,
        trial_id="trial_001",
        phase="initial",
        bo_round=None,
        compile_base_id="x_init",
    )
    result = await RealTrialRunner(seed=experiment.seed)(TrialSpec(
        run_mode=RunMode.REAL,
        trace=trace,
        engine_launch=EngineLaunchSpec(experiment.engine, trace.final_config),
        workload=experiment.workload,
        run_dir=run_dir,
        startup_timeout_s=experiment.engine.startup_timeout_s,
        benchmark_timeout_s=experiment.engine.benchmark_timeout_s,
        gpu_uuid=experiment.engine.allocated_gpu_uuids[0],
        sampling_interval_s=experiment.metrics.sampling_interval_s,
        min_valid_samples=experiment.metrics.min_valid_samples,
    ))
    save_json_atomic(run_dir / "smoke_summary.json", {
        "status": result.status.value,
        "metrics": result.metrics,
        "missing_reasons": result.metric_missing_reasons,
        "throughput_tps": result.throughput_tps,
        "trial_dir": str(run_dir / "trials" / result.trial_id),
    })
    return result, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    if args.prompts < 1 or args.warmup < 0:
        raise SystemExit("prompt counts are invalid")
    result, run_dir = asyncio.run(run(args.config, args.run_root, args.prompts, args.warmup))
    print(json.dumps({
        "status": result.status.value,
        "metrics": result.metrics,
        "throughput_tps": result.throughput_tps,
        "raw_metrics": str(run_dir / "trials/trial_001/metrics.jsonl"),
        "metrics_summary": str(run_dir / "trials/trial_001/metrics_summary.json"),
        "result": str(run_dir / "trials/trial_001/result.json"),
    }, indent=2))
    if result.status.value != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()