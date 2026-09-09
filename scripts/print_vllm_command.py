"""Print the exact zero-Action vLLM command and assigned-device environment."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from dibo.compiler import compile_config
from dibo.controller import load_controller_config
from dibo.engine import build_argv
from dibo.schemas import ACTION_IDS, EngineAdapter, RunMode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_controller_config(args.config)
    if config.experiment.run_mode != RunMode.REAL:
        raise SystemExit("command rendering requires a real configuration")
    trace = compile_config(
        config.x_init,
        config.actions,
        ACTION_IDS,
        {action_id: 0.0 for action_id in ACTION_IDS},
        config.environment,
        phase="initial",
        bo_round=None,
        compile_base_id="x_init",
    )
    engine = config.experiment.engine
    if engine.adapter != EngineAdapter.VLLM_SUBPROCESS:
        raise SystemExit("real config must use vllm_subprocess")
    print(f"export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(engine.allocated_gpu_uuids))}")
    print(f"export VLLM_USE_V1={'1' if engine.execution_mode == 'V1' else '0'}")
    print(shlex.join(build_argv(engine, trace.final_config)))


if __name__ == "__main__":
    main()