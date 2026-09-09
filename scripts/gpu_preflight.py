"""Validate an allocated GPU, local model, workload and DIBO real configuration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from dibo.controller import load_controller_config
from dibo.schemas import RunMode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_controller_config(args.config)
    if config.experiment.run_mode != RunMode.REAL:
        raise SystemExit("GPU preflight requires a real configuration")
    import pynvml

    pynvml.nvmlInit()
    try:
        detected = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            detected.append({"index": index, "uuid": uuid, "name": name,
                             "memory_total_bytes": memory.total})
    finally:
        pynvml.nvmlShutdown()
    assigned = set(config.experiment.engine.allocated_gpu_uuids)
    available = {item["uuid"] for item in detected}
    missing = sorted(assigned - available)
    result = {
        "ok": not missing and torch.cuda.is_available() and shutil.which("vllm") is not None,
        "assigned_gpu_uuids": sorted(assigned),
        "detected_gpus": detected,
        "missing_assigned_uuids": missing,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "vllm_executable": shutil.which("vllm"),
        "model": config.experiment.engine.model,
        "workload": config.experiment.workload.request_file,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()