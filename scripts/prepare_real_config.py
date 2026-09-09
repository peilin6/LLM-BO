"""Create a runnable real config from the checked-in template and assigned GPU UUIDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gpu-uuid", action="append", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("V0", "V1"), required=True)
    parser.add_argument("--llm-model", default="disabled")
    parser.add_argument("--m04-trigger", type=float, required=True)
    parser.add_argument("--m04-target", type=float, required=True)
    parser.add_argument("--m04-scale", type=float, required=True)
    parser.add_argument("--max-total-trials", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    if args.m04_scale <= 0:
        raise SystemExit("--m04-scale must be positive")
    payload = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    model_info = json.loads((args.model_dir / "DIBO_MODEL_INFO.json").read_text(encoding="utf-8"))
    payload["experiment_id"] = args.experiment_id
    payload["engine"].update(
        execution_mode=args.execution_mode,
        model=str(args.model_dir.resolve()),
        tokenizer=str(args.model_dir.resolve()),
        model_revision=model_info["revision"],
        allocated_gpu_count=len(args.gpu_uuid),
        allocated_gpu_uuids=args.gpu_uuid,
    )
    payload["workload"].update(
        request_file=str(args.request_file.resolve()),
        request_sha256=hashlib.sha256(args.request_file.read_bytes()).hexdigest(),
    )
    payload["llm"]["model"] = args.llm_model
    payload["thresholds"]["m04"] = {
        "kind": "upper",
        "trigger": args.m04_trigger,
        "target": args.m04_target,
        "scale": args.m04_scale,
    }
    if args.max_total_trials is not None:
        payload["tuning"]["max_total_trials"] = args.max_total_trials
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps({"config": str(args.output.resolve()), "request_sha256": payload["workload"]["request_sha256"]}))


if __name__ == "__main__":
    main()