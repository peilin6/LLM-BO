"""Download one immutable Hugging Face model snapshot into the shared model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("/home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty model directory: {args.output}")
    info = HfApi(endpoint=args.endpoint).model_info(args.repo)
    downloaded = snapshot_download(
        repo_id=args.repo,
        revision=info.sha,
        local_dir=args.output,
        endpoint=args.endpoint,
        max_workers=args.max_workers,
    )
    record = {"repo_id": args.repo, "revision": info.sha,
              "endpoint": args.endpoint, "local_dir": str(Path(downloaded).resolve())}
    (args.output / "DIBO_MODEL_INFO.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()