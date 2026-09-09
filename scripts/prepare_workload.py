"""Generate a deterministic chat JSONL workload and print its SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=128)
    args = parser.parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be positive")
    topics = (
        "batching and throughput",
        "time to first token",
        "KV cache management",
        "request queue growth",
        "prefix caching",
        "chunked prefill",
        "GPU utilization",
        "tail latency",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index in range(args.rows):
            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise technical assistant. Give accurate answers.",
                    },
                    {
                        "role": "user",
                        "content": f"Request {index + 1}: explain {topics[index % len(topics)]} in practical terms.",
                    },
                ]
            }
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"path": str(args.output.resolve()), "rows": args.rows, "sha256": digest}))


if __name__ == "__main__":
    main()