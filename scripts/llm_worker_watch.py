"""Watch one run's llm_io directory and process each request exactly once."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dibo.llm_update import call_llm_api, process_review_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("io_dir", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--api-key-env", default="DIBO_LLM_API_KEY")
    parser.add_argument("--max-request-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.timeout <= 0 or args.poll_interval <= 0 or args.max_request_bytes <= 0:
        raise SystemExit("timeouts, polling interval and size limit must be positive")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    requests_dir = args.io_dir / "requests"
    responses_dir = args.io_dir / "responses"
    while True:
        if not requests_dir.exists():
            time.sleep(args.poll_interval)
            continue
        for request_path in sorted(requests_dir.glob("*.json")):
            if (responses_dir / request_path.name).exists():
                continue
            if request_path.stat().st_size > args.max_request_bytes:
                raise SystemExit(f"request exceeds size limit: {request_path}")
            responses_dir.mkdir(parents=True, exist_ok=True)
            process_review_request(
                request_path,
                responses_dir,
                lambda payload: call_llm_api(
                    payload,
                    endpoint=args.endpoint,
                    model=args.model,
                    api_key=api_key,
                    timeout_s=args.timeout,
                ),
            )
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()