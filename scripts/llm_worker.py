"""Process one DIBO LLM review request from a shared directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dibo.llm_update import call_llm_api, process_review_request

DEFAULT_MAX_REQUEST_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="Path to one pending request JSON file")
    parser.add_argument("responses_dir", type=Path, help="Directory for the atomic response JSON")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible chat completion endpoint")
    parser.add_argument("--model", required=True, help="LLM model name")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--api-key-env", default="DIBO_LLM_API_KEY", help="API key environment variable")
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BYTES,
        help="Maximum accepted request file size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_request_bytes <= 0:
        raise SystemExit("--max-request-bytes must be positive")
    if args.request.stat().st_size > args.max_request_bytes:
        raise SystemExit("request exceeds --max-request-bytes")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")

    process_review_request(
        args.request,
        args.responses_dir,
        lambda payload: call_llm_api(
            payload,
            endpoint=args.endpoint,
            model=args.model,
            api_key=api_key,
            timeout_s=args.timeout,
        ),
    )


if __name__ == "__main__":
    main()