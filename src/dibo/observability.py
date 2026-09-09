"""Small, dependency-free structured runtime logging for DIBO runs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def emit_event(run_dir: Path, event: str, /, **fields: Any) -> dict[str, Any]:
    """Append one JSONL event and print a concise, secret-free status line."""
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        output.flush()
    summary = " ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if key not in {"metrics", "targets", "weights", "error"}
    )
    print(
        f"[{record['timestamp']}] {event}" + (f" {summary}" if summary else ""),
        file=sys.stderr,
        flush=True,
    )
    return record


def stop_requested(run_dir: Path) -> bool:
    """Return whether an operator requested a safe stop between operations."""
    return (run_dir / "STOP").is_file()


def stop_message(run_dir: Path) -> str:
    """Read an optional operator message without exposing arbitrary large content."""
    path = run_dir / "STOP"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:500].strip()