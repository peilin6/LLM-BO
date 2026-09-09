import json
from pathlib import Path

from scripts.prepare_workload import main as workload_main


def test_checked_in_workload_is_valid_jsonl() -> None:
    path = Path(__file__).parents[2] / "data/requests/qwen_smoke.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) >= 8
    assert all(row["messages"][0]["role"] == "system" for row in rows)


def test_prepare_workload_help_entry_exists() -> None:
    assert callable(workload_main)