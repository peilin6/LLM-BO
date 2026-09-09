"""Atomic local persistence for DIBO runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dibo.schemas import TrialResult


def save_json_atomic(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    """Write JSON through a sibling temporary file and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_trial(result: TrialResult, run_dir: Path) -> Path:
    """Persist one immutable TrialResult exactly once."""
    trial_dir = run_dir / "trials" / result.trial_id
    result_path = trial_dir / "result.json"
    save_json_atomic(result_path, result.model_dump(mode="json"))
    return result_path


def load_trials(run_dir: Path) -> list[TrialResult]:
    """Load saved trials in stable trial-id order."""
    trials_dir = run_dir / "trials"
    if not trials_dir.exists():
        return []
    results: list[TrialResult] = []
    for result_path in sorted(trials_dir.glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as file:
            results.append(TrialResult.model_validate(json.load(file)))
    return results
