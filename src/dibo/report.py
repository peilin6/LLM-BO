"""Offline reporting from immutable Trial records."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dibo.schemas import METRIC_IDS, RunMode, TrialResult, TrialStatus
from dibo.store import load_trials


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)


def build_report(
    history: Sequence[TrialResult],
    base_trial: TrialResult | None,
    selections: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    *,
    run_mode: RunMode | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Report actual records only; reject mixed synthetic/real populations."""
    modes = {item.run_mode for item in history}
    if run_mode is not None:
        modes.add(RunMode(run_mode))
    if len(modes) != 1:
        raise ValueError(
            "report requires one explicit run type; mixed real/synthetic history is forbidden"
        )
    mode = next(iter(modes))
    if len({item.trial_id for item in history}) != len(history):
        raise ValueError("duplicate Trial IDs in report")
    if base_trial is not None and base_trial not in history:
        raise ValueError("base Trial is not present in history")
    successes = [item for item in history if item.status == TrialStatus.SUCCESS]
    best = (
        min(
            successes,
            key=lambda item: (
                -item.throughput_tps,
                item.metrics["m06"] if item.metrics["m06"] is not None else float("inf"),
                item.trial_id,
            ),
        )
        if successes
        else None
    )
    baseline = next(
        (
            item
            for item in history
            if item.trace.phase == "initial" and not any(item.trace.coefficients.values())
        ),
        None,
    )
    lines = [
        "# DIBO Experiment Report",
        "",
        f"Run type: **{mode.value}**",
        "",
        "Synthetic outcomes are test data, not vLLM measurements."
        if mode == RunMode.SYNTHETIC
        else "Recorded real Trial outcomes; predictions are not measurements.",
        "",
        f"Attempts: {len(history)}; success: {len(successes)}; failures: {len(history) - len(successes)}.",
        "",
        "## Environment And Budget",
        "",
        "```json",
        _json(dict(metadata or {})),
        "```",
        "",
    ]
    for title, trial in (
        ("x_init Baseline", baseline),
        ("Fixed x_base", base_trial),
        ("Best Recorded Trial", best),
    ):
        lines.extend([f"## {title}", ""])
        if trial is None:
            lines.extend(["Unavailable.", ""])
            continue
        lines.extend(
            [
                f"Trial: {trial.trial_id}; TPS: {trial.throughput_tps}; status: {trial.status.value}",
                "",
                "```json",
                _json(
                    {
                        "final_config": trial.trace.final_config,
                        "metrics": trial.metrics,
                        "config_hash": trial.trace.config_hash,
                    }
                ),
                "```",
                "",
            ]
        )
    for title, reference in (("x_init", baseline), ("x_base", base_trial)):
        if (
            best
            and reference
            and reference.throughput_tps
            and reference.status == TrialStatus.SUCCESS
        ):
            improvement = 100 * (best.throughput_tps / reference.throughput_tps - 1)
            lines.extend([f"TPS change against {title}: {improvement:.4f}% ({mode.value}).", ""])
    lines.extend(["## Selection And Predictions", ""])
    for selection in selections:
        if selection.get("selection_reason"):
            lines.extend(
                [
                    f"Round {selection.get('bo_round')}: {selection['selection_reason']}",
                    "",
                ]
            )
        lines.extend(["```json", _json(dict(selection)), "```", ""])
    lines.extend(["## Action Reviews", ""])
    for review in reviews:
        lines.extend(
            [
                "```json",
                _json(
                    {
                        "update": review.get("update"),
                        "rejection_reason": review.get("rejection_reason"),
                        "evidence_trial_ids": [
                            item["trial_id"]
                            for item in review.get("evidence", {}).get("trials", [])
                        ],
                    }
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Trial Outcomes",
            "",
            "| Trial | Type | Status | TPS | Issued rps | Completed rps | Queue p95 | Backlog | Version | Cleanup |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for trial in history:
        lines.append(
            f"| {trial.trial_id} | {trial.run_mode.value} | {trial.status.value} | {trial.throughput_tps} | {trial.issued_request_rate_rps} | {trial.completed_request_rate_rps} | {trial.queue_backlog_p95} | {trial.backlog_detected} | {trial.trace.action_version} | {trial.cleanup_result.replace('|', '/')} |"
        )
        lines.extend(
            [
                "",
                "```json",
                _json(
                    {
                        "metrics": trial.metrics,
                        "missing": trial.metric_missing_reasons,
                        "final_config": trial.trace.final_config,
                        "coefficients": trial.trace.coefficients,
                        "effective_parameter_delta": trial.trace.effective_parameter_delta,
                        "traffic": {
                            "request_count": trial.request_count,
                            "completed_requests": trial.completed_requests,
                            "successful_requests": trial.successful_requests,
                            "configured_request_rate_rps": trial.configured_request_rate_rps,
                            "issued_request_rate_rps": trial.issued_request_rate_rps,
                            "completed_request_rate_rps": trial.completed_request_rate_rps,
                            "queue_backlog_p95": trial.queue_backlog_p95,
                            "backlog_detected": trial.backlog_detected,
                        },
                    }
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "No GPU, model-loading, or real inference performance validation is implied by a synthetic run.",
            "Small samples, graph cross-effects, independent F posteriors and non-causal G counterfactuals limit interpretation.",
            "Missing metrics and not-ready models remain explicit; predicted TPS is never used as the best recorded Trial.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(run_dir: Path) -> Path:
    """Rebuild Markdown and CSV without importing the Controller or any engine."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    history = load_trials(run_dir)
    encoded_path = run_dir / "encoded_history.jsonl"
    if encoded_path.exists():
        encoded = {
            item["trial_id"]: item
            for item in (
                json.loads(line) for line in encoded_path.read_text(encoding="utf-8").splitlines()
            )
        }
        history = [
            item.model_copy(
                update={
                    "trace": item.trace.model_copy(
                        update={
                            "effective_parameter_delta": encoded[item.trial_id][
                                "effective_parameter_delta"
                            ]
                        }
                    )
                }
            )
            if item.trial_id in encoded
            else item
            for item in history
        ]
    base = next((item for item in history if item.trial_id == summary["base_trial_id"]), None)
    selections = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "selections").glob("*.json"))
    ]
    reviews = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "llm_reviews").glob("*.json"))
    ]
    snapshot = json.loads((run_dir / "experiment_snapshot.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    markdown = build_report(
        history,
        base,
        selections,
        reviews,
        run_mode=RunMode(summary["run_mode"]),
        metadata={"summary": summary, "experiment": snapshot, "environment": environment},
    )
    path = run_dir / "report.md"
    _write_atomic(path, markdown)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "trial_id",
            "run_mode",
            "status",
            "throughput_tps",
            "action_version",
            "configured_request_rate_rps",
            "issued_request_rate_rps",
            "completed_request_rate_rps",
            "queue_backlog_p95",
            "backlog_detected",
            *METRIC_IDS,
            *[f"p{number:02}" for number in range(1, 16)],
            "config_hash",
        ]
    )
    for item in history:
        writer.writerow(
            [
                item.trial_id,
                item.run_mode.value,
                item.status.value,
                item.throughput_tps,
                item.trace.action_version,
                item.configured_request_rate_rps,
                item.issued_request_rate_rps,
                item.completed_request_rate_rps,
                item.queue_backlog_p95,
                item.backlog_detected,
                *[item.metrics[metric] for metric in METRIC_IDS],
                *[item.trace.final_config[f"p{number:02}"] for number in range(1, 16)],
                item.trace.config_hash,
            ]
        )
    _write_atomic(run_dir / "trials.csv", output.getvalue())
    return path
