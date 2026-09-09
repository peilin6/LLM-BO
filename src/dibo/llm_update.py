"""Evidence construction and guarded LLM Action updates."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from dibo.observability import emit_event, stop_message, stop_requested
from dibo.schemas import (
    ACTION_IDS,
    METRIC_IDS,
    PARAMETER_IDS,
    ActionBundle,
    ActionUpdate,
    Graph,
    MetricSelection,
    StrictModel,
    TrialResult,
    TrialStatus,
    WeightUpdate,
)

_WEIGHT_TOLERANCE = 1e-9
_MAX_WEIGHT_DELTA = 0.10
_MIN_ANCHOR_MAGNITUDE = 0.80
_MIN_NON_ANCHOR_MAGNITUDE = 0.02
_SINGLE_GPU_A08_FORBIDDEN = frozenset({"p01", "p02", "p14"})


class UpdateValidationError(ValueError):
    """Raised when an LLM proposal violates the Day 13 contract."""


class ReviewAbortedError(RuntimeError):
    """Raised when the operator creates STOP while the controller waits for LLM."""


class ReviewRequest(StrictModel):
    request_id: str
    schema_version: Literal[1] = 1
    parent_action_version: int = Field(ge=1)
    payload: dict[str, Any]
    status: str = "pending"

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
            raise ValueError("request_id must be a non-empty file-safe name")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value != "pending":
            raise ValueError("request status must be pending")
        return value


class ReviewResponse(StrictModel):
    request_id: str
    schema_version: Literal[1] = 1
    parent_action_version: int = Field(ge=1)
    payload: dict[str, Any]
    status: str

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
            raise ValueError("request_id must be a non-empty file-safe name")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in {"ok", "error"}:
            raise ValueError("response status must be ok or error")
        return value


class FileRoundtripReviewer:
    """Publish one request and wait for its matching shared-directory response."""

    def __init__(
        self,
        io_dir: Path,
        *,
        timeout_s: float,
        poll_interval_s: float = 2.0,
        request_id_factory=None,
        heartbeat_s: float = 30.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        if timeout_s <= 0 or poll_interval_s <= 0 or heartbeat_s <= 0:
            raise ValueError("review timeout and polling interval must be positive")
        self.io_dir = io_dir
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.request_id_factory = request_id_factory or (
            lambda evidence: f"round_{evidence.bo_round:03}_{uuid.uuid4().hex}"
        )
        self.heartbeat_s = heartbeat_s
        self.sleep = sleep
        self.clock = clock

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        evidence = EvidenceBundle.model_validate(payload)
        request_id = self.request_id_factory(evidence)
        write_review_request(self.io_dir, request_id, evidence)
        response_path = self.io_dir / "responses" / f"{request_id}.json"
        deadline = self.clock() + self.timeout_s
        next_heartbeat = self.clock()
        emit_event(
            self.io_dir.parent,
            "llm_request_published",
            request_id=request_id,
            bo_round=evidence.bo_round,
            parent_action_version=evidence.parent_action_version,
            timeout_s=self.timeout_s,
        )
        while self.clock() < deadline:
            if stop_requested(self.io_dir.parent):
                message = stop_message(self.io_dir.parent)
                emit_event(
                    self.io_dir.parent,
                    "llm_wait_aborted",
                    request_id=request_id,
                    message=message,
                )
                raise ReviewAbortedError("operator requested stop while waiting for LLM response")
            if response_path.exists():
                update = read_review_response(
                    response_path,
                    evidence,
                    request_id=request_id,
                )
                emit_event(
                    self.io_dir.parent,
                    "llm_response_received",
                    request_id=request_id,
                    decision=update.decision,
                )
                return update.model_dump(mode="json")
            if self.clock() >= next_heartbeat:
                emit_event(
                    self.io_dir.parent,
                    "llm_waiting",
                    request_id=request_id,
                    elapsed_s=round(self.timeout_s - (deadline - self.clock()), 1),
                    remaining_s=round(max(0.0, deadline - self.clock()), 1),
                )
                next_heartbeat = self.clock() + self.heartbeat_s
            self.sleep(min(self.poll_interval_s, max(0.0, deadline - self.clock())))
        emit_event(
            self.io_dir.parent,
            "llm_wait_timeout",
            request_id=request_id,
            timeout_s=self.timeout_s,
        )
        raise TimeoutError(f"LLM response timed out for {request_id}")


class EvidenceTrial(StrictModel):
    trial_id: str
    current_round: bool
    action_version: int = Field(ge=1)
    coefficients: dict[str, float]
    final_config: dict[str, Any]
    effective_parameter_delta: dict[str, Any]
    compile_trace: dict[str, Any]
    metrics: dict[str, float | None]
    throughput_tps: float | None
    request_success_rate: float
    status: TrialStatus
    failure_reason: str | None


class ActionTrialEvidence(StrictModel):
    trial_id: str
    current_round: bool
    action_version: int = Field(ge=1)
    coefficient: float
    effective_parameter_delta: dict[str, Any]
    adjacent_metric_values: dict[str, float | None]
    throughput_tps: float | None
    status: TrialStatus
    failure_reason: str | None


class ActionEvidence(StrictModel):
    action_id: str
    adjacent_metrics: tuple[str, ...]
    trial_ids: tuple[str, ...]
    trials: tuple[ActionTrialEvidence, ...]


class EvidenceBundle(StrictModel):
    schema_version: int = 1
    parent_action_version: int = Field(ge=1)
    bo_round: int = Field(ge=1)
    action_bundle: ActionBundle
    selected_metrics: tuple[str, ...]
    selected_actions: tuple[str, ...]
    reference_trial_id: str
    targets: dict[str, dict[str, Any]]
    weights: dict[str, float]
    trials: tuple[EvidenceTrial, ...]
    action_evidence: dict[str, ActionEvidence]
    relative_outcomes: dict[str, dict[str, float | None]]
    model_evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence(self) -> EvidenceBundle:
        if self.parent_action_version != self.action_bundle.action_version:
            raise ValueError("evidence parent_action_version must match its ActionBundle")
        if tuple(self.action_evidence) != ACTION_IDS:
            raise ValueError("action_evidence must contain A01 through A08 in order")
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("evidence trial IDs must be unique")
        return self


def _adjacent_metrics(graph: Graph, action_id: str) -> tuple[str, ...]:
    action_index = graph.action_order.index(action_id)
    return tuple(
        metric_id
        for metric_id, connected in zip(graph.metric_order, graph.adjacency[action_index])
        if connected
    )


def _failure_reason(result: TrialResult) -> str | None:
    if result.status == TrialStatus.SUCCESS:
        return None
    missing = "; ".join(
        f"{metric_id}: {reason}"
        for metric_id, reason in sorted(result.metric_missing_reasons.items())
    )
    return f"{result.status.value}: {missing}" if missing else result.status.value


def build_evidence(
    action_bundle: ActionBundle,
    graph: Graph,
    selection: MetricSelection,
    history: Sequence[TrialResult],
    round_results: Sequence[TrialResult],
    *,
    bo_round: int,
    model_evidence: Mapping[str, Any] | None = None,
) -> EvidenceBundle:
    """Build a compact, ordered view of all history compatible with the current bundle."""
    if bo_round < 1:
        raise ValueError("bo_round must be positive")
    if not round_results:
        raise ValueError("an empty round must not trigger LLM review")
    history_by_id = {result.trial_id: result for result in history}
    if len(history_by_id) != len(history):
        raise ValueError("history contains duplicate trial IDs")
    round_ids = {result.trial_id for result in round_results}
    if not round_ids.issubset(history_by_id):
        raise ValueError("round_results must be present in history")

    compatible = [
        result
        for result in history
        if 1 <= result.trace.action_version <= action_bundle.action_version
    ]
    original_index = {result.trial_id: index for index, result in enumerate(history)}
    related_actions = {
        action_id
        for metric_id in selection.selected_metrics
        for action_id in graph.neighbors(metric_id)
    }
    compatible.sort(
        key=lambda result: (
            abs(action_bundle.action_version - result.trace.action_version),
            not bool(set(result.trace.selected_metrics) & set(selection.selected_metrics)),
            not any(result.trace.coefficients[action_id] != 0.0 for action_id in related_actions),
            -max(
                (abs(result.trace.coefficients[action_id]) for action_id in related_actions),
                default=0.0,
            ),
            -original_index[result.trial_id],
        )
    )

    trials = tuple(
        EvidenceTrial(
            trial_id=result.trial_id,
            current_round=result.trial_id in round_ids,
            action_version=result.trace.action_version,
            coefficients=result.trace.coefficients,
            final_config=result.trace.final_config,
            effective_parameter_delta=result.trace.effective_parameter_delta,
            compile_trace=result.trace.model_dump(mode="json"),
            metrics=result.metrics,
            throughput_tps=result.throughput_tps,
            request_success_rate=(
                result.successful_requests / result.request_count if result.request_count else 0.0
            ),
            status=result.status,
            failure_reason=_failure_reason(result),
        )
        for result in compatible
    )
    action_evidence = {
        action_id: ActionEvidence(
            action_id=action_id,
            adjacent_metrics=_adjacent_metrics(graph, action_id),
            trial_ids=tuple(
                trial.trial_id
                for trial in trials
                if trial.coefficients[action_id] != 0.0
            ),
            trials=tuple(
                ActionTrialEvidence(
                    trial_id=trial.trial_id,
                    current_round=trial.current_round,
                    action_version=trial.action_version,
                    coefficient=trial.coefficients[action_id],
                    effective_parameter_delta=trial.effective_parameter_delta,
                    adjacent_metric_values={
                        metric_id: trial.metrics[metric_id]
                        for metric_id in _adjacent_metrics(graph, action_id)
                    },
                    throughput_tps=trial.throughput_tps,
                    status=trial.status,
                    failure_reason=trial.failure_reason,
                )
                for trial in trials
                if trial.coefficients[action_id] != 0.0
            ),
        )
        for action_id in ACTION_IDS
    }
    reference = next(
        (trial for trial in trials if trial.trial_id == selection.reference_trial_id),
        None,
    )
    relative_outcomes: dict[str, dict[str, float | None]] = {}
    for outcome_id in (*METRIC_IDS, "throughput_tps"):
        reference_value = (
            reference.throughput_tps
            if reference is not None and outcome_id == "throughput_tps"
            else reference.metrics.get(outcome_id) if reference is not None else None
        )
        observed = [
            trial.throughput_tps if outcome_id == "throughput_tps" else trial.metrics[outcome_id]
            for trial in trials
            if trial.status == TrialStatus.SUCCESS
        ]
        deltas = [
            float(value - reference_value)
            for value in observed
            if value is not None and reference_value is not None
        ]
        relative_outcomes[outcome_id] = {
            "reference": reference_value,
            "minimum_delta": min(deltas) if deltas else None,
            "maximum_delta": max(deltas) if deltas else None,
        }
    return EvidenceBundle(
        parent_action_version=action_bundle.action_version,
        bo_round=bo_round,
        action_bundle=action_bundle,
        selected_metrics=selection.selected_metrics,
        selected_actions=tuple(
            action_id
            for action_id in ACTION_IDS
            if any(action_id in graph.neighbors(metric_id) for metric_id in selection.selected_metrics)
        ),
        reference_trial_id=selection.reference_trial_id,
        targets={
            metric_id: target.model_dump(mode="json")
            for metric_id, target in selection.targets.items()
        },
        weights=selection.weights,
        trials=trials,
        action_evidence=action_evidence,
        relative_outcomes=relative_outcomes,
        model_evidence=dict(model_evidence or {}),
    )


def keep_update(evidence: EvidenceBundle) -> ActionUpdate:
    """Return the canonical safe fallback for a failed or unusable review."""
    return ActionUpdate(
        decision="keep",
        parent_action_version=evidence.parent_action_version,
        effective_from_bo_round=evidence.bo_round + 1,
        updates=(),
    )


def review_round(
    evidence: EvidenceBundle,
    reviewer: Callable[[dict[str, Any]], Mapping[str, Any] | str],
) -> ActionUpdate:
    """Call a reviewer exactly once and degrade malformed or failed output to keep."""
    try:
        raw = reviewer(evidence.model_dump(mode="json"))
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        update = ActionUpdate.model_validate(payload)
        if update.parent_action_version != evidence.parent_action_version:
            raise ValueError("review parent_action_version mismatch")
        if update.effective_from_bo_round != evidence.bo_round + 1:
            raise ValueError("review effective round mismatch")
        return update
    except (
        TimeoutError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
        httpx.HTTPError,
    ):
        return keep_update(evidence)


def call_llm_api(
    payload: Mapping[str, Any],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_s: float,
    post: Callable[..., httpx.Response] = httpx.post,
) -> Mapping[str, Any] | str:
    """Make one bounded OpenAI-compatible request and return its structured content."""
    if not endpoint or not model or not api_key:
        raise ValueError("endpoint, model, and API key are required")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    response = post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only one JSON ActionUpdate matching the supplied evidence.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"]


def write_review_request(io_dir: Path, request_id: str, evidence: EvidenceBundle) -> Path:
    """Atomically publish one review request under the shared directory protocol."""
    from dibo.store import save_json_atomic

    request = ReviewRequest(
        request_id=request_id,
        parent_action_version=evidence.parent_action_version,
        payload=evidence.model_dump(mode="json"),
    )
    path = io_dir / "requests" / f"{request_id}.json"
    save_json_atomic(path, request.model_dump(mode="json"))
    return path


def process_review_request(
    request_path: Path,
    responses_dir: Path,
    reviewer: Callable[[dict[str, Any]], Mapping[str, Any] | str],
) -> Path:
    """Process one request once and atomically publish its response."""
    from dibo.store import save_json_atomic

    with request_path.open("r", encoding="utf-8") as file:
        request = ReviewRequest.model_validate(json.load(file))
    if request_path.stem != request.request_id:
        raise ValueError("request filename and request_id differ")
    if (responses_dir / f"{request.request_id}.json").exists():
        raise FileExistsError("request already has a response; refusing replay")
    if request.payload.get("parent_action_version") != request.parent_action_version:
        raise ValueError("request and evidence parent versions differ")
    emit_event(
        responses_dir.parent,
        "llm_worker_request_started",
        request_id=request.request_id,
        parent_action_version=request.parent_action_version,
    )
    status = "ok"
    error_summary: str | None = None
    try:
        raw = reviewer(request.payload)
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        update = ActionUpdate.model_validate(payload)
        if update.parent_action_version != request.parent_action_version:
            raise ValueError("review parent_action_version mismatch")
    except (
        TimeoutError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
        httpx.HTTPError,
    ) as error:
        status = "error"
        payload = {}
        error_summary = f"{type(error).__name__}: {error}"[:500]
    response = ReviewResponse(
        request_id=request.request_id,
        parent_action_version=request.parent_action_version,
        payload=payload,
        status=status,
    )
    path = responses_dir / f"{request.request_id}.json"
    save_json_atomic(path, response.model_dump(mode="json"))
    emit_event(
        responses_dir.parent,
        "llm_worker_response_written",
        request_id=request.request_id,
        status=status,
        error=error_summary,
    )
    return path


def read_review_response(
    response_path: Path,
    evidence: EvidenceBundle,
    *,
    request_id: str,
) -> ActionUpdate:
    """Read one matching response; malformed, stale, or error responses become keep."""
    try:
        with response_path.open("r", encoding="utf-8") as file:
            response = ReviewResponse.model_validate(json.load(file))
        if (
            response.status != "ok"
            or response.request_id != request_id
            or response.parent_action_version != evidence.parent_action_version
        ):
            raise ValueError("response identity or parent version mismatch")
        return review_round(evidence, lambda _: response.payload)
    except (OSError, ValueError, json.JSONDecodeError, ValidationError):
        return keep_update(evidence)


def _weight(bundle: ActionBundle, action_id: str, parameter_id: str) -> tuple[float, bool]:
    action = bundle.actions[ACTION_IDS.index(action_id)]
    return (
        action.direction_vector[PARAMETER_IDS.index(parameter_id)],
        parameter_id in action.anchor_parameter_ids,
    )


def _validate_operation(item: WeightUpdate) -> None:
    old = item.old_weight
    new = item.new_weight
    if not isfinite(old) or not isfinite(new) or not -1 <= new <= 1:
        raise UpdateValidationError("weights must be finite and within [-1, 1]")
    delta = abs(new - old)
    if delta > _MAX_WEIGHT_DELTA + _WEIGHT_TOLERANCE:
        raise UpdateValidationError("weight delta exceeds 0.10")
    same_sign = old != 0 and new != 0 and (old > 0) == (new > 0)
    operation_valid = {
        "strengthen": same_sign and abs(new) > abs(old),
        "weaken": same_sign and 0 < abs(new) < abs(old),
        "reverse": old != 0 and new != 0 and (old > 0) != (new > 0),
        "set_zero": old != 0 and new == 0,
        "keep": isclose(old, new, abs_tol=_WEIGHT_TOLERANCE),
    }[item.weight_operation]
    if not operation_valid:
        raise UpdateValidationError(f"{item.weight_operation} does not match old/new weights")


def _validate_observation_consistency(
    successful: Sequence[EvidenceTrial],
    action_id: str,
    target_metrics: Sequence[str],
) -> None:
    for metric_id in target_metrics:
        directions: set[int] = set()
        for left_index, left in enumerate(successful):
            for right in successful[left_index + 1 :]:
                coefficient_delta = right.coefficients[action_id] - left.coefficients[action_id]
                left_metric = left.metrics[metric_id]
                right_metric = right.metrics[metric_id]
                if coefficient_delta == 0 or left_metric is None or right_metric is None:
                    continue
                metric_delta = right_metric - left_metric
                if metric_delta != 0:
                    directions.add(1 if coefficient_delta * metric_delta > 0 else -1)
        if len(directions) > 1:
            raise UpdateValidationError("conflicting observations require keep")


def _validate_evidence_for_item(
    item: WeightUpdate,
    evidence: EvidenceBundle,
    graph: Graph,
) -> None:
    trial_by_id = {trial.trial_id: trial for trial in evidence.trials}
    if not item.evidence_trial_ids:
        raise UpdateValidationError("each update must cite evidence trials")
    if any(trial_id not in trial_by_id for trial_id in item.evidence_trial_ids):
        raise UpdateValidationError("update cites an unknown Trial")
    if len(set(item.evidence_trial_ids)) != len(item.evidence_trial_ids):
        raise UpdateValidationError("evidence_trial_ids contains duplicates")
    cited = [trial_by_id[trial_id] for trial_id in item.evidence_trial_ids]
    successful = [
        trial
        for trial in cited
        if trial.status == TrialStatus.SUCCESS and trial.coefficients[item.action_id] != 0.0
        and trial.request_success_rate >= 1.0
        and trial.throughput_tps is not None and isfinite(trial.throughput_tps)
    ]
    if len(successful) < 2:
        raise UpdateValidationError("an Action update requires at least two successful experiments")
    coefficient_values = {trial.coefficients[item.action_id] for trial in successful}
    parameter_deltas = {
        repr(trial.effective_parameter_delta.get(item.parameter_id)) for trial in successful
    }
    if len(coefficient_values) == 1 and len(parameter_deltas) == 1:
        raise UpdateValidationError("successful evidence must contain distinct coefficients or effects")

    adjacent = set(_adjacent_metrics(graph, item.action_id))
    if not item.target_metrics or any(metric_id not in adjacent for metric_id in item.target_metrics):
        raise UpdateValidationError("target Metrics must be adjacent to the updated Action")
    if any(metric_id not in METRIC_IDS for metric_id in item.target_metrics):
        raise UpdateValidationError("update contains an unknown Metric")
    if any(trial.metrics[metric_id] is None or not isfinite(trial.metrics[metric_id])
           for trial in successful for metric_id in item.target_metrics):
        raise UpdateValidationError("successful evidence requires observed finite target Metrics")
    if any(trial.status != TrialStatus.SUCCESS for trial in cited) and item.weight_operation not in {
        "weaken",
        "set_zero",
    }:
        raise UpdateValidationError("failed Trials may only support weaken or set_zero")
    _validate_observation_consistency(successful, item.action_id, item.target_metrics)

    effects = {
        1 if trial.coefficients[item.action_id] * (item.new_weight - item.old_weight) > 0 else -1
        for trial in successful
        if trial.coefficients[item.action_id] * (item.new_weight - item.old_weight) != 0
    }
    expected = {"increase": 1, "decrease": -1, "neutral": 0}[item.desired_parameter_effect]
    if expected == 0:
        if item.weight_operation not in {"weaken", "set_zero"}:
            raise UpdateValidationError("neutral requires weaken or set_zero")
    elif effects != {expected}:
        raise UpdateValidationError("desired_parameter_effect conflicts with coefficient and weight signs")


def validate_update(
    bundle: ActionBundle,
    update: ActionUpdate,
    evidence: EvidenceBundle,
    graph: Graph,
    *,
    gpu_count: int = 1,
) -> None:
    """Validate an LLM response as one indivisible proposal."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if update.parent_action_version != bundle.action_version:
        raise UpdateValidationError("parent_action_version does not match the current bundle")
    if evidence.parent_action_version != bundle.action_version:
        raise UpdateValidationError("evidence was built for a different Action version")
    if update.effective_from_bo_round != evidence.bo_round + 1:
        raise UpdateValidationError("an update must take effect in the next BO round")
    if update.decision == "keep":
        if update.updates:
            raise UpdateValidationError("keep must not contain updates")
        return
    if not update.updates:
        raise UpdateValidationError("modify requires at least one update")
    if len(update.updates) > 5:
        raise UpdateValidationError("a review may change at most five cells")

    cells = [(item.action_id, item.parameter_id) for item in update.updates]
    if len(set(cells)) != len(cells):
        raise UpdateValidationError("a review cannot update the same cell twice")
    counts = Counter(item.action_id for item in update.updates)
    if any(count > 2 for count in counts.values()):
        raise UpdateValidationError("a review may change at most two parameters per Action")

    for item in update.updates:
        if item.action_id not in ACTION_IDS or item.parameter_id not in PARAMETER_IDS:
            raise UpdateValidationError("update contains an unknown Action or parameter")
        current_weight, anchor = _weight(bundle, item.action_id, item.parameter_id)
        if not isclose(item.old_weight, current_weight, abs_tol=_WEIGHT_TOLERANCE):
            raise UpdateValidationError("old_weight does not match the current bundle")
        _validate_operation(item)
        if item.weight_operation == "keep":
            raise UpdateValidationError("modify must not contain keep cells")
        if anchor:
            if (
                abs(item.new_weight) < _MIN_ANCHOR_MAGNITUDE - _WEIGHT_TOLERANCE
                or current_weight * item.new_weight <= 0
            ):
                raise UpdateValidationError("anchor magnitude/sign constraint violated")
            if item.weight_operation == "set_zero":
                raise UpdateValidationError("anchor weights cannot be set to zero")
        elif item.new_weight != 0 and abs(item.new_weight) < _MIN_NON_ANCHOR_MAGNITUDE:
            raise UpdateValidationError("non-anchor magnitude must be zero or at least 0.02")
        if gpu_count == 1 and item.action_id == "A08" and item.parameter_id in _SINGLE_GPU_A08_FORBIDDEN:
            raise UpdateValidationError("single-GPU evidence cannot update this A08 parameter")
        _validate_evidence_for_item(item, evidence, graph)


def apply_update(
    bundle: ActionBundle,
    update: ActionUpdate,
    evidence: EvidenceBundle,
    graph: Graph,
    *,
    gpu_count: int = 1,
) -> ActionBundle:
    """Return a new ActionBundle after validation, leaving the parent immutable."""
    validate_update(bundle, update, evidence, graph, gpu_count=gpu_count)
    if update.decision == "keep":
        return bundle
    replacements = {
        (item.action_id, item.parameter_id): item.new_weight for item in update.updates
    }
    actions = []
    for action in bundle.actions:
        direction_vector = tuple(
            replacements.get((action.action_id, parameter_id), old_weight)
            for parameter_id, old_weight in zip(PARAMETER_IDS, action.direction_vector)
        )
        actions.append(action.model_copy(update={"direction_vector": direction_vector}))
    return bundle.model_copy(
        update={
            "action_version": bundle.action_version + 1,
            "parent_action_version": bundle.action_version,
            "source": "llm_review",
            "actions": tuple(actions),
        }
    )


def save_action_bundle(bundle: ActionBundle, directory: Path) -> Path:
    """Atomically save one immutable actions_vN.yaml without overwriting an old version."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"actions_v{bundle.action_version}.yaml"
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                bundle.model_dump(mode="json"),
                file,
                allow_unicode=False,
                sort_keys=False,
            )
            file.flush()
            os.fsync(file.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path