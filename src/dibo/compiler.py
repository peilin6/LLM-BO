"""Deterministic compilation from Action coefficients to vLLM parameters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from math import isfinite
from typing import Any

from dibo.schemas import (
    ACTION_IDS,
    PARAMETER_IDS,
    ActionBundle,
    CompileTrace,
    ParameterCatalog,
    ParameterKind,
)


@dataclass(frozen=True)
class CompileEnvironment:
    """Pure constraints and context used while compiling a candidate."""

    parameter_catalog: ParameterCatalog
    allocated_gpu_count: int
    num_attention_heads: int
    num_hidden_layers: int
    max_model_len: int
    supported_block_sizes: tuple[int, ...]
    supported_parameter_ids: tuple[str, ...] = PARAMETER_IDS
    model_context: str = ""
    workload_context: str = ""
    execution_mode: str = ""
    engine_version: str = ""


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _aligned_half_up(value: int, alignment: int, low: int, high: int) -> int:
    aligned = _round_half_up(Decimal(value) / Decimal(alignment)) * alignment
    minimum = ((low + alignment - 1) // alignment) * alignment
    maximum = (high // alignment) * alignment
    return min(max(aligned, minimum), maximum)


def _canonical_hash(final_config: Mapping[str, Any], environment: CompileEnvironment) -> str:
    payload = {
        "final_config": {parameter_id: final_config[parameter_id] for parameter_id in PARAMETER_IDS},
        "model_context": environment.model_context,
        "workload_context": environment.workload_context,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _legal_ordered_choices(
    parameter_id: str,
    choices: tuple[int, ...],
    environment: CompileEnvironment,
) -> tuple[int, ...]:
    if parameter_id in {"p01", "p02"}:
        return tuple(choice for choice in choices if choice <= environment.allocated_gpu_count)
    if parameter_id == "p05":
        return tuple(choice for choice in choices if choice in environment.supported_block_sizes)
    return choices


def _validate_constraints(
    final_config: Mapping[str, Any], environment: CompileEnvironment
) -> list[str]:
    errors: list[str] = []
    tensor_parallel = int(final_config["p01"])
    pipeline_parallel = int(final_config["p02"])
    if tensor_parallel * pipeline_parallel > environment.allocated_gpu_count:
        errors.append("p01 * p02 exceeds allocated_gpu_count")
    if environment.num_attention_heads % tensor_parallel != 0:
        errors.append("num_attention_heads is not divisible by p01")
    if environment.num_hidden_layers % pipeline_parallel != 0:
        errors.append("num_hidden_layers is not divisible by p02")
    if int(final_config["p04"]) < int(final_config["p03"]):
        errors.append("p04 must be greater than or equal to p03")
    if not 1 <= int(final_config["p10"]) <= int(final_config["p09"]):
        errors.append("p10 must be between 1 and p09")
    if not 256 <= int(final_config["p11"]) <= environment.max_model_len:
        errors.append("p11 must be between 256 and max_model_len")
    if int(final_config["p04"]) % 256 or int(final_config["p11"]) % 256:
        errors.append("p04 and p11 must be aligned to 256")
    if int(final_config["p05"]) not in environment.supported_block_sizes:
        errors.append("p05 is not supported by the current backend")
    if environment.engine_version == "0.11.2":
        if int(final_config["p09"]) != 1:
            errors.append(
                "p09 must equal 1 in vLLM 0.11.2 because concurrent partial prefill is unsupported"
            )
        if int(final_config["p10"]) != 1:
            errors.append(
                "p10 must equal 1 in vLLM 0.11.2 because concurrent partial prefill is unsupported"
            )
    return errors


def compile_config(
    base_config: Mapping[str, Any],
    action_bundle: ActionBundle,
    selected_actions: Sequence[str],
    coefficients: Mapping[str, float],
    environment: CompileEnvironment,
    *,
    trial_id: str = "candidate",
    phase: str = "bo",
    bo_round: int | None = 1,
    compile_base_id: str = "x_base",
    base_trial_id: str | None = None,
    selected_metrics: Sequence[str] = (),
) -> CompileTrace:
    """Compile one candidate from a fixed base without mutating prior candidates."""
    if tuple(base_config) != PARAMETER_IDS:
        raise ValueError("base_config must contain p01 through p15 in order")
    if len(set(selected_actions)) != len(selected_actions):
        raise ValueError("selected_actions contains duplicates")
    if any(action_id not in ACTION_IDS for action_id in selected_actions):
        raise ValueError("selected_actions contains an unknown action")
    if any(action_id not in ACTION_IDS for action_id in coefficients):
        raise ValueError("coefficients contains an unknown action")

    requested = dict(coefficients)
    full_coefficients = {action_id: 0.0 for action_id in ACTION_IDS}
    constraint_errors: list[str] = []
    selected = set(selected_actions)
    for action_id, coefficient in requested.items():
        value = float(coefficient)
        if not isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"coefficient for {action_id} must be finite in [-1, 1]")
        if action_id not in selected:
            if value != 0.0:
                constraint_errors.append(f"unselected action {action_id} requested non-zero coefficient")
            continue
        full_coefficients[action_id] = value

    actions = {action.action_id: action for action in action_bundle.actions}
    specifications = {spec.parameter_id: spec for spec in environment.parameter_catalog.parameters}
    final_config: dict[str, Any] = {}
    numeric_decisions: dict[str, dict[str, Any]] = {}
    categorical_decisions: dict[str, dict[str, Any]] = {}

    for parameter_index, parameter_id in enumerate(PARAMETER_IDS):
        specification = specifications[parameter_id]
        base_value = base_config[parameter_id]
        if parameter_id not in environment.supported_parameter_ids:
            constraint_errors.append(f"unsupported parameter: {parameter_id}")

        if specification.kind in {ParameterKind.INT, ParameterKind.FLOAT}:
            contributions = {
                action_id: float(
                    _decimal(specification.action_scale)
                    * _decimal(full_coefficients[action_id])
                    * _decimal(actions[action_id].direction_vector[parameter_index])
                )
                for action_id in ACTION_IDS
            }
            raw = _decimal(base_value) + sum(
                (_decimal(value) for value in contributions.values()),
                start=Decimal(0),
            )
            clipped = min(max(raw, _decimal(specification.low)), _decimal(specification.high))
            rounded: int | float
            if specification.kind == ParameterKind.INT:
                rounded = _round_half_up(clipped)
                if specification.alignment and specification.alignment > 1:
                    rounded = _aligned_half_up(
                        rounded,
                        specification.alignment,
                        int(specification.low),
                        int(specification.high),
                    )
                final_value: int | float = int(rounded)
            else:
                rounded = float(clipped)
                final_value = rounded
            final_config[parameter_id] = final_value
            numeric_decisions[parameter_id] = {
                "base_value": base_value,
                "contributions": contributions,
                "raw_value": float(raw),
                "clipped_value": float(clipped),
                "rounded_value": rounded,
                "final_value": final_value,
            }
        else:
            base_vote = Decimal("0.35") if specification.kind == ParameterKind.BOOL and base_value else Decimal("-0.35")
            if specification.kind == ParameterKind.ORDERED_CHOICE:
                base_vote = Decimal(0)
            decimal_votes = {
                action_id: _decimal(full_coefficients[action_id])
                * _decimal(actions[action_id].direction_vector[parameter_index])
                for action_id in ACTION_IDS
            }
            total_vote = base_vote + sum(decimal_votes.values(), start=Decimal(0))
            at_boundary = False
            legal_choices: tuple[int, ...] = ()
            if specification.kind == ParameterKind.BOOL:
                if total_vote > Decimal("0.35"):
                    final_value = True
                elif total_vote < Decimal("-0.35"):
                    final_value = False
                else:
                    final_value = bool(base_value)
            else:
                legal_choices = _legal_ordered_choices(
                    parameter_id,
                    specification.choices,
                    environment,
                )
                if base_value not in legal_choices:
                    constraint_errors.append(
                        f"base value for {parameter_id} is not a legal ordered choice"
                    )
                    final_value = base_value
                else:
                    base_index = legal_choices.index(base_value)
                    target_index = base_index
                    if total_vote > Decimal("0.35"):
                        target_index = min(base_index + 1, len(legal_choices) - 1)
                    elif total_vote < Decimal("-0.35"):
                        target_index = max(base_index - 1, 0)
                    at_boundary = target_index == base_index and abs(total_vote) > Decimal("0.35")
                    final_value = legal_choices[target_index]
            final_config[parameter_id] = final_value
            categorical_decisions[parameter_id] = {
                "base_value": base_value,
                "base_vote": float(base_vote),
                "votes": {action_id: float(vote) for action_id, vote in decimal_votes.items()},
                "total_vote": float(total_vote),
                "threshold": 0.35,
                "legal_choices": legal_choices,
                "anchor": base_value,
                "at_boundary": at_boundary,
                "final_value": final_value,
            }

    constraint_errors.extend(_validate_constraints(final_config, environment))

    return CompileTrace(
        schema_version=8,
        trial_id=trial_id,
        phase=phase,
        bo_round=bo_round if phase == "bo" else None,
        action_version=action_bundle.action_version,
        compile_base_id=compile_base_id,
        base_trial_id=base_trial_id,
        selected_metrics=tuple(selected_metrics),
        selected_actions=tuple(selected_actions),
        requested_coefficients=requested,
        coefficients=full_coefficients,
        numeric_decisions=numeric_decisions,
        categorical_decisions=categorical_decisions,
        final_config=final_config,
        effective_parameter_delta={},
        constraint_errors=tuple(constraint_errors),
        config_hash=_canonical_hash(final_config, environment),
    )
