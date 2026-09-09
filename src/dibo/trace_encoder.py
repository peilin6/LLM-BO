"""Stable encoding of executed configurations for trace and LLM evidence."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from dibo.schemas import CompileTrace, EncodedConfig, ParameterCatalog, ParameterKind


def encode_trace(
    trace: CompileTrace,
    base_config: Mapping[str, Any],
    parameter_specs: ParameterCatalog,
    *,
    base_trial_id: str,
) -> EncodedConfig:
    """Encode the executed configuration relative to the fixed x_base."""
    if not trace.valid:
        raise ValueError("invalid compile traces cannot be encoded as reproducible evidence")
    if tuple(base_config) != parameter_specs.parameter_order:
        raise ValueError("base_config must use the canonical parameter order")
    if tuple(trace.final_config) != parameter_specs.parameter_order:
        raise ValueError("trace final_config must contain all parameters")

    features: list[float] = []
    deltas: dict[str, float | int] = {}
    for specification in parameter_specs.parameters:
        parameter_id = specification.parameter_id
        final_value = trace.final_config[parameter_id]
        base_value = base_config[parameter_id]
        if specification.kind in {ParameterKind.INT, ParameterKind.FLOAT}:
            delta = float(final_value) - float(base_value)
            feature = delta / float(specification.action_scale)
        elif specification.kind == ParameterKind.ORDERED_CHOICE:
            try:
                delta = specification.choices.index(final_value) - specification.choices.index(base_value)
            except ValueError as error:
                raise ValueError(f"{parameter_id} is outside its fixed ordered choices") from error
            feature = float(delta)
        else:
            delta = int(bool(final_value)) - int(bool(base_value))
            feature = float(delta)
        if not isfinite(feature):
            raise ValueError(f"non-finite encoded value for {parameter_id}")
        features.append(feature)
        deltas[parameter_id] = delta

    return EncodedConfig(
        trial_id=trace.trial_id,
        compile_base_id=trace.compile_base_id,
        base_trial_id=base_trial_id,
        action_version=trace.action_version,
        parameter_order=parameter_specs.parameter_order,
        features=tuple(features),
        effective_parameter_delta=deltas,
        final_config_hash=trace.config_hash,
    )
