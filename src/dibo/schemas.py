"""Strict configuration schemas for DIBO."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PARAMETER_IDS = tuple(f"p{number:02}" for number in range(1, 16))
ACTION_IDS = tuple(f"A{number:02}" for number in range(1, 9))
METRIC_IDS = tuple(f"m{number:02}" for number in range(1, 7))
ModelT = TypeVar("ModelT", bound="StrictModel")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParameterKind(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ORDERED_CHOICE = "ordered_choice"


class RunMode(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class EngineAdapter(str, Enum):
    VLLM_SUBPROCESS = "vllm_subprocess"
    FAKE = "fake"


class MetricsAdapter(str, Enum):
    PROMETHEUS_NVML = "prometheus_nvml"
    FAKE = "fake"


class LlmTransport(str, Enum):
    SIMPLE_FILE_ROUNDTRIP = "simple_file_roundtrip"
    FAKE = "fake"


class ParameterSpec(StrictModel):
    parameter_id: str
    name: str
    kind: ParameterKind
    initial: Any
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[int, ...] = ()
    unit: str | None = None
    action_scale: float | int | None = None
    alignment: int | None = None

    @model_validator(mode="after")
    def validate_domain(self) -> ParameterSpec:
        if self.kind in {ParameterKind.INT, ParameterKind.FLOAT}:
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError("numeric parameters need ordered low/high bounds")
            if self.action_scale is None or not isfinite(float(self.action_scale)):
                raise ValueError("numeric parameters need a finite action_scale")
        if self.kind == ParameterKind.ORDERED_CHOICE and not self.choices:
            raise ValueError("ordered_choice parameters need choices")
        if self.alignment is not None and self.alignment <= 0:
            raise ValueError("alignment must be positive")
        return self


class ParameterCatalog(StrictModel):
    parameter_order: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> ParameterCatalog:
        if self.parameter_order != PARAMETER_IDS:
            raise ValueError("parameter_order must be p01 through p15")
        if tuple(spec.parameter_id for spec in self.parameters) != self.parameter_order:
            raise ValueError("parameters must match parameter_order exactly")
        return self


class ActionVector(StrictModel):
    action_id: str
    name: str
    positive_semantics: str
    direction_vector: tuple[float, ...]
    anchor_parameter_ids: tuple[str, ...]

    @field_validator("direction_vector")
    @classmethod
    def validate_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != len(PARAMETER_IDS):
            raise ValueError("direction_vector must have 15 values")
        if any(not isfinite(item) or item < -1 or item > 1 for item in value):
            raise ValueError("direction_vector values must be finite in [-1, 1]")
        return value


class ActionBundle(StrictModel):
    schema_version: Literal[8]
    action_version: int = Field(ge=1)
    parent_action_version: int | None = None
    source: str
    action_order: tuple[str, ...]
    parameter_order: tuple[str, ...]
    metric_order: tuple[str, ...]
    actions: tuple[ActionVector, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> ActionBundle:
        if self.action_order != ACTION_IDS or self.parameter_order != PARAMETER_IDS:
            raise ValueError("ActionBundle IDs must use the canonical order")
        if self.metric_order != METRIC_IDS:
            raise ValueError("ActionBundle metric_order must be m01 through m06")
        if tuple(action.action_id for action in self.actions) != self.action_order:
            raise ValueError("actions must match action_order exactly")
        if self.action_version == 1 and self.parent_action_version is not None:
            raise ValueError("action version 1 has no parent")
        if self.action_version > 1 and self.parent_action_version != self.action_version - 1:
            raise ValueError("updated action versions require the immediate parent")
        return self


class Graph(StrictModel):
    action_order: tuple[str, ...]
    metric_order: tuple[str, ...]
    adjacency: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def validate_graph(self) -> Graph:
        if self.action_order != ACTION_IDS or self.metric_order != METRIC_IDS:
            raise ValueError("Graph IDs must use canonical order")
        if len(self.adjacency) != 8 or any(len(row) != 6 for row in self.adjacency):
            raise ValueError("adjacency must be 8 by 6")
        if any(value not in {0, 1} for row in self.adjacency for value in row):
            raise ValueError("adjacency values must be binary")
        if sum(sum(row) for row in self.adjacency) != 24:
            raise ValueError("the v8 graph must contain 24 edges")
        return self

    def neighbors(self, metric_id: str) -> tuple[str, ...]:
        metric_index = self.metric_order.index(metric_id)
        return tuple(action for action, row in zip(self.action_order, self.adjacency) if row[metric_index])


class EngineSpec(StrictModel):
    adapter: EngineAdapter
    version: str
    execution_mode: Literal["V0", "V1", "REPLACE_AFTER_SMOKE_WITH_V1_OR_V0"]
    model: str
    tokenizer: str
    model_revision: str
    dtype: str
    served_model_name: str
    host: str
    port: int = Field(ge=1, le=65535)
    max_model_len: int = Field(ge=256)
    allocated_gpu_count: int = Field(ge=1)
    allocated_gpu_uuids: tuple[str, ...]
    startup_timeout_s: int = Field(gt=0)
    benchmark_timeout_s: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_allocation(self) -> EngineSpec:
        if self.host != "127.0.0.1":
            raise ValueError("engine host must remain bound to 127.0.0.1")
        if len(self.allocated_gpu_uuids) != self.allocated_gpu_count:
            raise ValueError("allocated GPU UUID count must match allocated_gpu_count")
        if len(set(self.allocated_gpu_uuids)) != len(self.allocated_gpu_uuids):
            raise ValueError("allocated GPU UUIDs must be unique")
        return self


class WorkloadSpec(StrictModel):
    request_file: str
    request_sha256: str
    backend: str
    endpoint: str
    streaming: bool
    num_prompts: int = Field(gt=0)
    warmup_prompts: int = Field(ge=0)
    request_rate: float = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    burstiness: float = Field(gt=0)
    output_len: int = Field(gt=0)
    ignore_eos: bool
    chat_template_id: str
    prefix_profile: str
    required_success_rate: float = Field(ge=0, le=1)
    ttft_slo_s: float = Field(gt=0)


class MetricsSpec(StrictModel):
    adapter: MetricsAdapter
    sampling_interval_s: float = Field(gt=0)
    min_valid_samples: int = Field(gt=1)


class TuningSpec(StrictModel):
    initial_trials: int = Field(ge=5, le=10)
    initial_probe_coefficient: float = Field(gt=0, le=1)
    metric_top_k: int = Field(ge=1, le=3)
    metric_top_k_min: int = Field(ge=1, le=3)
    metric_top_k_max: int = Field(ge=1, le=3)
    third_metric_ratio: float = Field(gt=0, le=1)
    bo_rounds: int = Field(ge=0)
    trials_per_bo_round: int = Field(ge=1, le=5)
    max_total_trials: int = Field(gt=0)
    coefficient_low: float = Field(ge=-1, le=1)
    coefficient_high: float = Field(ge=-1, le=1)
    candidate_pool_size: int = Field(gt=0)
    max_candidate_pools: int = Field(gt=0)
    mc_samples: int = Field(gt=0)
    acquisition: Literal["metric_loss_ei"]
    acquisition_tie_tolerance: float = Field(ge=0)
    loss_tie_tolerance: float = Field(ge=0)
    g_counterfactual_beta: float = Field(ge=0)
    g_counterfactual_grid_size: int = Field(ge=2)
    time_limit_s: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_schedule(self) -> TuningSpec:
        if not self.metric_top_k_min <= self.metric_top_k <= self.metric_top_k_max:
            raise ValueError("metric top-k bounds are inconsistent")
        if self.coefficient_low >= self.coefficient_high:
            raise ValueError("coefficient bounds are inconsistent")
        if self.max_total_trials > self.initial_trials + self.bo_rounds * self.trials_per_bo_round:
            raise ValueError("max_total_trials must not exceed the configured schedule")
        return self


class ModelSpec(StrictModel):
    kernel: Literal["matern_5_2_ard"]
    min_train_samples: int = Field(ge=2)
    min_unique_configs: int = Field(ge=1)
    historical_action_version_noise_multiplier: float = Field(ge=1)
    cpu_threads: int = Field(ge=1)
    dtype: Literal["float64"]


class LlmSpec(StrictModel):
    transport: LlmTransport
    model: str
    api_key_env: str
    llm_update_interval_trials: int = Field(ge=1, le=5)
    timeout_s: int = Field(gt=0)
    max_weight_change: float = Field(gt=0, le=1)
    anchor_min_abs: float = Field(ge=0, le=1)
    non_anchor_min_abs: float = Field(gt=0, le=1)
    max_updated_cells: int = Field(ge=1)
    max_updated_parameters_per_action: int = Field(ge=1)


class Threshold(StrictModel):
    kind: Literal["upper", "lower", "interval"]
    trigger: float | str | None = None
    target: float | str | None = None
    trigger_from: str | None = None
    target_from: str | None = None
    scale: float | str


class ExperimentConfig(StrictModel):
    schema_version: Literal[8]
    experiment_id: str
    run_mode: RunMode
    seed: int
    engine: EngineSpec
    workload: WorkloadSpec
    metrics: MetricsSpec
    tuning: TuningSpec
    models: ModelSpec
    llm: LlmSpec
    thresholds: dict[str, Threshold]

    @model_validator(mode="after")
    def validate_intervals(self) -> ExperimentConfig:
        if self.llm.llm_update_interval_trials != self.tuning.trials_per_bo_round:
            raise ValueError("LLM updates must occur at BO round boundaries")
        if self.run_mode == RunMode.SYNTHETIC:
            if self.engine.adapter != EngineAdapter.FAKE:
                raise ValueError("synthetic mode requires the fake engine adapter")
            if self.metrics.adapter != MetricsAdapter.FAKE:
                raise ValueError("synthetic mode requires the fake metrics adapter")
            if self.llm.transport != LlmTransport.FAKE:
                raise ValueError("synthetic mode requires the fake LLM transport")
        else:
            if self.engine.adapter != EngineAdapter.VLLM_SUBPROCESS:
                raise ValueError("real mode requires the vllm_subprocess engine adapter")
            if self.metrics.adapter != MetricsAdapter.PROMETHEUS_NVML:
                raise ValueError("real mode requires the prometheus_nvml metrics adapter")
            if self.llm.transport != LlmTransport.SIMPLE_FILE_ROUNDTRIP:
                raise ValueError("real mode requires simple_file_roundtrip LLM transport")
        return self

    def validate_for_execution(self) -> None:
        if self.run_mode != RunMode.REAL:
            return
        values = (
            self.engine.execution_mode,
            self.engine.model_revision,
            self.workload.request_sha256,
            self.llm.model,
            *self.engine.allocated_gpu_uuids,
            *(threshold.trigger for threshold in self.thresholds.values()),
            *(threshold.target for threshold in self.thresholds.values()),
            *(threshold.scale for threshold in self.thresholds.values()),
        )
        if any(
            isinstance(value, str)
            and ("REPLACE" in value or value.startswith("GPU-REPLACE"))
            for value in values
        ):
            raise ValueError("real execution configuration still contains placeholders")


class CompileTrace(StrictModel):
    schema_version: Literal[8]
    trial_id: str
    phase: Literal["initial", "bo"]
    bo_round: int | None = Field(default=None, ge=1)
    action_version: int = Field(ge=1)
    compile_base_id: str
    base_trial_id: str | None = None
    selected_metrics: tuple[str, ...]
    selected_actions: tuple[str, ...]
    requested_coefficients: dict[str, float]
    coefficients: dict[str, float]
    numeric_decisions: dict[str, dict[str, Any]]
    categorical_decisions: dict[str, dict[str, Any]]
    final_config: dict[str, Any]
    effective_parameter_delta: dict[str, Any]
    constraint_errors: tuple[str, ...]
    config_hash: str

    @model_validator(mode="after")
    def validate_trace(self) -> CompileTrace:
        if tuple(self.coefficients) != ACTION_IDS:
            raise ValueError("coefficients must contain A01 through A08 in order")
        if any(not isfinite(value) or value < -1 or value > 1 for value in self.coefficients.values()):
            raise ValueError("coefficients must be finite in [-1, 1]")
        if any(metric not in METRIC_IDS for metric in self.selected_metrics):
            raise ValueError("selected_metrics contains an unknown metric")
        if any(action not in ACTION_IDS for action in self.selected_actions):
            raise ValueError("selected_actions contains an unknown action")
        if self.final_config and tuple(self.final_config) != PARAMETER_IDS:
            raise ValueError("final_config must contain p01 through p15 in order")
        return self

    @property
    def valid(self) -> bool:
        return not self.constraint_errors


class EncodedConfig(StrictModel):
    trial_id: str
    compile_base_id: str
    base_trial_id: str
    action_version: int = Field(ge=1)
    encoder_version: Literal[1] = 1
    parameter_order: tuple[str, ...]
    features: tuple[float, ...]
    effective_parameter_delta: dict[str, float | int]
    final_config_hash: str

    @model_validator(mode="after")
    def validate_encoding(self) -> EncodedConfig:
        if self.parameter_order != PARAMETER_IDS or len(self.features) != len(PARAMETER_IDS):
            raise ValueError("encoded configuration must use the canonical 15 parameters")
        if any(not isfinite(value) for value in self.features):
            raise ValueError("encoded features must be finite")
        return self


class TrialStatus(str, Enum):
    SUCCESS = "success"
    STARTUP_FAILED = "startup_failed"
    BENCHMARK_FAILED = "benchmark_failed"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    INTERNAL_ERROR = "internal_error"


class TrialResult(StrictModel):
    trial_id: str
    run_mode: RunMode
    status: TrialStatus
    trace: CompileTrace
    vllm_execution_mode: Literal["V0", "V1"] | None = None
    metrics: dict[str, float | None]
    metric_missing_reasons: dict[str, str]
    throughput_tps: float | None = Field(default=None, ge=0)
    request_count: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    successful_requests: int = Field(ge=0)
    configured_request_rate_rps: float | None = Field(default=None, gt=0)
    issued_request_rate_rps: float | None = Field(default=None, ge=0)
    completed_request_rate_rps: float | None = Field(default=None, ge=0)
    queue_backlog_p95: float | None = Field(default=None, ge=0)
    backlog_detected: bool = False
    duration_s: float | None = Field(default=None, gt=0)
    sampling_interval_s: float | None = Field(default=None, gt=0)
    valid_sample_counts: dict[str, int]
    log_paths: dict[str, str]
    cleanup_result: str

    @model_validator(mode="after")
    def validate_result(self) -> TrialResult:
        if set(self.metrics) != set(METRIC_IDS):
            raise ValueError("metrics must contain m01 through m06")
        if not 0 <= self.successful_requests <= self.completed_requests <= self.request_count:
            raise ValueError("request counts are inconsistent")
        if self.status == TrialStatus.SUCCESS and self.throughput_tps is None:
            raise ValueError("successful trials require throughput_tps")
        return self


class MetricTarget(StrictModel):
    kind: Literal["upper", "lower", "interval", "point"]
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    scale: float = Field(gt=0)


class MetricSelection(StrictModel):
    reference_trial_id: str
    mode: Literal["hard_threshold", "g_counterfactual", "uncertainty", "rotation"]
    selected_metrics: tuple[str, ...]
    scores: dict[str, float]
    weights: dict[str, float]
    targets: dict[str, MetricTarget]
    credible_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    g_posterior: dict[str, dict[str, float]] = Field(default_factory=dict)


class BOSelection(StrictModel):
    bo_round: int = Field(ge=1)
    action_version: int = Field(ge=1)
    base_trial_id: str
    selected_metrics: tuple[str, ...]
    selected_actions: tuple[str, ...]
    bo_dimension: int = Field(ge=1, le=8)
    coefficients: dict[str, float]
    predictions: dict[str, Any]
    acquisition_name: str
    acquisition_value: float | None = None
    source: Literal["metric_ei", "uncertainty_exploration", "sobol_exploration"]


class WeightUpdate(StrictModel):
    action_id: str
    parameter_id: str
    desired_parameter_effect: Literal["increase", "decrease", "neutral"]
    weight_operation: Literal["strengthen", "weaken", "reverse", "set_zero", "keep"]
    old_weight: float
    new_weight: float
    target_metrics: tuple[str, ...]
    evidence_trial_ids: tuple[str, ...]
    reason: str


class ActionUpdate(StrictModel):
    decision: Literal["keep", "modify"]
    parent_action_version: int = Field(ge=1)
    effective_from_bo_round: int = Field(ge=1)
    updates: tuple[WeightUpdate, ...]


def load_yaml_model(path: Path, model_type: type[ModelT]) -> ModelT:
    with path.open("r", encoding="utf-8") as file:
        return model_type.model_validate(yaml.safe_load(file))


def load_parameters(path: Path) -> ParameterCatalog:
    return load_yaml_model(path, ParameterCatalog)


def load_actions(path: Path) -> ActionBundle:
    return load_yaml_model(path, ActionBundle)


def load_graph(path: Path) -> Graph:
    return load_yaml_model(path, Graph)


def load_experiment(path: Path, *, for_execution: bool = False) -> ExperimentConfig:
    config = load_yaml_model(path, ExperimentConfig)
    if for_execution:
        config.validate_for_execution()
    return config
