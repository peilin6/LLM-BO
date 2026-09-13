"""Sequential CPU-testable orchestration of the DIBO experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from dibo import benchmark as benchmark_module
from dibo import engine as engine_module
from dibo import store
from dibo.benchmark import BenchmarkResult
from dibo.compiler import CompileEnvironment, compile_config
from dibo.engine import EngineLaunchSpec
from dibo.initial_design import (
    InitialDesignConfig,
    InitialDesignError,
    choose_base_trial,
    generate_initial_candidates,
)
from dibo.llm_update import (
    ReviewAbortedError,
    UpdateValidationError,
    apply_update,
    build_evidence,
    keep_update,
    review_round,
    save_action_bundle,
)
from dibo.metrics import MetricsResult, MetricValue, PrometheusNvmlMetrics
from dibo.models import GPModel, fit_f, fit_g
from dibo.observability import emit_event, stop_message, stop_requested
from dibo.optimizer import OptimizerConfig, suggest
from dibo.schemas import (
    ACTION_IDS,
    ActionBundle,
    CompileTrace,
    ExperimentConfig,
    Graph,
    RunMode,
    TrialResult,
    TrialStatus,
    load_actions,
    load_experiment,
    load_graph,
    load_parameters,
)
from dibo.selector import SelectorConfig, select_metrics, union_neighbor_actions
from dibo.trace_encoder import encode_trace
from dibo.trial import TrialSpec, run_trial

Runner = Callable[[TrialSpec], Awaitable[TrialResult]]
Reviewer = Callable[[dict[str, Any]], Mapping[str, Any] | str]


@dataclass(frozen=True)
class ControllerConfig:
    experiment: ExperimentConfig
    actions: ActionBundle
    graph: Graph
    environment: CompileEnvironment
    x_init: dict[str, Any]
    run_dir: Path


@dataclass
class ExperimentSummary:
    run_mode: RunMode
    run_dir: Path
    attempted_trials: int = 0
    stop_reason: str = "completed"
    history: list[TrialResult] = field(default_factory=list)
    base_trial: TrialResult | None = None
    best_trial: TrialResult | None = None
    selections: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0


def load_controller_config(path: Path, *, run_root: Path = Path("runs")) -> ControllerConfig:
    """Load a synthetic profile or a fully resolved and locally reproducible real profile."""
    experiment = load_experiment(path)
    directory = path.parent
    catalog = load_parameters(directory / "parameters.yaml")
    workload_path = Path(experiment.workload.request_file)
    if not workload_path.is_absolute():
        workload_path = (directory / workload_path).resolve()
    experiment = experiment.model_copy(update={"workload": experiment.workload.model_copy(
        update={"request_file": str(workload_path)}
    )})
    if experiment.run_mode == RunMode.SYNTHETIC:
        resolved = {"MIN_FIT_TP": 1, "PROMPT_P95_OR_8192": 8192, "PROMPT_P75": 2048,
                    "FROM_REQUEST_PREFIX_PROFILE": "shared_prefix" in experiment.workload.prefix_profile}
        model_heads = model_layers = 32
    else:
        experiment.validate_for_execution()
        if experiment.workload.backend != "openai-chat":
            raise ValueError("real HTTP benchmark currently requires backend=openai-chat")
        if not workload_path.is_file():
            raise ValueError(f"workload request file does not exist: {workload_path}")
        digest = hashlib.sha256(workload_path.read_bytes()).hexdigest()
        if digest != experiment.workload.request_sha256:
            raise ValueError("workload request SHA256 does not match configuration")
        model_path = Path(experiment.engine.model)
        model_config_path = model_path / "config.json"
        if not model_config_path.is_file():
            raise ValueError(f"model config does not exist: {model_config_path}")
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        model_heads = int(model_config["num_attention_heads"])
        model_layers = int(model_config["num_hidden_layers"])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(experiment.engine.tokenizer, local_files_only=True)
        lengths = []
        for row in benchmark_module.load_requests(workload_path):
            tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                     add_generation_prompt=True)
            lengths.append(len(tokens))
        if not lengths:
            raise ValueError("workload produced no token lengths")
        import numpy as np

        def aligned_quantile(value: float) -> int:
            return min(experiment.engine.max_model_len,
                       max(256, math.floor((value + 255) / 256) * 256))

        resolved = {
            "MIN_FIT_TP": 1,
            "PROMPT_P95_OR_8192": min(32768, max(8192, aligned_quantile(float(np.quantile(lengths, 0.95, method="linear"))))),
            "PROMPT_P75": aligned_quantile(float(np.quantile(lengths, 0.75, method="linear"))),
            "FROM_REQUEST_PREFIX_PROFILE": "shared_prefix" in experiment.workload.prefix_profile,
        }
    initial = {spec.parameter_id: resolved[spec.initial] if isinstance(spec.initial, str)
               else spec.initial for spec in catalog.parameters}
    environment = CompileEnvironment(
        catalog,
        experiment.engine.allocated_gpu_count,
        model_heads,
        model_layers,
        experiment.engine.max_model_len,
        (8, 16, 32),
        model_context=experiment.engine.model,
        workload_context=experiment.workload.request_sha256,
        execution_mode=experiment.engine.execution_mode,
        engine_version=experiment.engine.version,
    )
    if Path(
        experiment.experiment_id
    ).name != experiment.experiment_id or experiment.experiment_id in {"", ".", ".."}:
        raise ValueError("experiment_id must be a single directory name")
    return ControllerConfig(
        experiment,
        load_actions(directory / "actions_v1.yaml"),
        load_graph(directory / "graph.yaml"),
        environment,
        initial,
        run_root / experiment.experiment_id,
    )


def synthetic_metrics(final_config: Mapping[str, Any]) -> dict[str, float]:
    """A deterministic test surface depending on executed parameters, never on z."""
    concurrency = (float(final_config["p03"]) - 128.0) / 800.0
    tokens = (float(final_config["p04"]) - 8192.0) / 32768.0
    kv = min(0.99, max(0.05, 0.89 + concurrency - 2 * (float(final_config["p06"]) - 0.9)))
    return {
        "m01": kv,
        "m02": 0.001 + max(0.0, kv - 0.9) * 0.4,
        "m03": min(0.99, max(0.01, 0.6 + tokens + concurrency)),
        "m04": max(0.0, 4.0 - 10 * concurrency + 5 * tokens),
        "m05": min(0.99, max(0.01, 0.45 + tokens - concurrency)),
        "m06": max(0.05, 0.55 + tokens + 0.05 * bool(final_config["p15"])),
    }


class _FakeEngine:
    async def start(self, config: EngineLaunchSpec, run_dir: Path) -> dict[str, Any]:
        return dict(config.final_config)

    async def wait_ready(self, handle: dict[str, Any], timeout_s: float) -> str:
        return "V1"

    async def stop(self, handle: dict[str, Any]) -> None:
        return None


class _FakeBenchmark:
    async def run_warmup(self, handle: dict[str, Any], workload: Any) -> None:
        return None

    async def run_measurement(self, handle: dict[str, Any], workload: Any) -> BenchmarkResult:
        values = synthetic_metrics(handle)
        throughput = 1500 * (1 - 0.3 * values["m01"]) / (1 + values["m02"] + values["m06"])
        tokens = workload.num_prompts * workload.output_len
        return BenchmarkResult(
            workload.num_prompts,
            workload.num_prompts,
            0,
            tokens,
            tokens / throughput,
            throughput,
            values["m06"],
            1.0,
            (),
        )


class _FakeMetrics:
    def start_sampling(
        self, handle: dict[str, Any], gpu_uuid: str, interval_s: float, *, min_valid_samples: int
    ):
        return synthetic_metrics(handle), min_valid_samples

    def stop_and_aggregate(
        self, sampler: Any, completed_requests: int, *, ttft_p95_s: float | None
    ):
        values, count = sampler
        return MetricsResult(
            {metric: MetricValue(value, valid_samples=count) for metric, value in values.items()}
        )


class FakeTrialRunner:
    """Use the real Trial lifecycle with injectable, resource-free adapters."""

    def __init__(self, *, engine=None, benchmark=None, metrics=None) -> None:
        self.engine = engine if engine is not None else _FakeEngine()
        self.benchmark = benchmark if benchmark is not None else _FakeBenchmark()
        self.metrics = metrics if metrics is not None else _FakeMetrics()

    async def __call__(self, spec: TrialSpec) -> TrialResult:
        if spec.run_mode != RunMode.SYNTHETIC:
            raise ValueError("FakeTrialRunner cannot label synthetic results as real")
        return await run_trial(spec, self.engine, self.benchmark, self.metrics, store)


class RealTrialRunner:
    """Compose the real engine, HTTP workload and Prometheus/NVML boundaries."""

    def __init__(self, *, seed: int = 42) -> None:
        self.benchmark = benchmark_module.VLLMHttpBenchmark(seed=seed)
        self.metrics = PrometheusNvmlMetrics()

    async def __call__(self, spec: TrialSpec) -> TrialResult:
        if spec.run_mode != RunMode.REAL:
            raise ValueError("RealTrialRunner only accepts real Trial specs")

        class EngineBoundary:
            start = staticmethod(engine_module.start)
            wait_ready = staticmethod(engine_module.wait_ready)
            stop = staticmethod(engine_module.stop)

        benchmark = self.benchmark

        class BenchmarkBoundary:
            @staticmethod
            async def run_warmup(handle, workload) -> None:
                await benchmark.run_requests(handle, workload, warmup=True)

            @staticmethod
            async def run_measurement(handle, workload) -> BenchmarkResult:
                payload = await benchmark.run_requests(handle, workload, warmup=False)
                return benchmark_module.aggregate_measurement(payload)

        return await run_trial(spec, EngineBoundary(), BenchmarkBoundary(), self.metrics, store)


def _model_metadata(model: GPModel) -> dict[str, Any]:
    return {
        "fit_status": model.fit_status,
        "failure_reason": model.failure_reason,
        "feature_order": model.feature_order,
        "training_trial_ids": model.training_trial_ids,
        "observation_noise": model.observation_noise,
        "n_unique_configs": model.n_unique_configs,
    }


def _append_event(run_dir: Path, event: Mapping[str, Any]) -> None:
    payload = dict(event)
    emit_event(run_dir, str(payload.pop("event")), **payload)


def _selection_reason(selection: Any, reference: TrialResult) -> str:
    metrics = ", ".join(selection.selected_metrics)
    if selection.mode == "hard_threshold":
        return (
            f"selected {metrics} because the reference trial {reference.trial_id} "
            "violated configured hard thresholds; scores are normalized threshold gaps"
        )
    if selection.mode == "g_counterfactual":
        return (
            f"selected {metrics} because one-at-a-time G-model counterfactuals predicted "
            "a positive lower-confidence TPS improvement over the reference metrics"
        )
    if selection.mode == "uncertainty":
        return (
            f"selected {metrics} because hard thresholds and G counterfactuals were inactive, "
            "so F-model posterior uncertainty at z=0 drove exploration"
        )
    return (
        f"selected {metrics} by deterministic rotation because neither thresholds, "
        "G counterfactuals, nor ready F-model uncertainties were available"
    )


async def run_experiment(
    config: ControllerConfig,
    *,
    runner: Runner | None = None,
    reviewer: Reviewer | None = None,
    initial_only: bool = False,
) -> ExperimentSummary:
    """Run one new sequential experiment using adapters selected by its strict run mode."""
    experiment = ExperimentConfig.model_validate(config.experiment.model_dump())
    if experiment.run_mode == RunMode.REAL:
        experiment.validate_for_execution()
    baseline_trace = compile_config(
        config.x_init, config.actions, (), {}, config.environment, phase="initial"
    )
    if not baseline_trace.valid:
        raise ValueError(f"invalid x_init: {baseline_trace.constraint_errors}")
    config.run_dir.mkdir(parents=True, exist_ok=False)
    store.save_json_atomic(
        config.run_dir / "experiment_snapshot.yaml", experiment.model_dump(mode="json")
    )
    environment_payload = asdict(config.environment)
    environment_payload["parameter_catalog"] = config.environment.parameter_catalog.model_dump(
        mode="json"
    )
    store.save_json_atomic(
        config.run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "run_mode": experiment.run_mode.value,
            "compiler": environment_payload,
            "synthetic_initialization": experiment.run_mode == RunMode.SYNTHETIC,
        },
    )
    store.save_json_atomic(config.run_dir / "graph.yaml", config.graph.model_dump(mode="json"))
    store.save_json_atomic(config.run_dir / "x_init.json", config.x_init)
    save_action_bundle(config.actions, config.run_dir / "action_versions")
    summary = ExperimentSummary(experiment.run_mode, config.run_dir)
    started = time.monotonic()
    trial_runner = runner if runner is not None else (
        FakeTrialRunner() if experiment.run_mode == RunMode.SYNTHETIC else RealTrialRunner(seed=experiment.seed)
    )
    current_bundle = config.actions
    tuning = experiment.tuning
    predictions: dict[str, Any] = {}

    def capacity() -> bool:
        if stop_requested(config.run_dir):
            summary.stop_reason = "stop_requested"
            _append_event(config.run_dir, {
                "event": "stop_requested",
                "message": stop_message(config.run_dir),
            })
            return False
        if summary.attempted_trials >= tuning.max_total_trials:
            return False
        if tuning.time_limit_s is not None and time.monotonic() - started >= tuning.time_limit_s:
            summary.stop_reason = "time_limit"
            return False
        return summary.stop_reason == "completed"

    async def execute(trace: CompileTrace) -> TrialResult:
        _append_event(config.run_dir, {
            "event": "trial_started", "trial_id": trace.trial_id,
            "phase": trace.phase, "bo_round": trace.bo_round,
            "action_version": trace.action_version, "config_hash": trace.config_hash,
            "selected_metrics": trace.selected_metrics,
            "selected_actions": trace.selected_actions,
            "coefficients": trace.coefficients,
            "final_config": trace.final_config,
        })
        summary.attempted_trials += 1
        result = await trial_runner(
            TrialSpec(
                run_mode=experiment.run_mode,
                trace=trace,
                engine_launch=EngineLaunchSpec(experiment.engine, trace.final_config),
                workload=experiment.workload,
                run_dir=config.run_dir,
                startup_timeout_s=experiment.engine.startup_timeout_s,
                benchmark_timeout_s=experiment.engine.benchmark_timeout_s,
                gpu_uuid=experiment.engine.allocated_gpu_uuids[0],
                sampling_interval_s=experiment.metrics.sampling_interval_s,
                min_valid_samples=experiment.metrics.min_valid_samples,
            )
        )
        if (
            result.run_mode != experiment.run_mode
            or result.trace != trace
            or result.trial_id != trace.trial_id
        ):
            raise ValueError(
                "Runner returned a different run type, Trial identity, or configuration"
            )
        summary.history.append(result)
        backlog_basis = "m04 queue_length p95 from vLLM Prometheus samples"
        _append_event(config.run_dir, {
            "event": "trial_completed", "trial_id": result.trial_id,
            "status": result.status.value, "action_version": result.trace.action_version,
            "selected_metrics": result.trace.selected_metrics,
            "selected_actions": result.trace.selected_actions,
            "config_hash": result.trace.config_hash, "metrics": result.metrics,
            "throughput_tps": result.throughput_tps, "cleanup_result": result.cleanup_result,
            "request_count": result.request_count,
            "completed_requests": result.completed_requests,
            "successful_requests": result.successful_requests,
            "configured_request_rate_rps": result.configured_request_rate_rps,
            "issued_request_rate_rps": result.issued_request_rate_rps,
            "completed_request_rate_rps": result.completed_request_rate_rps,
            "queue_backlog_p95": result.queue_backlog_p95,
            "backlog_detected": result.backlog_detected,
            "backlog_basis": backlog_basis,
        })
        if "cleanup_failed" in result.cleanup_result:
            summary.stop_reason = "cleanup_failed"
        elif result.status == TrialStatus.INTERRUPTED:
            summary.stop_reason = "interrupted"
        return result

    def encoded_history() -> list[TrialResult]:
        assert summary.base_trial is not None
        enriched = []
        for result in summary.history:
            encoded = encode_trace(
                result.trace,
                summary.base_trial.trace.final_config,
                config.environment.parameter_catalog,
                base_trial_id=summary.base_trial.trial_id,
            )
            enriched.append(
                result.model_copy(
                    update={
                        "trace": result.trace.model_copy(
                            update={
                                "base_trial_id": summary.base_trial.trial_id,
                                "effective_parameter_delta": encoded.effective_parameter_delta,
                            }
                        )
                    }
                )
            )
        return enriched

    def fit_models():
        history = encoded_history()
        options = {
            "min_train_samples": experiment.models.min_train_samples,
            "min_unique_configs": experiment.models.min_unique_configs,
            "seed": experiment.seed,
        }
        with threadpool_limits(limits=experiment.models.cpu_threads):
            return fit_f(
                history,
                config.graph,
                current_bundle,
                historical_noise_multiplier=experiment.models.historical_action_version_noise_multiplier,
                **options,
            ), fit_g(history, **options)

    try:
        _append_event(config.run_dir, {"event": "initial_design_started", "count": tuning.initial_trials})
        candidates = generate_initial_candidates(
            InitialDesignConfig(
                tuning.initial_trials, tuning.initial_probe_coefficient, experiment.seed
            ),
            lambda coefficients, index: compile_config(
                config.x_init,
                current_bundle,
                ACTION_IDS,
                coefficients,
                config.environment,
                trial_id=f"design_{index:03}",
                phase="initial",
                compile_base_id="x_init",
            ),
        )
    except InitialDesignError:
        candidates = ()
        summary.stop_reason = "no_initial_candidate"
    for candidate in candidates:
        if not capacity():
            break
        trace = candidate.trace.model_copy(
            update={"trial_id": f"trial_{summary.attempted_trials + 1:03}"}
        )
        await execute(trace)
    try:
        summary.base_trial = choose_base_trial(
            summary.history, required_success_rate=experiment.workload.required_success_rate
        )
    except InitialDesignError:
        if summary.stop_reason == "completed":
            summary.stop_reason = "no_successful_initial_trial"
    if summary.base_trial is not None:
        _append_event(config.run_dir, {
            "event": "base_trial_selected", "trial_id": summary.base_trial.trial_id,
            "throughput_tps": summary.base_trial.throughput_tps,
        })
        store.save_json_atomic(
            config.run_dir / "base_trial.json", summary.base_trial.model_dump(mode="json")
        )
    for bo_round in range(1, tuning.bo_rounds + 1):
        if initial_only or summary.base_trial is None or not capacity():
            break
        _append_event(config.run_dir, {"event": "model_fit_started", "bo_round": bo_round})
        f_models, g_model = fit_models()
        _append_event(config.run_dir, {"event": "model_fit_completed", "bo_round": bo_round,
            "f_status": {metric: model.fit_status for metric, model in f_models.items()},
            "g_status": g_model.fit_status})
        eligible = [item for item in summary.history if item.status == TrialStatus.SUCCESS]
        reference = min(
            eligible,
            key=lambda item: (
                -item.throughput_tps,
                item.metrics["m06"] if item.metrics["m06"] is not None else float("inf"),
                item.trial_id,
            ),
        )
        thresholds = {}
        for metric, threshold in experiment.thresholds.items():
            values = threshold.model_dump()
            for field_name in ("trigger", "target"):
                if values.get(f"{field_name}_from") == "workload.ttft_slo_s":
                    values[field_name] = experiment.workload.ttft_slo_s
            thresholds[metric] = type(threshold).model_validate(values)
        selection = select_metrics(
            reference,
            eligible,
            f_models,
            g_model,
            thresholds,
            config=SelectorConfig(
                metric_top_k=tuning.metric_top_k,
                metric_top_k_max=tuning.metric_top_k_max,
                third_metric_ratio=tuning.third_metric_ratio,
                beta=tuning.g_counterfactual_beta,
                grid_size=tuning.g_counterfactual_grid_size,
                rotation_index=experiment.seed + bo_round - 1,
            ),
        )
        selected_actions = union_neighbor_actions(selection.selected_metrics, config.graph)
        selection_reason = _selection_reason(selection, reference)
        _append_event(config.run_dir, {
            "event": "round_selected", "bo_round": bo_round,
            "action_version": current_bundle.action_version,
            "reference_trial_id": selection.reference_trial_id,
            "reference_throughput_tps": reference.throughput_tps,
            "reference_metrics": reference.metrics,
            "mode": selection.mode, "selected_metrics": selection.selected_metrics,
            "selected_actions": selected_actions, "targets": {
                metric: target.model_dump(mode="json") for metric, target in selection.targets.items()
            }, "scores": selection.scores,
            "weights": selection.weights,
            "credible_ranges": selection.credible_ranges,
            "g_posterior": selection.g_posterior,
            "selection_reason": selection_reason,
        })
        record = {
            "bo_round": bo_round,
            "action_version": current_bundle.action_version,
            "base_trial_id": summary.base_trial.trial_id,
            "selection": selection.model_dump(mode="json"),
            "selection_reason": selection_reason,
            "reference": {
                "trial_id": reference.trial_id,
                "throughput_tps": reference.throughput_tps,
                "metrics": reference.metrics,
            },
            "selected_actions": selected_actions,
            "suggestions": [],
            "models": {},
        }
        round_results = []
        for _ in range(tuning.trials_per_bo_round):
            if not capacity():
                break
            with threadpool_limits(limits=experiment.models.cpu_threads):
                suggestion = suggest(
                    selection,
                    selected_actions,
                    f_models,
                    g_model,
                    summary.base_trial.trace.final_config,
                    current_bundle,
                    summary.history,
                    config=OptimizerConfig(
                        environment=config.environment,
                        bo_round=bo_round,
                        base_trial_id=summary.base_trial.trial_id,
                        candidate_pool_size=tuning.candidate_pool_size,
                        max_candidate_pools=tuning.max_candidate_pools,
                        mc_samples=tuning.mc_samples,
                        seed=experiment.seed + summary.attempted_trials,
                        coefficient_low=tuning.coefficient_low,
                        coefficient_high=tuning.coefficient_high,
                        acquisition_tie_tolerance=tuning.acquisition_tie_tolerance,
                        loss_tie_tolerance=tuning.loss_tie_tolerance,
                    ),
                )
            if suggestion is None:
                summary.stop_reason = "no_candidate"
                break
            _append_event(config.run_dir, {"event": "bo_candidate_selected", "bo_round": bo_round,
                "source": suggestion.source, "acquisition_value": suggestion.acquisition_value,
                "selected_metrics": selection.selected_metrics,
                "selected_actions": selected_actions,
                "coefficients": suggestion.coefficients,
                "requested_z": suggestion.predictions["requested_z"],
                "predicted_loss": suggestion.predictions["predicted_loss"],
                "predicted_tps": suggestion.predictions["predicted_tps"],
                "f_mean": suggestion.predictions["f_mean"],
                "f_variance": suggestion.predictions["f_variance"],
                "candidate_count": suggestion.predictions["candidate_count"],
                "invalid_count": suggestion.predictions["invalid_count"],
                "duplicate_count": suggestion.predictions["duplicate_count"],
                "config_hash": suggestion.predictions["final_config_hash"]})
            trace = compile_config(
                summary.base_trial.trace.final_config,
                current_bundle,
                selected_actions,
                suggestion.coefficients,
                config.environment,
                trial_id=f"trial_{summary.attempted_trials + 1:03}",
                phase="bo",
                bo_round=bo_round,
                compile_base_id=summary.base_trial.trial_id,
                base_trial_id=summary.base_trial.trial_id,
                selected_metrics=selection.selected_metrics,
            )
            if not trace.valid or trace.config_hash != suggestion.predictions["final_config_hash"]:
                raise ValueError("suggestion does not reproduce the compiled candidate")
            if trace.config_hash in {item.trace.config_hash for item in summary.history}:
                summary.stop_reason = "no_candidate"
                break
            trial_result = await execute(trace)
            observed_errors = {
                metric: trial_result.metrics[metric] - mean
                for metric, mean in suggestion.predictions["f_mean"].items()
                if trial_result.metrics[metric] is not None
                and trial_result.status == TrialStatus.SUCCESS
            }
            suggestion = suggestion.model_copy(
                update={
                    "predictions": {
                        **suggestion.predictions,
                        "f_error": observed_errors,
                        "posterior_sample_projection": "ratio_[0,1];other_metrics_nonnegative",
                    }
                }
            )
            predictions[trace.trial_id] = suggestion.predictions
            record["suggestions"].append(
                {"trial_id": trace.trial_id, **suggestion.model_dump(mode="json")}
            )
            round_results.append(trial_result)
            if trial_result.status == TrialStatus.SUCCESS:
                f_models, g_model = fit_models()
        record["models"] = {
            **{metric: _model_metadata(model) for metric, model in f_models.items()},
            "G": _model_metadata(g_model),
        }
        summary.selections.append(record)
        store.save_json_atomic(config.run_dir / "selections" / f"round_{bo_round:03}.json", record)
        if round_results:
            _append_event(config.run_dir, {"event": "llm_review_started", "bo_round": bo_round,
                "round_trial_ids": [item.trial_id for item in round_results]})
            enriched = encoded_history()
            round_ids = {item.trial_id for item in round_results}
            evidence = build_evidence(
                current_bundle,
                config.graph,
                selection,
                enriched,
                [item for item in enriched if item.trial_id in round_ids],
                bo_round=bo_round,
                model_evidence={
                    "models": record["models"],
                    "predictions": predictions,
                    "base_trial_id": summary.base_trial.trial_id,
                    "base_metrics": summary.base_trial.metrics,
                },
            )
            try:
                proposal = review_round(evidence, reviewer) if reviewer is not None else keep_update(evidence)
            except ReviewAbortedError:
                summary.stop_reason = "stop_requested"
                _append_event(config.run_dir, {"event": "experiment_stop_during_llm_wait", "bo_round": bo_round})
                break
            rejection = None
            try:
                next_bundle = apply_update(
                    current_bundle,
                    proposal,
                    evidence,
                    config.graph,
                    gpu_count=config.environment.allocated_gpu_count,
                )
            except UpdateValidationError as error:
                rejection = str(error)
                proposal = keep_update(evidence)
                next_bundle = current_bundle
            review = {
                "evidence": evidence.model_dump(mode="json"),
                "update": proposal.model_dump(mode="json"),
                "rejection_reason": rejection,
            }
            summary.reviews.append(review)
            _append_event(config.run_dir, {
                "event": "llm_review", "bo_round": bo_round,
                "decision": proposal.decision,
                "parent_action_version": proposal.parent_action_version,
                "next_action_version": next_bundle.action_version,
                "rejection_reason": rejection,
            })
            store.save_json_atomic(
                config.run_dir / "llm_reviews" / f"round_{bo_round:03}.json", review
            )
            if next_bundle.action_version != current_bundle.action_version:
                save_action_bundle(next_bundle, config.run_dir / "action_versions")
            current_bundle = next_bundle
    eligible = [item for item in summary.history if item.status == TrialStatus.SUCCESS]
    if eligible:
        summary.best_trial = min(
            eligible,
            key=lambda item: (
                -item.throughput_tps,
                item.metrics["m06"] if item.metrics["m06"] is not None else float("inf"),
                item.trial_id,
            ),
        )
    summary.elapsed_s = time.monotonic() - started
    if summary.base_trial is not None:
        with (config.run_dir / "encoded_history.jsonl").open("x", encoding="utf-8") as output:
            for result in summary.history:
                encoded = encode_trace(
                    result.trace,
                    summary.base_trial.trace.final_config,
                    config.environment.parameter_catalog,
                    base_trial_id=summary.base_trial.trial_id,
                )
                output.write(encoded.model_dump_json() + "\n")
    store.save_json_atomic(
        config.run_dir / "summary.json",
        {
            "run_mode": summary.run_mode.value,
            "attempted_trials": summary.attempted_trials,
            "stop_reason": summary.stop_reason,
            "elapsed_s": summary.elapsed_s,
            "base_trial_id": summary.base_trial.trial_id if summary.base_trial else None,
            "best_trial_id": summary.best_trial.trial_id if summary.best_trial else None,
        },
    )
    _append_event(config.run_dir, {
        "event": "experiment_finished", "run_mode": summary.run_mode.value,
        "attempted_trials": summary.attempted_trials, "stop_reason": summary.stop_reason,
        "base_trial_id": summary.base_trial.trial_id if summary.base_trial else None,
        "best_trial_id": summary.best_trial.trial_id if summary.best_trial else None,
    })
    return summary
