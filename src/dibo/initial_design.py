"""Deterministic initial Action-space design and base-trial selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

from scipy.stats import qmc

from dibo.schemas import ACTION_IDS, CompileTrace, TrialResult, TrialStatus


class InitialDesignError(RuntimeError):
    """Raised when the bounded initial design cannot produce a valid result."""


@dataclass(frozen=True)
class InitialDesignConfig:
    initial_trials: int
    probe_coefficient: float
    seed: int
    max_candidate_attempts: int = 256

    def __post_init__(self) -> None:
        if not 5 <= self.initial_trials <= 10:
            raise ValueError("initial_trials must be between 5 and 10")
        if not 0 < self.probe_coefficient <= 1:
            raise ValueError("probe_coefficient must be in (0, 1]")
        if self.max_candidate_attempts < self.initial_trials:
            raise ValueError("max_candidate_attempts must cover initial_trials")


@dataclass(frozen=True)
class InitialCandidate:
    sequence_number: int
    coefficients: dict[str, float]
    trace: CompileTrace
    source: str


Compiler = Callable[[dict[str, float], int], CompileTrace]


def _zero_point() -> dict[str, float]:
    return {action_id: 0.0 for action_id in ACTION_IDS}


def _probe_point(action_id: str, coefficient: float) -> dict[str, float]:
    point = _zero_point()
    point[action_id] = coefficient
    return point


def _sobol_points(seed: int):
    sampler = qmc.Sobol(d=len(ACTION_IDS), scramble=True, seed=seed)
    while True:
        point = qmc.scale(sampler.random(1), -1.0, 1.0)[0]
        yield {action_id: float(value) for action_id, value in zip(ACTION_IDS, point, strict=True)}


def generate_initial_candidates(
    config: InitialDesignConfig,
    compiler: Compiler,
) -> tuple[InitialCandidate, ...]:
    """Compile a bounded, deterministic set of unique valid initial candidates."""
    planned: list[tuple[dict[str, float], str]] = [(_zero_point(), "zero")]
    sobol = _sobol_points(config.seed)
    if config.initial_trials == 10:
        planned.extend(
            (_probe_point(action_id, config.probe_coefficient), f"probe_{action_id}")
            for action_id in ACTION_IDS
        )
        planned.append((next(sobol), "sobol"))
    else:
        planned.extend((next(sobol), "sobol") for _ in range(config.initial_trials - 1))

    candidates: list[InitialCandidate] = []
    seen_hashes: set[str] = set()
    attempts = 0
    while len(candidates) < config.initial_trials and attempts < config.max_candidate_attempts:
        if attempts < len(planned):
            coefficients, source = planned[attempts]
        else:
            coefficients, source = next(sobol), "sobol_replacement"
        sequence_number = attempts + 1
        trace = compiler(coefficients, sequence_number)
        attempts += 1
        if not trace.valid or trace.config_hash in seen_hashes:
            continue
        seen_hashes.add(trace.config_hash)
        candidates.append(InitialCandidate(sequence_number, coefficients, trace, source))

    if len(candidates) != config.initial_trials:
        raise InitialDesignError(
            f"initial candidate generation exhausted after {attempts} attempts; "
            f"created {len(candidates)} of {config.initial_trials}"
        )
    return tuple(candidates)


def choose_base_trial(
    results: Sequence[TrialResult],
    *,
    required_success_rate: float = 1.0,
) -> TrialResult:
    """Choose the immutable base from valid measured initial Trials."""
    if not 0 <= required_success_rate <= 1:
        raise ValueError("required_success_rate must be in [0, 1]")
    eligible = [
        result
        for result in results
        if result.trace.phase == "initial"
        and result.status == TrialStatus.SUCCESS
        and result.throughput_tps is not None
        and isfinite(result.throughput_tps)
        and result.request_count > 0
        and result.successful_requests / result.request_count >= required_success_rate
    ]
    if not eligible:
        raise InitialDesignError("no successful initial Trial is eligible as x_base")

    def rank(result: TrialResult) -> tuple[float, float, str]:
        ttft = result.metrics.get("m06")
        finite_ttft = float(ttft) if ttft is not None and isfinite(ttft) else float("inf")
        return (-float(result.throughput_tps), finite_ttft, result.trial_id)

    return min(eligible, key=rank)