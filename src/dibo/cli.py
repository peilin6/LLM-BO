"""Command-line entry point for DIBO."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import typer
import yaml

from dibo import __version__

app = typer.Typer(
    name="dibo",
    help="Dynamic-neighbor Action Bayesian optimization research prototype.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the DIBO version and exit.",
    ),
) -> None:
    """Run DIBO commands."""


def _failure(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


@app.command("env-check")
def env_check(config: Annotated[Path, typer.Option("--config")]) -> None:
    """Validate config and installed package metadata without probing GPU devices."""
    from dibo.schemas import load_experiment

    try:
        experiment = load_experiment(config)
        packages = {}
        for name in (
            "dibo",
            "httpx",
            "numpy",
            "pydantic",
            "PyYAML",
            "scipy",
            "scikit-learn",
            "typer",
        ):
            packages[name] = package_version(name)
        blocker = None
        try:
            experiment.validate_for_execution()
        except ValueError as error:
            blocker = str(error)
        typer.echo(
            json.dumps(
                {
                    "run_mode": experiment.run_mode.value,
                    "packages": packages,
                    "gpu_probed": False,
                    "execution_blocker": blocker,
                },
                indent=2,
            )
        )
    except (ValueError, OSError, PackageNotFoundError) as error:
        _failure(error)


def _run(ctx: typer.Context, config: Path, initial_only: bool, *, disable_llm: bool = False) -> None:
    from dibo.controller import load_controller_config, run_experiment
    from dibo.llm_update import FileRoundtripReviewer
    from dibo.report import write_report
    from dibo.schemas import RunMode

    overrides = ctx.obj or {}
    try:
        settings = load_controller_config(config, run_root=Path(overrides.get("run_root", "runs")))
        reviewer = overrides.get("reviewer")
        if reviewer is None and settings.experiment.run_mode == RunMode.REAL and not disable_llm:
            reviewer = FileRoundtripReviewer(
                settings.run_dir / "llm_io",
                timeout_s=settings.experiment.llm.timeout_s,
            )
        summary = asyncio.run(
            run_experiment(
                settings,
                runner=overrides.get("runner"),
                reviewer=reviewer,
                initial_only=initial_only,
            )
        )
        path = write_report(settings.run_dir)
        typer.echo(
            json.dumps(
                {
                    "run_mode": summary.run_mode.value,
                    "attempted_trials": summary.attempted_trials,
                    "stop_reason": summary.stop_reason,
                    "report": str(path),
                }
            )
        )
    except (ValueError, RuntimeError, OSError) as error:
        _failure(error)


@app.command("initial-run")
def initial_run(ctx: typer.Context, config: Annotated[Path, typer.Option("--config")]) -> None:
    """Execute the synthetic initial design in a new run directory."""
    _run(ctx, config, True, disable_llm=True)


@app.command()
def tune(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config")],
    no_llm: Annotated[bool, typer.Option("--no-llm")] = False,
) -> None:
    """Execute the complete synthetic loop; real execution is not enabled in this stage."""
    _run(ctx, config, False, disable_llm=no_llm)


@app.command("compile")
def compile_command(
    run: Annotated[Path, typer.Option("--run")],
    base_trial: Annotated[str, typer.Option("--base-trial")],
    actions_version: Annotated[int, typer.Option("--actions-version", min=1)],
    coefficients_file: Annotated[Path, typer.Option("--z")],
    round_number: Annotated[int | None, typer.Option("--round", min=1)] = None,
) -> None:
    """Compile from the saved fixed base and saved round Action union, without execution."""
    from dibo.compiler import CompileEnvironment, compile_config
    from dibo.schemas import ParameterCatalog, TrialResult, load_actions

    try:
        base = TrialResult.model_validate_json((run / "base_trial.json").read_text())
        if base.trial_id != base_trial:
            raise ValueError("--base-trial must match the fixed base in this run")
        records = [
            json.loads(path.read_text())
            for path in sorted((run / "selections").glob("round_*.json"))
        ]
        matches = [
            record
            for record in records
            if record["action_version"] == actions_version
            and (round_number is None or record["bo_round"] == round_number)
        ]
        if not matches:
            raise ValueError("no saved selection for this Action version and round")
        selection = matches[-1]
        coefficients = yaml.safe_load(coefficients_file.read_text(encoding="utf-8"))
        if not isinstance(coefficients, dict):
            raise TypeError("--z must contain an Action-to-coefficient mapping")
        payload = json.loads((run / "environment.json").read_text())["compiler"]
        payload["parameter_catalog"] = ParameterCatalog.model_validate(payload["parameter_catalog"])
        environment = CompileEnvironment(**payload)
        bundle = load_actions(run / "action_versions" / f"actions_v{actions_version}.yaml")
        trace = compile_config(
            base.trace.final_config,
            bundle,
            selection["selected_actions"],
            coefficients,
            environment,
            base_trial_id=base_trial,
            compile_base_id=base_trial,
            bo_round=selection["bo_round"],
            selected_metrics=selection["selection"]["selected_metrics"],
        )
        if not trace.valid:
            raise ValueError(f"invalid candidate: {trace.constraint_errors}")
        typer.echo(trace.model_dump_json(indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        _failure(error)


@app.command("llm-review")
def llm_review(
    ctx: typer.Context,
    run: Annotated[Path, typer.Option("--run")],
    round_number: Annotated[int, typer.Option("--round", min=1)],
) -> None:
    """Review saved evidence without running an additional Trial or changing its history."""
    from dibo.llm_update import (
        EvidenceBundle,
        UpdateValidationError,
        keep_update,
        review_round,
        validate_update,
    )
    from dibo.schemas import Graph
    from dibo.store import save_json_atomic

    try:
        saved = json.loads((run / "llm_reviews" / f"round_{round_number:03}.json").read_text())
        evidence = EvidenceBundle.model_validate(saved["evidence"])
        reviewer = (ctx.obj or {}).get("reviewer")
        update = review_round(evidence, reviewer) if reviewer is not None else keep_update(evidence)
        graph = Graph.model_validate(json.loads((run / "graph.yaml").read_text()))
        environment = json.loads((run / "environment.json").read_text())["compiler"]
        reason = None
        try:
            validate_update(
                evidence.action_bundle,
                update,
                evidence,
                graph,
                gpu_count=environment["allocated_gpu_count"],
            )
        except UpdateValidationError as error:
            reason = str(error)
            update = keep_update(evidence)
        result = {"update": update.model_dump(mode="json"), "rejection_reason": reason}
        save_json_atomic(run / "manual_reviews" / f"round_{round_number:03}.json", result)
        typer.echo(json.dumps(result))
    except (ValueError, OSError, KeyError) as error:
        _failure(error)


@app.command()
def report(run: Annotated[Path, typer.Option("--run")]) -> None:
    """Rebuild report and CSV from recorded outcomes without GPU or API access."""
    from dibo.report import write_report

    try:
        typer.echo(str(write_report(run)))
    except (ValueError, OSError, KeyError) as error:
        _failure(error)
