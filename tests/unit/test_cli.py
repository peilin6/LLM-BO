from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dibo.cli import app

runner = CliRunner()


def test_help_lists_command_skeleton() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("env-check", "compile", "initial-run", "tune", "llm-review", "report"):
        assert command in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


CONFIGS = Path(__file__).parents[2] / "configs"


def test_env_check_does_not_probe_gpu() -> None:
    result = runner.invoke(app, ["env-check", "--config", str(CONFIGS / "experiment_smoke.yaml")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["gpu_probed"] is False


def test_initial_cli_uses_fake_and_report_is_offline(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["initial-run", "--config", str(CONFIGS / "experiment_smoke.yaml")],
        obj={"run_root": tmp_path},
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout)["attempted_trials"] == 10
    run = tmp_path / "dibo_cpu_smoke"
    assert runner.invoke(app, ["report", "--run", str(run)]).exit_code == 0
    assert "synthetic" in (run / "report.md").read_text()
    repeated = runner.invoke(
        app,
        ["initial-run", "--config", str(CONFIGS / "experiment_smoke.yaml")],
        obj={"run_root": tmp_path},
    )
    assert repeated.exit_code == 1


def test_cli_refuses_real_execution_and_missing_inputs(tmp_path) -> None:
    result = runner.invoke(
        app, ["tune", "--config", str(CONFIGS / "experiment.yaml")], obj={"run_root": tmp_path}
    )
    assert result.exit_code == 1
    assert not list(tmp_path.iterdir())
    assert runner.invoke(app, ["compile"]).exit_code != 0
    assert runner.invoke(app, ["report", "--run", str(tmp_path / "missing")]).exit_code == 1


def test_tune_help_exposes_no_llm_mode() -> None:
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0
    assert "--no-llm" in result.stdout
