import json

from typer.testing import CliRunner

from dibo.cli import app
from dibo.controller import FakeTrialRunner
from tests.unit.test_controller import cpu_config, improving_review


def test_cli_tune_compile_review_report_interoperate_without_resources(tmp_path) -> None:
    config = cpu_config(tmp_path, name="cli_integrated")
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(config.experiment.model_dump_json(), encoding="utf-8")
    for name in ("parameters.yaml", "actions_v1.yaml", "graph.yaml"):
        original = __import__("pathlib").Path(__file__).parents[2] / "configs" / name
        (tmp_path / name).write_text(original.read_text(), encoding="utf-8")
    runner = CliRunner()
    injected = {
        "run_root": tmp_path / "runs",
        "runner": FakeTrialRunner(),
        "reviewer": improving_review,
    }
    tuned = runner.invoke(app, ["tune", "--config", str(config_file)], obj=injected)
    assert tuned.exit_code == 0, tuned.exception
    run_dir = tmp_path / "runs" / "cli_integrated"
    results_before = list(run_dir.glob("trials/*/result.json"))
    assert len(results_before) == 14
    record = json.loads((run_dir / "selections/round_001.json").read_text())
    selected = record["suggestions"][0]
    coefficients_file = tmp_path / "z.yaml"
    coefficients_file.write_text(json.dumps(selected["coefficients"]))
    compiled = runner.invoke(
        app,
        [
            "compile",
            "--run",
            str(run_dir),
            "--base-trial",
            selected["base_trial_id"],
            "--actions-version",
            "1",
            "--round",
            "1",
            "--z",
            str(coefficients_file),
        ],
    )
    assert compiled.exit_code == 0, compiled.output
    assert (
        json.loads(compiled.stdout)["config_hash"] == selected["predictions"]["final_config_hash"]
    )
    reviewed = runner.invoke(
        app, ["llm-review", "--run", str(run_dir), "--round", "1"], obj=injected
    )
    assert reviewed.exit_code == 0, reviewed.exception
    assert list(run_dir.glob("trials/*/result.json")) == results_before
    assert runner.invoke(app, ["report", "--run", str(run_dir)]).exit_code == 0
    assert "synthetic" in (run_dir / "report.md").read_text()
    assert (run_dir / "trials.csv").exists()
    coefficients = {**selected["coefficients"], "UNKNOWN": 0.5}
    coefficients_file.write_text(json.dumps(coefficients))
    assert (
        runner.invoke(
            app,
            [
                "compile",
                "--run",
                str(run_dir),
                "--base-trial",
                selected["base_trial_id"],
                "--actions-version",
                "1",
                "--z",
                str(coefficients_file),
            ],
        ).exit_code
        == 1
    )
