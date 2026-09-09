import pytest

from dibo.controller import run_experiment
from dibo.report import build_report, write_report
from dibo.schemas import RunMode
from tests.unit.test_controller import cpu_config
from tests.unit.test_initial_design import make_result


def test_report_separates_measured_best_from_predictions() -> None:
    base = make_result("base", tps=100)
    best = make_result("best", tps=110)
    text = build_report(
        [base, best], base, [{"predicted_tps": 99999}], [], run_mode=RunMode.SYNTHETIC
    )
    assert "Run type: **synthetic**" in text
    assert "Trial: best; TPS: 110" in text
    assert "TPS change against x_base: 10.0000%" in text
    assert "not vLLM measurements" in text


def test_report_handles_empty_and_no_improvement_history() -> None:
    assert "Unavailable" in build_report([], None, [], [], run_mode=RunMode.SYNTHETIC)
    base = make_result("base")
    assert "0.0000%" in build_report([base], base, [], [])


def test_mixed_run_types_are_rejected() -> None:
    synthetic = make_result("synthetic")
    real = make_result("real").model_copy(update={"run_mode": RunMode.REAL})
    with pytest.raises(ValueError, match="mixed"):
        build_report([synthetic, real], synthetic, [], [])


@pytest.mark.asyncio
async def test_offline_report_rebuilds_from_saved_results(tmp_path) -> None:
    config = cpu_config(tmp_path, budget=5, initial=5)
    await run_experiment(config, initial_only=True)
    first = write_report(config.run_dir).read_text()
    assert write_report(config.run_dir).read_text() == first
    assert len((config.run_dir / "trials.csv").read_text().splitlines()) == 6
