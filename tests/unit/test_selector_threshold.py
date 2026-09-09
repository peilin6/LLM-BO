from __future__ import annotations

from types import SimpleNamespace

import pytest

from dibo.schemas import Threshold
from dibo.selector import SelectorConfig, select_metrics
from tests.unit.test_initial_design import make_result


def thresholds() -> dict[str, Threshold]:
    return {
        "m01": Threshold(kind="upper", trigger=0.9, target=0.85, scale=0.05),
        "m02": Threshold(kind="upper", trigger=0.02, target=0.01, scale=0.01),
        "m04": Threshold(kind="upper", trigger=8.0, target=4.0, scale=2.0),
        "m06": Threshold(kind="upper", trigger=1.0, target=1.0, scale=0.1),
    }


def reference_with(**metrics: float):
    reference = make_result("trial_reference")
    return reference.model_copy(update={"metrics": {**reference.metrics, **metrics}})


def test_hard_threshold_has_priority_and_equal_trigger_is_selected() -> None:
    reference = reference_with(m01=0.9, m02=0.02)
    selection = select_metrics(
        reference,
        [reference],
        {},
        SimpleNamespace(fit_status="ready"),
        thresholds(),
    )

    assert selection.mode == "hard_threshold"
    assert selection.selected_metrics == ("m01", "m02")
    assert selection.scores == {"m01": 0.0, "m02": 0.0}
    assert selection.weights == {"m01": 0.5, "m02": 0.5}
    assert selection.targets["m01"].value == 0.85


def test_hard_threshold_third_metric_uses_eighty_percent_rule() -> None:
    reference = reference_with(m01=0.95, m02=0.029, m04=9.6, m06=0.5)
    selection = select_metrics(
        reference,
        [reference],
        {},
        SimpleNamespace(fit_status="not_ready"),
        thresholds(),
    )
    assert selection.selected_metrics == ("m01", "m02", "m04")

    not_close = reference_with(m01=0.95, m02=0.029, m04=9.4, m06=0.5)
    selection = select_metrics(
        not_close,
        [not_close],
        {},
        SimpleNamespace(fit_status="not_ready"),
        thresholds(),
    )
    assert selection.selected_metrics == ("m01", "m02")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SelectorConfig(metric_top_k=1, metric_top_k_max=3), ("m01",)),
        (SelectorConfig(metric_top_k=2, metric_top_k_max=2), ("m01", "m02")),
        (SelectorConfig(metric_top_k=3, metric_top_k_max=3), ("m01", "m02", "m04")),
    ],
)
def test_top_k_bounds_control_expansion(config: SelectorConfig, expected: tuple[str, ...]) -> None:
    reference = reference_with(m01=0.95, m02=0.029, m04=9.6, m06=0.5)
    selection = select_metrics(
        reference,
        [reference],
        {},
        SimpleNamespace(fit_status="not_ready"),
        thresholds(),
        config=config,
    )
    assert selection.selected_metrics == expected


def test_selector_config_rejects_inconsistent_bounds() -> None:
    with pytest.raises(ValueError, match="top-k"):
        SelectorConfig(metric_top_k=3, metric_top_k_max=2)
