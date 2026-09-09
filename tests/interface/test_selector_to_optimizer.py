from pathlib import Path
from types import SimpleNamespace

import pytest

from dibo.optimizer import suggest
from dibo.schemas import MetricSelection, MetricTarget, load_actions, load_graph
from dibo.selector import union_neighbor_actions
from tests.unit.test_initial_design import BASE_CONFIG
from tests.unit.test_optimizer import optimizer_config

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize(
    ("metrics", "expected_dimension"),
    [
        (("m01", "m02"), 4),
        (("m03", "m06"), 6),
        (("m01", "m03", "m06"), 8),
    ],
)
def test_real_graph_union_drives_optimizer_dimension(
    metrics: tuple[str, ...], expected_dimension: int
) -> None:
    actions = union_neighbor_actions(metrics, load_graph(CONFIGS / "graph.yaml"))
    weight = 1.0 / len(metrics)
    selection = MetricSelection(
        reference_trial_id="trial_base",
        mode="rotation",
        selected_metrics=metrics,
        scores={metric_id: 0.0 for metric_id in metrics},
        weights={metric_id: weight for metric_id in metrics},
        targets={
            metric_id: MetricTarget(kind="upper", value=1.0, scale=1.0)
            for metric_id in metrics
        },
    )

    result = suggest(
        selection,
        actions,
        {},
        SimpleNamespace(fit_status="not_ready"),
        BASE_CONFIG,
        load_actions(CONFIGS / "actions_v1.yaml"),
        [],
        config=optimizer_config(seed=expected_dimension),
    )

    assert result is not None
    assert result.selected_actions == actions
    assert result.bo_dimension == expected_dimension
    assert all(value == 0.0 for action, value in result.coefficients.items() if action not in actions)