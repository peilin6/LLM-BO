import numpy as np
import pytest

from dibo.optimizer import metric_distance, metric_loss, monte_carlo_ei
from dibo.schemas import MetricTarget


@pytest.mark.parametrize(
    ("target", "values", "expected"),
    [
        (MetricTarget(kind="upper", value=10.0, scale=2.0), [8.0, 10.0, 14.0], [0.0, 0.0, 2.0]),
        (MetricTarget(kind="lower", value=10.0, scale=2.0), [6.0, 10.0, 12.0], [2.0, 0.0, 0.0]),
        (
            MetricTarget(kind="interval", lower=8.0, upper=12.0, scale=2.0),
            [6.0, 10.0, 14.0],
            [1.0, 0.0, 1.0],
        ),
        (MetricTarget(kind="point", value=10.0, scale=2.0), [6.0, 10.0, 14.0], [2.0, 0.0, 2.0]),
    ],
)
def test_four_normalized_metric_distances(
    target: MetricTarget, values: list[float], expected: list[float]
) -> None:
    np.testing.assert_allclose(metric_distance(values, target), expected)


def test_metric_loss_is_weighted_in_real_metric_units() -> None:
    loss = metric_loss(
        {"m01": [12.0, 8.0], "m06": [0.4, 0.8]},
        {
            "m01": MetricTarget(kind="upper", value=10.0, scale=2.0),
            "m06": MetricTarget(kind="point", value=0.5, scale=0.1),
        },
        {"m01": 0.25, "m06": 0.75},
    )

    np.testing.assert_allclose(loss, [1.0, 2.25])


def test_monte_carlo_ei_matches_hand_calculation() -> None:
    assert monte_carlo_ei([1.0, 2.0, 4.0, 5.0], incumbent_loss=3.0) == pytest.approx(0.75)
