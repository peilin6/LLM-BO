from __future__ import annotations

import numpy as np
import pytest

from dibo.models import fit_g, posterior
from tests.unit.test_f_models import training_results


def test_g_uses_complete_six_metric_rows_and_predicts_shapes() -> None:
    results = training_results()
    metrics = dict(results[0].metrics)
    metrics["m05"] = None
    results[0] = results[0].model_copy(update={"metrics": metrics})

    model = fit_g(results, min_train_samples=5)

    assert model.fit_status == "ready"
    assert model.n_samples == 5
    assert len(model.feature_order) == 6
    prediction = posterior(model, np.zeros((3, 6)))
    assert prediction.mean.shape == (3,)
    assert prediction.variance.shape == (3,)


def test_g_rejects_wrong_posterior_shape_and_reports_small_data() -> None:
    model = fit_g(training_results())
    with pytest.raises(ValueError, match="expects 6 features"):
        posterior(model, np.zeros((1, 5)))

    small = fit_g(training_results()[:2])
    assert small.fit_status == "not_ready"
    with pytest.raises(RuntimeError, match="not ready"):
        posterior(small, np.zeros((1, 6)))