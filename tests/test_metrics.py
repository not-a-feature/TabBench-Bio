"""Tests for classification and regression metrics."""

import numpy as np
import pytest
from sklearn.metrics import log_loss

from tabbench_bio.dataset import TaskType
from tabbench_bio.metrics import ClassificationMetrics, RegressionMetrics, compute_metrics


class TestClassificationMetrics:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    def test_accuracy(self):
        m = ClassificationMetrics()
        assert 0.0 <= m.accuracy(self.y_true, self.y_pred) <= 1.0

    def test_f1(self):
        m = ClassificationMetrics()
        assert 0.0 <= m.f1_score(self.y_true, self.y_pred) <= 1.0

    def test_balanced_accuracy(self):
        m = ClassificationMetrics()
        assert 0.0 <= m.balanced_accuracy(self.y_true, self.y_pred) <= 1.0

    def test_compute_all_keys(self):
        m = ClassificationMetrics()
        result = m.compute_all(self.y_true, self.y_pred)
        for key in ("accuracy", "f1_score", "balanced_accuracy", "cohen_kappa"):
            assert key in result

    def test_perfect_score(self):
        m = ClassificationMetrics()
        result = m.compute_all(self.y_true, self.y_true)
        assert result["accuracy"] == pytest.approx(1.0)


@pytest.mark.filterwarnings("error:The y_prob values do not sum to one")
@pytest.mark.parametrize("classes", [2, 3])
def test_log_loss_corrects_saved_probability_rounding(classes):
    truth = np.arange(classes).repeat(2)
    probabilities = np.full((len(truth), classes), 0.1)
    probabilities[np.arange(len(truth)), truth] = 1 - (classes - 1) * 0.1
    probabilities *= 1 + np.linspace(-2.7e-7, 2.7e-7, len(truth))[:, None]
    original = probabilities.copy()
    metrics = ClassificationMetrics()
    result = metrics.compute_all(truth, truth, probabilities)
    expected = probabilities / probabilities.sum(axis=1, keepdims=True)
    assert result["log_loss"] == pytest.approx(log_loss(truth, expected))
    assert result["roc_auc"] == metrics.roc_auc(truth, original)
    assert result["f1_macro"] == 1.0
    np.testing.assert_array_equal(probabilities, original)


@pytest.mark.parametrize("row", [[0.2, 0.3], [0, 0], [-0.1, 1.1], [np.nan, 0.5], [np.inf, 0.5]])
def test_invalid_probabilities_are_not_silently_rescaled(row):
    with pytest.raises(AssertionError, match="Class probabilities"):
        ClassificationMetrics().compute_all(
            np.array([0, 1]), np.array([0, 1]), np.array([row, row])
        )


class TestRegressionMetrics:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    def test_rmse_nonneg(self):
        m = RegressionMetrics()
        assert m.rmse(self.y_true, self.y_pred) >= 0.0

    def test_r2_perfect(self):
        m = RegressionMetrics()
        assert m.r2(self.y_true, self.y_true) == pytest.approx(1.0)

    def test_compute_all_keys(self):
        m = RegressionMetrics()
        result = m.compute_all(self.y_true, self.y_pred)
        for key in ("rmse", "mae", "r2", "mse"):
            assert key in result

    def test_mape_zero_targets(self):
        m = RegressionMetrics()
        y = np.array([0.0, 0.0])
        assert m.mape(y, y) == 0.0


def test_compute_metrics_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    result = compute_metrics(y_true, y_pred, task_type=TaskType.Classification)
    assert "accuracy" in result


def test_compute_metrics_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    result = compute_metrics(y_true, y_pred, task_type=TaskType.Regression)
    assert "rmse" in result
