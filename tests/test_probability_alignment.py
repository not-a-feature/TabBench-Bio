import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from tabbench_bio.evaluation import _compute_metrics_from_store
from tabbench_bio.leaderboard import _metrics_from_sqlite
from tabbench_bio.metrics import ClassificationMetrics, compute_metrics
from tabbench_bio.result_store import ResultRepository


@pytest.mark.parametrize("truth", [[0, 2, 0, 2], [0, 0, 0, 0], [0, 1, 2, 0]])
def test_full_probability_columns_survive_missing_fold_classes(truth):
    probability = np.full((4, 3), 0.15)
    probability[np.arange(4), truth] = 0.7
    frame = pd.DataFrame(probability, columns=[0, 1, 2])[[2, 0, 1]]
    metrics = compute_metrics(np.array(truth), np.array(truth), y_proba=frame)
    assert metrics["log_loss"] == pytest.approx(log_loss(truth, probability, labels=[0, 1, 2]))
    if len(set(truth)) < 3:
        assert np.isnan(metrics["roc_auc"])
    else:
        assert metrics["roc_auc"] == 1


def test_ambiguous_matrix_and_missing_columns_fail():
    with pytest.raises(AssertionError, match="full class labels"):
        ClassificationMetrics().roc_auc([0, 2], np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]))
    with pytest.raises(AssertionError, match="missing from probability columns"):
        compute_metrics([0, 2], [0, 2], y_proba=pd.DataFrame({0: [0.8, 0.2], 1: [0.2, 0.8]}))


def test_both_result_readers_keep_the_full_class_space(debug_config):
    repository = ResultRepository(debug_config["output_dir"], debug_config)
    truth = pd.DataFrame({"target": [0, 2, 0, 2]})
    probability = pd.DataFrame(
        {"2": [0.1, 0.7, 0.1, 0.7], "0": [0.7, 0.1, 0.7, 0.1], "1": [0.2] * 4}
    )
    repository.write(
        {"dataset": "OpenML-1138_0", "model": "RF", "status": "pass"},
        seed=42,
        ground_truth=truth,
        prediction=truth,
        probability=probability,
    )
    _compute_metrics_from_store(debug_config, repository)
    frames = list((Path(debug_config["output_dir"]) / "metrics").glob("*.csv"))
    assert frames
    with sqlite3.connect(repository.writer_path) as connection:
        _, classification, _ = _metrics_from_sqlite(connection, repository.cell)
    assert classification.iloc[0].log_loss == pytest.approx(-np.log(0.7))
    assert np.isnan(classification.iloc[0].roc_auc)
