"""Utility functions for metrics computation."""

import numpy as np
import pandas as pd

from tabbench_bio.dataset import TaskType
from tabbench_bio.metrics.classification import ClassificationMetrics
from tabbench_bio.metrics.regression import RegressionMetrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType = TaskType.Classification,
    y_proba: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    **kwargs,
) -> dict[str, float]:
    """Compute all metrics for the given task type.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels or values.
    y_pred : np.ndarray
        Predicted labels or values.
    task_type : TaskType
        ``TaskType.Classification`` or ``TaskType.Regression``.
    y_proba : np.ndarray, optional
        Class probabilities (classification only, used for ROC-AUC).
    **kwargs
        Forwarded to :class:`ClassificationMetrics` (e.g. ``average``).

    Returns
    -------
    dict[str, float]
        Metric names mapped to values.
    """
    if task_type == TaskType.Classification:
        if isinstance(y_proba, pd.DataFrame):
            assert labels is None, "Probability columns already declare the class labels"
            probability = y_proba.copy()
            probability.columns = probability.columns.astype(str)
            assert probability.columns.is_unique, "Duplicate probability class columns"
            y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
            labels = np.sort(probability.columns.to_numpy())
            assert np.isin(y_true, labels).all(), "True labels are missing from probability columns"
            assert np.isin(y_pred, labels).all(), (
                "Predicted labels are missing from probability columns"
            )
            y_proba = probability.reindex(columns=labels).to_numpy()
        return ClassificationMetrics(**kwargs).compute_all(y_true, y_pred, y_proba, labels=labels)
    if task_type == TaskType.Regression:
        return RegressionMetrics().compute_all(y_true, y_pred)
    raise ValueError(f"Unknown task type: {task_type}")
