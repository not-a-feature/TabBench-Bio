"""Rebuild benchmark metrics from transactional result bundles.

Metrics CSV files are derived outputs. Every run replaces them from the current passing
attempts in the result database, so stale rows cannot survive a retry or consolidation.

Usage
-----
::

    from tabbench_bio.config import load_config
    from tabbench_bio.evaluation import compute_metrics_from_predictions

    config = load_config("configs/benchmark_v0.1.json")
    compute_metrics_from_predictions(config)
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from tabbench_bio.benchmark import configure_benchmark
from tabbench_bio.bio.datasets import BIO_DATASETS
from tabbench_bio.dataset import TaskType
from tabbench_bio.io_utils import atomic_to_csv
from tabbench_bio.metrics import compute_metrics
from tabbench_bio.result_store import (
    ResultRepository,
    result_location,
)

logger = logging.getLogger(__name__)


def _compute_metrics_from_store(config, repository: ResultRepository) -> None:
    """Rebuild cell metrics from transactional prediction artifacts."""
    output_dir = config["output_dir"]
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    excluded_keys = _build_excluded_keys(config)
    benchmark = configure_benchmark(config, init_benchmark=False)
    regression_names = set(benchmark.dataset_names_regression)
    rows = {
        ("classification", False): [],
        ("regression", False): [],
        ("classification", True): [],
        ("regression", True): [],
    }

    for attempt in repository.current_attempts(cell=repository.cell):
        if attempt.status != "pass":
            continue
        data_test = repository.dataframe(attempt, "ground_truth").sort_index()
        y_pred = repository.dataframe(attempt, "prediction").sort_index()
        assert np.array_equal(data_test.index, y_pred.index), attempt.key
        dataset_name, target_idx = benchmark.split_key(attempt.dataset)
        if dataset_name in BIO_DATASETS and not BIO_DATASETS[dataset_name].enabled:
            continue
        task_type = (
            TaskType.Regression if dataset_name in regression_names else TaskType.Classification
        )
        y_proba = None
        if task_type == TaskType.Classification:
            probability = repository.dataframe(attempt, "probability")
            assert probability is not None, (
                f"Passing classification unit lacks probabilities: {attempt.key}"
            )
            probability = probability.sort_index()
            assert np.array_equal(data_test.index, probability.index), attempt.key
            wanted = [str(value) for value in np.unique(data_test["target"])]
            probability.columns = [str(column) for column in probability.columns]
            y_proba = (
                probability.reindex(columns=wanted).to_numpy()
                if set(wanted).issubset(probability.columns)
                else probability.to_numpy()
            )
        row = {
            "seed": attempt.seed,
            "key": attempt.dataset,
            "dataset": dataset_name,
            "task_type": task_type,
            "target_idx": target_idx,
            "model": attempt.model,
        }
        row.update(
            compute_metrics(data_test["target"], y_pred["target"], task_type, y_proba=y_proba)
        )
        task = "regression" if task_type == TaskType.Regression else "classification"
        rows[(task, attempt.dataset in excluded_keys)].append(row)

    outputs = {
        ("classification", False): "classification_metrics.csv",
        ("regression", False): "regression_metrics.csv",
        ("classification", True): "excluded_classification_metrics.csv",
        ("regression", True): "excluded_regression_metrics.csv",
    }
    identity_columns = ["seed", "key", "dataset", "task_type", "target_idx", "model"]
    for key, filename in outputs.items():
        frame = pd.DataFrame(rows[key]) if rows[key] else pd.DataFrame(columns=identity_columns)
        path = os.path.join(metrics_dir, filename)
        atomic_to_csv(frame, path, index=False)
        _write_summary(frame, path.replace(".csv", "_summary.csv"))
        logger.info("Wrote %d database-derived metric rows to %s.", len(frame), path)


def _build_excluded_keys(config) -> set[str]:
    """Return the set of dataset keys to exclude based on config.

    Three mechanisms (all applied in combination):

    * ``exclude_keys``    — exact keys (e.g. ``"sugar_mixtures_high_snr_4"``)
    * ``exclude_datasets``— all keys for a dataset (e.g. ``"timegate_fermentation"``)
    * ``exclude_targets`` — keys whose target name matches (e.g. ``"time_h"``)
    """
    excluded: set[str] = set(config["exclude_keys"])

    exclude_datasets = set(config["exclude_datasets"])
    exclude_names = set(config["exclude_targets"])

    if exclude_datasets or exclude_names:
        stats_path = os.path.join(config["output_dir"], "dataset_stats.json")
        if not os.path.exists(stats_path):
            logger.warning(
                "exclude_datasets/exclude_targets set but dataset_stats.json not found "
                "— dataset/name-based exclusions skipped."
            )
        else:
            with open(stats_path) as f:
                stats = json.load(f)
            for ds_id, s in stats.items():
                if ds_id in exclude_datasets:
                    n_targets = len((s or {}).get("target_names") or []) or 1
                    for idx in range(n_targets):
                        excluded.add(f"{ds_id}_{idx}")
                if exclude_names and s and s.get("target_names"):
                    for idx, name in enumerate(s["target_names"]):
                        if name in exclude_names:
                            excluded.add(f"{ds_id}_{idx}")

    return excluded


def compute_metrics_from_predictions(config):
    """Rebuild all metrics for one result cell from the result database.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration.
    """
    logger.info("=" * 60 + "\nSTEP 2: Computing Metrics")
    output_dir = config["output_dir"]
    root, cell = result_location(output_dir)
    repository = ResultRepository.from_root(root, cell=cell)
    assert repository.bundle_paths(), f"No result database found under {root}"
    _compute_metrics_from_store(config, repository)


def _write_summary(df: pd.DataFrame, path: str):
    """Write mean ± std per (key, model) across seeds."""
    if df.empty:
        atomic_to_csv(pd.DataFrame(), path, index=False)
        return
    group_cols = [
        c for c in ["key", "dataset", "task_type", "target_idx", "model"] if c in df.columns
    ]
    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in ("seed", "target_idx")
    ]
    agg = {col: ["mean", "std"] for col in numeric}
    summary = df.groupby(group_cols).agg(agg)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    atomic_to_csv(summary.reset_index(), path, index=False)
