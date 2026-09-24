"""Leaderboard utilities for ranking models from benchmark results.

The :class:`Leaderboard` class normalises per-(seed, dataset, model) metrics into a
ranked leaderboard.  Build one from a results directory produced by the pipeline, or
evaluate a new scikit-learn-compatible model in-process and add it.

Typical workflow
----------------
::

    from tabbench_bio import Leaderboard
    from sklearn.ensemble import RandomForestClassifier

    # 1. Load metrics from a results directory
    lb = Leaderboard.from_results_dir("results/bio_classification")

    # 2. Print current ranking
    print(lb.rank())

    # 3. Evaluate your model and add it to the leaderboard
    results = lb.evaluate_and_add("My-RF", RandomForestClassifier(), "config.json",
                                  task="classification")
    print(lb.rank())

    # 4. Visualise
    lb.plot()
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sqlite3
import time
import zlib
from collections import deque
from concurrent.futures import Executor
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from tqdm.auto import tqdm

from tabbench_bio.coverage import complete_folds, impute_failures, load_status
from tabbench_bio.dataset import TaskType
from tabbench_bio.elo import compute_elo, fold_scores
from tabbench_bio.metric_cache import MetricCache
from tabbench_bio.metrics import PRIMARY_CLF_METRIC, PRIMARY_REG_METRIC, compute_metrics
from tabbench_bio.seeds import get_seeds

logger = logging.getLogger(__name__)

_RANK_COL = "Rank"
_SQLITE_SCHEMA_VERSION = 1

# Columns from display metadata kept verbatim (not recomputed from raw metrics)
_META_COLS = [
    "Model",
    "Category",
    "Train Time s",
    "Infer. s/1K",
    "# Failed",
    "# Skipped",
]

_CURRENT_SQLITE_ATTEMPTS = """
WITH ranked_attempts AS (
    SELECT
        seed,
        dataset,
        model,
        status,
        reason,
        prediction_sha256,
        probability_sha256,
        ground_truth_sha256,
        record_json,
        ROW_NUMBER() OVER (
            PARTITION BY cell, seed, dataset, model
            ORDER BY
                CASE WHEN status = 'skip' AND reason = 'benchmark_exclusion' THEN 2
                     WHEN status = 'pass' THEN 1 ELSE 0 END DESC,
                timestamp DESC,
                attempt_id DESC
        ) AS current_rank
    FROM attempts
    WHERE cell = ?
)
SELECT
    seed,
    dataset,
    model,
    status,
    reason,
    prediction_sha256,
    probability_sha256,
    ground_truth_sha256,
    record_json
FROM ranked_attempts
WHERE current_rank = 1
ORDER BY seed, dataset, model
"""


def _open_results_sqlite(path: Path) -> sqlite3.Connection:
    """Open one result bundle without granting SQLite write access."""
    assert path.is_file(), path
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=60)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _validate_results_sqlite(connection: sqlite3.Connection, path: Path) -> None:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    required = {"metadata", "cells", "blobs", "attempts"}
    missing = sorted(required - tables)
    if missing:
        raise ValueError(f"Not a TabBench Bio result database ({path}); missing: {missing}")
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    schema_version = int(metadata["schema_version"])
    if schema_version != _SQLITE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported result schema {schema_version}; expected {_SQLITE_SCHEMA_VERSION}"
        )


def _sqlite_frame(
    connection: sqlite3.Connection,
    digest: str,
) -> pd.DataFrame:
    row = connection.execute(
        "SELECT encoding, uncompressed_bytes, payload FROM blobs WHERE sha256 = ?",
        (digest,),
    ).fetchone()
    assert row is not None, f"Missing result blob {digest}"
    encoding, uncompressed_bytes, compressed = row
    assert encoding == "zlib", (digest, encoding)
    payload = zlib.decompress(bytes(compressed))
    assert len(payload) == int(uncompressed_bytes), digest
    assert hashlib.sha256(payload).hexdigest() == digest, digest
    return pd.read_csv(io.BytesIO(payload), index_col=0)


def _compute_metric_batch(batch):
    regression, classification = [], []
    for identity, task_type, truth, prediction, probability in batch:
        row = {**identity, **compute_metrics(truth, prediction, task_type, y_proba=probability)}
        (classification if task_type == TaskType.Classification else regression).append(row)
    return regression, classification


def _metrics_from_sqlite(
    connection: sqlite3.Connection,
    cell: str,
    *,
    executor: Executor | None = None,
    max_pending: int = 2,
    show_progress: bool = False,
    cache: MetricCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    ground_truth_cache: dict[str, pd.DataFrame] = {}
    assert max_pending > 0
    attempts = list(connection.execute(_CURRENT_SQLITE_ATTEMPTS, (cell,)))
    progress = tqdm(
        total=sum(row[3] == "pass" for row in attempts),
        desc=f"Metrics {cell}",
        unit="fold",
        mininterval=1,
        disable=not show_progress,
    )
    batch, pending = [], deque()
    cache_keys = {}
    cache_hits = 0

    def collect(result):
        regression, classification = result
        if cache is not None:
            cache.write(
                [
                    (
                        cache_keys.pop((row["seed"], row["key"], row["model"])),
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("seed", "key", "model")
                        },
                    )
                    for row in [*regression, *classification]
                ]
            )
        regression_rows.extend(regression)
        classification_rows.extend(classification)
        progress.update(len(regression) + len(classification))

    for row in attempts:
        (
            seed,
            dataset,
            model,
            status,
            reason,
            prediction_sha256,
            probability_sha256,
            ground_truth_sha256,
            record_json,
        ) = row
        status_rows.append(
            {
                **json.loads(record_json),
                "ground_truth_sha256": ground_truth_sha256,
                "seed": int(seed),
                "key": str(dataset),
                "model": str(model),
                "status": str(status),
                "reason": str(reason),
            }
        )
        if status != "pass":
            continue
        assert prediction_sha256 is not None, (cell, seed, dataset, model)
        assert ground_truth_sha256 is not None, (cell, seed, dataset, model)
        metric_row: dict[str, object] = {
            "seed": int(seed),
            "key": str(dataset),
            "model": str(model),
        }
        task_type = (
            TaskType.Classification if probability_sha256 is not None else TaskType.Regression
        )
        if cache is not None:
            cache_key = cache.key(prediction_sha256, probability_sha256, ground_truth_sha256)
            cached = cache.read(cache_key)
            if cached is not None:
                metric_row.update(cached)
                (
                    classification_rows if task_type == TaskType.Classification else regression_rows
                ).append(metric_row)
                cache_hits += 1
                progress.update(1)
                continue
            cache_keys[(int(seed), str(dataset), str(model))] = cache_key
        ground_truth_digest = str(ground_truth_sha256)
        if ground_truth_digest not in ground_truth_cache:
            ground_truth_cache[ground_truth_digest] = _sqlite_frame(
                connection, ground_truth_digest
            ).sort_index()
        data_test = ground_truth_cache[ground_truth_digest]
        y_pred = _sqlite_frame(connection, str(prediction_sha256)).sort_index()
        assert np.array_equal(data_test.index, y_pred.index), (cell, seed, dataset, model)

        y_proba = None
        if probability_sha256 is not None:
            probability = _sqlite_frame(connection, str(probability_sha256)).sort_index()
            assert np.array_equal(data_test.index, probability.index), (cell, seed, dataset, model)
            probability.columns = probability.columns.astype(str)
            wanted = [str(value) for value in np.unique(data_test["target"])]
            y_proba = (
                probability.reindex(columns=wanted).to_numpy()
                if set(wanted).issubset(probability.columns)
                else probability.to_numpy()
            )
        batch.append(
            (
                metric_row,
                task_type,
                data_test["target"].to_numpy(),
                y_pred["target"].to_numpy(),
                y_proba,
            )
        )
        if len(batch) == 16:
            if executor is None:
                collect(_compute_metric_batch(batch))
            else:
                pending.append(executor.submit(_compute_metric_batch, batch))
                if len(pending) >= max_pending:
                    collect(pending.popleft().result())
            batch = []
    if batch:
        if executor is None:
            collect(_compute_metric_batch(batch))
        else:
            pending.append(executor.submit(_compute_metric_batch, batch))
    for future in pending:
        collect(future.result())
    progress.close()
    if cache is not None and show_progress:
        total = len(regression_rows) + len(classification_rows)
        print(
            f"Fold-metric cache: reused {cache_hits}/{total}; computed {total - cache_hits}",
            flush=True,
        )
    for rows in (regression_rows, classification_rows):
        rows.sort(key=lambda row: (row["seed"], row["key"], row["model"]))

    status_frame = (
        pd.DataFrame(status_rows)
        if status_rows
        else pd.DataFrame(columns=["seed", "key", "model", "status", "reason"])
    )
    return (
        pd.DataFrame(regression_rows),
        pd.DataFrame(classification_rows),
        status_frame,
    )


class Leaderboard:
    """Manage and extend the TabBench Bio leaderboard.

    Parameters
    ----------
    reg_metrics : pd.DataFrame
        Raw per-(seed, key, model) regression metrics.  Must contain columns
        ``seed``, ``key``, ``model``, ``rmse`` (and optionally ``mse``,
        ``mae``, ``r2``, …).
    clf_metrics : pd.DataFrame
        Raw per-(seed, key, model) classification metrics.  Must contain
        columns ``seed``, ``key``, ``model``, ``f1_macro`` (the primary
        ranking metric; and optionally ``matthews_corrcoef``, ``f1_macro``,
        ``f1_score``, ``roc_auc``, …).
    display_meta : pd.DataFrame
        Per-model display metadata indexed by ``model_id``.  Columns:
        ``model_id``, ``Model``, ``Category``, ``Elo``,
        ``Train Time s``, ``Infer. s/1K``.  Missing models (e.g. newly
        added) are filled with sensible defaults.

    Notes
    -----
    Use the class method :meth:`from_results_dir` to construct instances from a
    pipeline results directory — do not call ``__init__`` directly.
    """

    def __init__(
        self,
        reg_metrics: pd.DataFrame,
        clf_metrics: pd.DataFrame,
        display_meta: pd.DataFrame | None = None,
    ):
        self._reg_metrics = reg_metrics.copy()
        self._clf_metrics = clf_metrics.copy()
        self._display_meta = display_meta.copy() if display_meta is not None else pd.DataFrame()
        self._added_models: list[str] = []
        self._reference_sqlite: Path | None = None
        self._reference_config: dict = {}
        self._reference_truth: dict[tuple[int, str], str] = {}
        self._expected_seeds: set[int] = set()

        # Populated by _rebuild()
        self._overall: pd.DataFrame = pd.DataFrame()
        self._clf: pd.DataFrame = pd.DataFrame()
        self._reg: pd.DataFrame = pd.DataFrame()
        self._rebuild()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_results_dir(cls, results_dir: str) -> Leaderboard:
        """Load leaderboard from a local results directory.

        Expects ``metrics/classification_metrics.csv`` and
        ``metrics/regression_metrics.csv`` inside *results_dir*.

        Parameters
        ----------
        results_dir : str
            Path produced by running the benchmark pipeline
            (``scripts/feature_sweep.py``).

        Returns
        -------
        Leaderboard
        """
        metrics_dir = os.path.join(results_dir, "metrics")
        clf_path = os.path.join(metrics_dir, "classification_metrics.csv")
        reg_path = os.path.join(metrics_dir, "regression_metrics.csv")

        reg_df = pd.read_csv(reg_path) if os.path.exists(reg_path) else pd.DataFrame()
        clf_df = pd.read_csv(clf_path) if os.path.exists(clf_path) else pd.DataFrame()

        # Score failed fits at chance rather than omitting them, so a model cannot improve
        # its standing by crashing on the targets it finds hard.
        status = load_status(results_dir)
        with open(os.path.join(results_dir, "config.json")) as f:
            seeds = get_seeds(json.load(f))
        leaderboard = cls(
            complete_folds(impute_failures(reg_df, status), status, seeds),
            complete_folds(impute_failures(clf_df, status), status, seeds),
        )
        leaderboard._expected_seeds = set(seeds)
        return leaderboard

    @staticmethod
    def sqlite_cells(sqlite_path: str | os.PathLike[str]) -> list[str]:
        """Return the result cells available in a published SQLite bundle."""
        path = Path(sqlite_path).resolve()
        with closing(_open_results_sqlite(path)) as connection:
            _validate_results_sqlite(connection, path)
            return [
                str(row[0]) for row in connection.execute("SELECT cell FROM cells ORDER BY cell")
            ]

    @classmethod
    def from_sqlite(
        cls,
        sqlite_path: str | os.PathLike[str],
        *,
        cell: str | None = None,
    ) -> Leaderboard:
        """Load a leaderboard directly from a published result database.

        The database is opened with SQLite's ``mode=ro`` flag. No writer bundle,
        metrics CSV, cache, or sidecar file is created. Published bundles can contain
        several feature/sample cells; pass ``cell=...`` in that case. Use
        :meth:`sqlite_cells` to list the available values.

        Ranking metrics are recomputed from the content-addressed ground-truth and
        prediction blobs in the bundle. This deliberately follows the benchmark's
        source-of-truth model: metrics CSV files are derived artifacts, not database
        tables.
        """
        path = Path(sqlite_path).resolve()
        with closing(_open_results_sqlite(path)) as connection:
            _validate_results_sqlite(connection, path)
            cells = [
                str(row[0]) for row in connection.execute("SELECT cell FROM cells ORDER BY cell")
            ]
            if cell is None:
                if len(cells) != 1:
                    choices = ", ".join(cells)
                    raise ValueError(
                        f"Result database contains {len(cells)} cells; choose one with "
                        f"cell=... Available cells: {choices}"
                    )
                selected_cell = cells[0]
            else:
                if cell not in cells:
                    choices = ", ".join(cells)
                    raise ValueError(f"Unknown result cell {cell!r}. Available cells: {choices}")
                selected_cell = cell
            reg_df, clf_df, status = _metrics_from_sqlite(connection, selected_cell)
            (config_json,) = connection.execute(
                "SELECT config_json FROM cells WHERE cell = ?", (selected_cell,)
            ).fetchone()
        seeds = get_seeds(json.loads(config_json))
        leaderboard = cls(
            complete_folds(impute_failures(reg_df, status), status, seeds),
            complete_folds(impute_failures(clf_df, status), status, seeds),
        )
        leaderboard._reference_sqlite = path
        leaderboard._reference_config = json.loads(config_json)
        leaderboard._expected_seeds = set(seeds)
        with closing(_open_results_sqlite(path)) as connection:
            for row in connection.execute(_CURRENT_SQLITE_ATTEMPTS, (selected_cell,)):
                if row[7] is not None:
                    key = (int(row[0]), str(row[1]))
                    if key in leaderboard._reference_truth:
                        assert leaderboard._reference_truth[key] == row[7], key
                    leaderboard._reference_truth[key] = str(row[7])
        return leaderboard

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self, task: str = "overall") -> pd.DataFrame:
        """Return a ranked leaderboard DataFrame.

        Parameters
        ----------
        task : {"overall", "classification", "regression"}
            Which leaderboard to return.

        Returns
        -------
        pd.DataFrame
            Sorted by fold-level Elo (descending), with target-bootstrap intervals.
            Without the RF anchor, Elo and Rank are missing; Score stays descriptive.
        """
        df = self._select_leaderboard(task).copy()
        if task not in self._ratings:
            self._ratings[task] = compute_elo(
                fold_scores(
                    self._clf_metrics if task != "regression" else None,
                    self._reg_metrics if task != "classification" else None,
                )
            )
        df = df.merge(self._ratings[task].drop(columns="n_targets"), on="model_id", how="left")
        df = df.sort_values(
            ["Elo", "model_id"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
        df[_RANK_COL] = df["Elo"].rank(method="min", ascending=False).astype("Int64")
        cols = [_RANK_COL] + [c for c in df.columns if c != _RANK_COL]
        return df[cols]

    def summary(self, task: str = "overall") -> str:
        """Return a human-readable summary of the current leaderboard."""
        df = self.rank(task)
        lines = [f"TabBench Bio — {task} fold-level Elo (RF = 1000)", "=" * 72]
        for _, row in df.iterrows():
            model = row.get("Model", row.get("model_id", "?"))
            score = row.get("Score", float("nan"))
            elo = row.get("Elo", float("nan"))
            if pd.isna(elo):
                lines.append(f"   —  {model:<28}  Elo unavailable  Score={score:.3f}")
            else:
                lines.append(
                    f"  #{int(row[_RANK_COL]):2d}  {model:<28}  Elo={elo:.0f}  "
                    f"95% CI [{row['Elo_lo']:.0f}, {row['Elo_hi']:.0f}]  "
                    f"Targets={int(row['# Targets'])}"
                )
        if df["Elo"].isna().all():
            lines.append("No anchored Elo ranking: include RF and at least one comparable model.")
        if self._added_models:
            lines.append(f"\nAdded models: {', '.join(self._added_models)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Adding new models
    # ------------------------------------------------------------------

    def add_results(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
    ) -> None:
        """Add raw per-(seed, key) metrics for a new model to the leaderboard.

        The metrics are stored in the same format as the paper's CSV files and
        the leaderboard scores are recomputed from all models combined
        (including the new one), so the normalization is updated automatically.

        Parameters
        ----------
        model_name : str
            Display name and internal identifier for the new model.
        metrics_df : pd.DataFrame
            DataFrame with one row per (seed, dataset_key) containing metric
            columns.  Must include ``seed`` and ``key``.  Regression rows need
            ``rmse``; classification rows need ``f1_macro``.  A ``model``
            column is added automatically.
        """
        df = metrics_df.copy()
        df["model"] = model_name
        assert not df.duplicated(["seed", "key"]).any(), "Duplicate model/target/fold results"
        for frame in (self._reg_metrics, self._clf_metrics):
            if not frame.empty:
                assert model_name not in set(frame["model"]), f"Model already present: {model_name}"
        if self._expected_seeds:
            assert all(
                folds == self._expected_seeds for folds in df.groupby("key")["seed"].agg(set)
            ), "New models must provide every configured fold for each evaluated target"

        reg_cols = {"rmse", "mse", "mae", "r2"}
        clf_cols = {
            "f1_score",
            "f1_macro",
            "balanced_accuracy",
            "matthews_corrcoef",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
        }

        if reg_cols & set(df.columns):
            keep = ["seed", "key", "model"] + [c for c in df.columns if c in reg_cols]
            self._reg_metrics = pd.concat(
                [self._reg_metrics, df[keep].dropna(subset=[PRIMARY_REG_METRIC])],
                ignore_index=True,
            )

        if clf_cols & set(df.columns):
            keep = ["seed", "key", "model"] + [c for c in df.columns if c in clf_cols]
            self._clf_metrics = pd.concat(
                [self._clf_metrics, df[keep].dropna(subset=[PRIMARY_CLF_METRIC])],
                ignore_index=True,
            )

        if model_name not in self._added_models:
            self._added_models.append(model_name)

        self._rebuild()
        logger.info("Added model '%s' to leaderboard.", model_name)

    def evaluate_and_add(
        self,
        model_name: str,
        model: Any,
        config_path: str,
        seeds: int | None = None,
        task: str = "overall",
    ) -> pd.DataFrame:
        """Run a model through the full benchmark and add it to the leaderboard.

        This is a convenience wrapper that:

        1. Loads all benchmark datasets via :class:`~tabbench_bio.benchmark.TabBenchBio`.
        2. Fits and evaluates *model* on each train/test split.
        3. Computes metrics.
        4. Calls :meth:`add_results` to insert the model into the leaderboard.

        *model* must expose a scikit-learn-compatible API:
        ``fit(X, y)`` and ``predict(X)``.

        Parameters
        ----------
        model_name : str
            Display name for the leaderboard.
        model : object
            A scikit-learn-compatible estimator.
        config_path : str
            Path to a benchmark config JSON describing the datasets to run on.
        seeds : int | None
            Defaults to every configured fold. An explicit count must match the config.
        task : str
            Filter to only regression or classification datasets when set to
            ``"regression"`` or ``"classification"``; default ``"overall"``
            runs both.

        Returns
        -------
        pd.DataFrame
            Per-(seed, key) metrics for the newly evaluated model, in the same
            format as the raw metrics CSVs.
        """
        from tabbench_bio.benchmark import configure_benchmark
        from tabbench_bio.config import load_config

        config = load_config(config_path)
        scheduled = get_seeds(config)
        if self._expected_seeds:
            assert set(scheduled) == self._expected_seeds, (
                "Evaluation folds differ from the baseline"
            )
        if seeds is not None and seeds != len(scheduled):
            raise ValueError(f"Use all {len(scheduled)} configured folds, not {seeds}.")
        if task not in {"overall", "classification", "regression"}:
            raise ValueError(f"Unknown task {task!r}")
        for field in (
            "cv_folds",
            "bio_max_features",
            "train_subsample",
            "min_samples_per_class",
            "max_classes",
            "group_regression_splits",
        ):
            if field in self._reference_config:
                assert config[field] == self._reference_config[field], (
                    f"Reference config mismatch: {field}"
                )
        if task == "classification":
            config["dataset_names_regression"] = []
        elif task == "regression":
            config["dataset_names_classification"] = []
        if config["dataset_names_classification"] and not is_classifier(model):
            raise ValueError(
                "Classification requires a scikit-learn classifier; set task='regression' for a regressor."
            )
        if config["dataset_names_regression"] and not is_regressor(model):
            raise ValueError(
                "Regression requires a scikit-learn regressor; set task='classification' for a classifier."
            )

        records = []
        fit_times: list[float] = []
        infer_s_per_1k: list[float] = []

        for seed in scheduled:
            config["random_state"] = seed
            bench = configure_benchmark(config)
            for train_df, test_df, key, task_type in bench:
                if train_df is None:
                    continue
                if task == "regression" and task_type != TaskType.Regression:
                    continue
                if task == "classification" and task_type != TaskType.Classification:
                    continue
                label_col = train_df.columns[-1]
                X_train = train_df.drop(columns=[label_col]).values
                y_train = train_df[label_col].values
                X_test = test_df.drop(columns=[label_col]).values
                y_test = test_df[label_col].values

                if self._reference_sqlite is not None:
                    unit = (seed, key)
                    assert unit in self._reference_truth, f"No reference test split for {unit}"
                    with closing(_open_results_sqlite(self._reference_sqlite)) as connection:
                        truth = _sqlite_frame(connection, self._reference_truth[unit]).sort_index()
                    observed_truth = test_df[[label_col]].sort_index()
                    # CSV storage parses numeric class names as numbers; compare their labels.
                    if task_type == TaskType.Classification:
                        observed_truth = observed_truth.astype(str)
                        truth = truth.astype(str)
                    pd.testing.assert_frame_equal(
                        observed_truth,
                        truth,
                        check_dtype=False,
                        obj=f"Reference test split {unit}",
                    )
                estimator = clone(model)
                t0 = time.perf_counter()
                estimator.fit(X_train, y_train)
                fit_s = time.perf_counter() - t0
                fit_times.append(fit_s)
                t1 = time.perf_counter()
                y_pred = estimator.predict(X_test)
                infer_s = time.perf_counter() - t1
                infer_s_per_1k.append((infer_s / len(X_test)) * 1000)
                metrics = compute_metrics(y_test, y_pred, task_type=task_type)
                records.append(
                    {
                        "seed": seed,
                        "key": key,
                        "model": model_name,
                        "task_type": task_type.name,
                        "train_time_s": fit_s,
                        "inference_time_s": infer_s,
                        **metrics,
                    }
                )

        metrics_df = pd.DataFrame(records)
        assert not metrics_df.empty, "No model evaluations completed"
        assert all(
            folds == set(scheduled) for folds in metrics_df.groupby("key")["seed"].agg(set)
        ), "Custom evaluation must complete every configured fold for each target"
        mean_train_s = float(np.mean(fit_times)) if fit_times else float("nan")
        mean_infer_s_per_1k = float(np.mean(infer_s_per_1k)) if infer_s_per_1k else float("nan")
        self._upsert_display_meta(
            model_name,
            {
                "Train Time s": mean_train_s,
                "Infer. s/1K": mean_infer_s_per_1k,
            },
        )
        self.add_results(model_name, metrics_df)
        return metrics_df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        task: str = "overall",
        n_top: int = 30,
        figsize: tuple = (10, 8),
    ):
        """Plot fold-level Elo with target-bootstrap 95% confidence intervals.

        Parameters
        ----------
        task : {"overall", "classification", "regression"}
            Which leaderboard to visualise.
        n_top : int
            Show only the top *n_top* models.
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch

        df = self.rank(task).dropna(subset=["Elo"]).head(n_top).iloc[::-1]
        if df.empty:
            raise ValueError(
                "No anchored Elo to plot. Include RF and at least one comparable model."
            )
        model_col = "Model" if "Model" in df.columns else "model_id"
        models = df[model_col].tolist()
        scores = df["Elo"].to_numpy(dtype=float)
        colors = ["#d97706" if m in self._added_models else "#2563eb" for m in df["model_id"]]
        fig = Figure(figsize=figsize, facecolor="white")
        ax = fig.subplots()
        y = np.arange(len(models))
        ax.errorbar(
            scores,
            y,
            xerr=np.vstack([scores - df["Elo_lo"], df["Elo_hi"] - scores]),
            fmt="none",
            ecolor="#64748b",
            capsize=3,
            linewidth=1.3,
            zorder=2,
        )
        ax.scatter(scores, y, c=colors, s=60, zorder=3)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11, color="#111827")
        ax.set_xlabel("Fold-level Bradley–Terry Elo · Random Forest = 1000", fontsize=11)
        ax.set_title("TabBench-Bio", loc="left", fontsize=19, fontweight="bold", pad=22)
        ax.text(
            0,
            1.015,
            f"{task.title()} · 95% target-bootstrap intervals",
            transform=ax.transAxes,
            fontsize=10,
            color="#475569",
        )
        ax.axvline(1000, color="#64748b", linewidth=1, linestyle="--", zorder=1)
        ax.grid(axis="x", color="#e2e8f0")
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if self._added_models:
            legend = [
                Patch(color="#2563eb", label="Baselines"),
                Patch(color="#d97706", label="Your model"),
            ]
            ax.legend(handles=legend, loc="lower right")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal: rebuild leaderboard from raw metrics
    # ------------------------------------------------------------------

    def _upsert_display_meta(self, model_id: str, values: dict) -> None:
        """Insert or update display metadata columns for *model_id*."""
        if self._display_meta.empty or "model_id" not in self._display_meta.columns:
            self._display_meta = pd.DataFrame([{"model_id": model_id, **values}])
            return
        mask = self._display_meta["model_id"] == model_id
        if mask.any():
            for col, val in values.items():
                self._display_meta.loc[mask, col] = val
        else:
            new_row = pd.DataFrame([{"model_id": model_id, **values}])
            self._display_meta = pd.concat([self._display_meta, new_row], ignore_index=True)

    def _rebuild(self) -> None:
        """Recompute Score, Avg Rank, Improvability from current raw metrics.

        These are descriptive fold-mean statistics. Headline ranks are computed from
        fold-level Elo in :meth:`rank`.
        """
        self._ratings: dict[str, pd.DataFrame] = {}
        reg_scores = _per_dataset_scores(
            self._reg_metrics, PRIMARY_REG_METRIC, higher_is_better=False
        )
        clf_scores = _per_dataset_scores(
            self._clf_metrics, PRIMARY_CLF_METRIC, higher_is_better=True
        )

        self._reg = self._merge_meta(_aggregate_leaderboard(reg_scores))
        self._clf = self._merge_meta(_aggregate_leaderboard(clf_scores))

        non_empty = [df for df in [reg_scores, clf_scores] if not df.empty]
        all_scores = pd.concat(non_empty, ignore_index=True) if non_empty else reg_scores
        self._overall = self._merge_meta(_aggregate_leaderboard(all_scores))

    def _merge_meta(self, lb: pd.DataFrame) -> pd.DataFrame:
        """Merge display metadata into a computed leaderboard DataFrame."""
        if lb.empty:
            return lb
        if self._display_meta.empty or "model_id" not in self._display_meta.columns:
            lb["Model"] = lb["model_id"]
            return lb
        available = [c for c in _META_COLS if c in self._display_meta.columns]
        meta = self._display_meta[["model_id"] + available]
        merged = lb.merge(meta, on="model_id", how="left")
        if "Model" in merged.columns:
            merged["Model"] = merged["Model"].fillna(merged["model_id"])
        else:
            merged["Model"] = merged["model_id"]
        col_order = ["model_id", "Model"] + [
            c
            for c in [
                "Category",
                "Elo",
                "Score",
                "Avg Rank",
                "Improvability",
                # Coverage sits next to Score on purpose: the two are only interpretable
                # together now that absent-by-design pairings are excluded rather than zeroed.
                "# Targets",
                "# Failed",
                "# Skipped",
                "Train Time s",
                "Infer. s/1K",
            ]
            if c in merged.columns
        ]
        extra = [c for c in merged.columns if c not in col_order]
        return merged[col_order + extra]

    def _select_leaderboard(self, task: str) -> pd.DataFrame:
        if task == "overall":
            return self._overall
        if task == "classification":
            return self._clf
        if task == "regression":
            return self._reg
        raise ValueError(
            f"Unknown task {task!r}. Use 'overall', 'classification', or 'regression'."
        )


# ------------------------------------------------------------------
# Helpers for building leaderboard from raw metrics
# ------------------------------------------------------------------


def _per_dataset_scores(
    metrics_df: pd.DataFrame,
    metric_col: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Compute per-(model, dataset) min-max normalized scores and ranks.

    For each dataset key the primary metric is min-max normalized across the whole
    field of models:
    - best model in that dataset  → 1.0
    - worst model in that dataset → 0.0
    - linear in between (no clipping)

    Anchoring the zero at the worst model, not the per-dataset median, keeps the
    bottom half of the field discriminated: a mediocre and a catastrophic model get
    different scores instead of both collapsing to 0.0. A dataset on which every
    model ties (e.g. a near-trivial task where all models are at ceiling) is
    non-discriminative — it scores every model 1.0, so it neither rewards nor
    penalises anyone.

    Returns a DataFrame with columns ``model``, ``key``, ``norm_score``,
    ``rank``.  Only datasets with at least two models are included.
    """
    if metrics_df.empty or metric_col not in metrics_df.columns:
        return pd.DataFrame(columns=["model", "key", "norm_score", "rank"])

    # Average the metric across seeds for each (model, dataset)
    per_ds = (
        metrics_df.groupby(["model", "key"])[metric_col]
        .mean()
        .reset_index()
        .dropna(subset=[metric_col])
    )

    results = []
    for key, group in per_ds.groupby("key"):
        if len(group) < 2:
            continue
        vals = group[metric_col].values
        e_vals = vals if higher_is_better else -vals

        best_e = float(np.max(e_vals))
        worst_e = float(np.min(e_vals))
        denom = best_e - worst_e

        if denom > 0:
            norm = (e_vals - worst_e) / denom
        else:
            # Every model tied on this dataset: indistinguishable, nothing to
            # improve. Score them all 1.0 so a non-discriminative dataset neither
            # rewards nor penalises anyone (0.0 would wrongly imply max Improvability).
            norm = np.ones(len(e_vals))

        ranks = pd.Series(e_vals).rank(ascending=False, method="min").values

        for i, row in enumerate(group.itertuples(index=False)):
            results.append(
                {
                    "model": row.model,
                    "key": key,
                    "norm_score": float(norm[i]),
                    "rank": float(ranks[i]),
                }
            )

    return pd.DataFrame(results)


def _aggregate_leaderboard(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-dataset scores into a leaderboard row per model.

    Every model is scored on the targets it actually has a result for, and ``# Targets``
    reports how many that is. Absent ``(model, key)`` pairings are **not** imputed here: a
    fit that was attempted and failed has already been imputed at chance level upstream
    (:func:`tabbench_bio.coverage.impute_failures`), so what remains absent is a unit the
    benchmark excluded by design — a declared input-size limit, a degenerate split, a
    duplicate grid cell. Scoring those as zero ranks the benchmark's own limits rather than
    the model, which is why ``# Targets`` must be read alongside ``Score`` and why Elo —
    which compares models through shared opponents instead of a mean over unequal pools —
    is the headline ranking.
    """
    if per_ds.empty:
        return pd.DataFrame(columns=["model_id", "Score", "Avg Rank", "Improvability", "# Targets"])

    agg = (
        per_ds.groupby("model")
        .agg(
            Score=("norm_score", "mean"),
            avg_rank=("rank", "mean"),
            n_targets=("key", "nunique"),
        )
        .reset_index()
        .rename(columns={"model": "model_id", "avg_rank": "Avg Rank", "n_targets": "# Targets"})
    )
    agg["Improvability"] = ((1.0 - agg["Score"]) * 100).round(1)
    agg["Score"] = agg["Score"].round(4)
    agg["Avg Rank"] = agg["Avg Rank"].round(1)
    return agg


def _build_leaderboard_from_metrics(
    clf_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall/clf/reg leaderboard DataFrames from raw metrics CSVs.

    Kept for backwards compatibility with :meth:`Leaderboard.from_results_dir`.
    """
    lb = Leaderboard(reg_df, clf_df)
    return lb.rank("overall"), lb.rank("classification"), lb.rank("regression")
