"""Metric computation step for the benchmark pipeline (Step 2).

Reads prediction CSV files produced by :mod:`tabbench_bio.predictions` and
computes per-(seed, dataset, model) metrics, writing them to
``metrics/classification_metrics.csv`` and ``metrics/regression_metrics.csv``
inside the results directory.

Incremental updates
-------------------
Already-computed (seed, key, model) triples are skipped, so you can safely
re-run this step after adding new models or seeds.

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
from tqdm import tqdm

from tabbench_bio.benchmark import configure_benchmark
from tabbench_bio.coverage import DESIGN_SKIPS, load_status
from tabbench_bio.dataset import TaskType
from tabbench_bio.io_utils import atomic_to_csv
from tabbench_bio.metrics import compute_metrics
from tabbench_bio.seeds import get_seeds

logger = logging.getLogger(__name__)


def _load_ground_truth(predictions_dir: str, key: str, task_type_lookup: dict):
    """Return ``(data_test, task_type)`` from saved files, or ``(None, None)``."""
    if key not in task_type_lookup:
        return None, None
    path = os.path.join(predictions_dir, f"{key}_ground_truth.csv")
    if not os.path.exists(path):
        return None, None
    return pd.read_csv(path, index_col=0), task_type_lookup[key]


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


def _filter_current_metric_rows(
    frame: pd.DataFrame,
    metric_path: str,
    output_dir: str,
    status_lookup: dict[tuple[int, str, str], str],
    *,
    requires_proba: bool,
) -> tuple[pd.DataFrame, set[tuple[int, str, str]], bool]:
    """Keep only passing metric rows whose source artifacts predate the metric file."""
    metric_mtime = os.path.getmtime(metric_path)
    keep = []
    for row in frame.itertuples(index=False):
        unit = (int(row.seed), row.key, row.model)
        seed_dir = os.path.join(output_dir, f"seed_{int(row.seed)}")
        artifacts = [
            os.path.join(seed_dir, "stats", f"{row.key}_{row.model}.json"),
            os.path.join(seed_dir, "predictions", f"{row.key}_{row.model}_predictions.csv"),
            os.path.join(seed_dir, "predictions", f"{row.key}_ground_truth.csv"),
        ]
        if requires_proba:
            artifacts.append(
                os.path.join(seed_dir, "predictions", f"{row.key}_{row.model}_proba.csv")
            )
        current = status_lookup.get(unit) == "pass"
        complete = all(os.path.isfile(artifact) for artifact in artifacts)
        unchanged = complete and (
            max(os.path.getmtime(artifact) for artifact in artifacts) <= metric_mtime
        )
        keep.append(current and unchanged)
    filtered = frame.loc[keep].copy()
    done = {(int(row.seed), row.key, row.model) for row in filtered.itertuples(index=False)}
    return filtered, done, len(filtered) != len(frame)


def compute_metrics_from_predictions(config):
    """Compute metrics from saved prediction CSV files.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration.
    """
    logger.info("=" * 60 + "\nSTEP 2: Computing Metrics")

    output_dir = config["output_dir"]
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    seeds = get_seeds(config)
    model_names = config["models"]
    excluded_keys = _build_excluded_keys(config)
    if excluded_keys:
        logger.info("Excluding %d key(s): %s", len(excluded_keys), sorted(excluded_keys))

    clf_csv = os.path.join(metrics_dir, "classification_metrics.csv")
    reg_csv = os.path.join(metrics_dir, "regression_metrics.csv")
    clf_excl_csv = os.path.join(metrics_dir, "excluded_classification_metrics.csv")
    reg_excl_csv = os.path.join(metrics_dir, "excluded_regression_metrics.csv")

    status = load_status(output_dir)
    status_lookup = {
        (int(row.seed), row.key, row.model): row.status for row in status.itertuples(index=False)
    }

    def _load_existing(path, *, requires_proba):
        if not os.path.isfile(path):
            return pd.DataFrame(), set(), False
        df = pd.read_csv(path)
        return _filter_current_metric_rows(
            df,
            path,
            output_dir,
            status_lookup,
            requires_proba=requires_proba,
        )

    clf_existing, clf_done, clf_dirty = _load_existing(clf_csv, requires_proba=True)
    reg_existing, reg_done, reg_dirty = _load_existing(reg_csv, requires_proba=False)
    clf_excl_existing, clf_excl_done, clf_excl_dirty = _load_existing(
        clf_excl_csv, requires_proba=True
    )
    reg_excl_existing, reg_excl_done, reg_excl_dirty = _load_existing(
        reg_excl_csv, requires_proba=False
    )
    already_done = clf_done | reg_done | clf_excl_done | reg_excl_done

    clf_rows, reg_rows, clf_excl_rows, reg_excl_rows = [], [], [], []

    def _save(rows, existing, path, *, force=False):
        """Merge freshly computed *rows* into the on-disk CSV and rewrite it (plus its
        summary). Returns the combined frame so the caller carries it forward; *rows* is
        cleared by the caller after. Called after every seed so an interrupted metrics step
        keeps the seeds it already finished instead of discarding them all."""
        if not rows and not force:
            return existing
        new = pd.DataFrame(rows)
        if existing.empty:
            combined = new if not new.empty else existing
        else:
            combined = pd.concat([existing, new], ignore_index=True)
        atomic_to_csv(combined, path, index=False)
        _write_summary(combined, path.replace(".csv", "_summary.csv"))
        logger.info("Wrote %d new rows to %s (total %d).", len(new), path, len(combined))
        return combined

    # Units the predictions step excluded by design. A resumed run can leave prediction CSVs
    # behind from before an exclusion applied (e.g. a grid cell later recognised as a
    # duplicate of the full-sample cell); scoring those would put rows in the metrics CSV
    # that the status records say should not exist.
    excluded = status[(status["status"] == "skip") & status["reason"].isin(DESIGN_SKIPS)]
    excluded_units = {(r.seed, r.key, r.model) for r in excluded.itertuples(index=False)}
    if excluded_units:
        logger.info("Excluding %d unit(s) skipped by design.", len(excluded_units))

    for seed in seeds:
        logger.info("--- Seed %s ---", seed)
        predictions_dir = os.path.join(output_dir, f"seed_{seed}", "predictions")

        # Build (key → task_type) lookup from ground-truth files
        gt_keys = (
            [
                f[: -len("_ground_truth.csv")]
                for f in os.listdir(predictions_dir)
                if f.endswith("_ground_truth.csv")
            ]
            if os.path.isdir(predictions_dir)
            else []
        )

        config["random_state"] = seed
        benchmark = configure_benchmark(config, init_benchmark=not bool(gt_keys))

        if gt_keys:
            # Build task_type_lookup directly from dataset name lists.
            # _index is empty when init_benchmark=False (data loading skipped),
            # so we infer task type from membership in the regression name set.
            reg_names = set(benchmark.dataset_names_regression)
            task_type_lookup = {
                key: (
                    TaskType.Regression
                    if benchmark.split_key(key)[0] in reg_names
                    else TaskType.Classification
                )
                for key in gt_keys
            }

            items = []
            for key in gt_keys:
                data_test, task_type = _load_ground_truth(predictions_dir, key, task_type_lookup)
                if data_test is not None:
                    items.append((data_test, key, task_type))
        else:
            items = [(dt, k, tt) for _, dt, k, tt in benchmark]

        pbar = tqdm(total=len(items) * len(model_names), desc=f"Metrics seed {seed}")

        for data_test, key, task_type in items:
            is_excluded = key in excluded_keys
            dataset_name, target_idx = benchmark.split_key(key)

            for model_name in model_names:
                pbar.set_description(f"Seed {seed} | {key} | {model_name}")
                pred_path = os.path.join(predictions_dir, f"{key}_{model_name}_predictions.csv")
                unit = (seed, key, model_name)

                if status_lookup.get(unit) != "pass" or not os.path.exists(pred_path):
                    pbar.update(1)
                    continue

                y_pred = pd.read_csv(pred_path, index_col=0).sort_index()
                data_test_ = data_test.sort_index()

                if not np.array_equal(data_test_.index, y_pred.index):
                    logger.warning(
                        "Index mismatch for %s / %s seed %s — leaving source artifact unchanged.",
                        key,
                        model_name,
                        seed,
                    )
                    pbar.update(1)
                    continue

                # Guard: classification predictions must not be continuous floats
                if task_type == TaskType.Classification:
                    pred_vals = y_pred["target"]
                    if (
                        pd.api.types.is_float_dtype(pred_vals)
                        and not pred_vals.apply(float.is_integer).all()
                    ):
                        logger.warning(
                            "Skipping %s / %s seed %s: continuous floats for Classification.",
                            key,
                            model_name,
                            seed,
                        )
                        pbar.update(1)
                        continue

                row = {
                    "seed": seed,
                    "key": key,
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "target_idx": target_idx,
                    "model": model_name,
                }
                # Classification is complete only with its probability matrix: predictions
                # resume reruns pass records missing this file, so scoring labels alone here
                # would freeze a partial metric row and prevent that repair from surfacing.
                y_proba = None
                if task_type == TaskType.Classification:
                    proba_path = pred_path.replace("_predictions.csv", "_proba.csv")
                    if not os.path.exists(proba_path):
                        pbar.update(1)
                        continue
                    proba_df = pd.read_csv(proba_path, index_col=0).sort_index()
                    if not np.array_equal(proba_df.index, data_test_.index):
                        logger.warning(
                            "Probability index mismatch for %s / %s seed %s — leaving source "
                            "artifact unchanged.",
                            key,
                            model_name,
                            seed,
                        )
                        pbar.update(1)
                        continue
                    # roc_auc / log_loss index the proba columns by sorted class order
                    # (np.unique / LabelBinarizer). Reorder the saved columns to that order.
                    wanted = [str(c) for c in np.unique(data_test_["target"])]
                    proba_df.columns = [str(c) for c in proba_df.columns]
                    if set(wanted).issubset(proba_df.columns):
                        y_proba = proba_df.reindex(columns=wanted).to_numpy()
                    else:
                        y_proba = proba_df.to_numpy()

                if unit in already_done:
                    pbar.update(1)
                    continue

                row.update(
                    compute_metrics(
                        data_test_["target"], y_pred["target"], task_type, y_proba=y_proba
                    )
                )

                if task_type == TaskType.Classification:
                    (clf_excl_rows if is_excluded else clf_rows).append(row)
                else:
                    (reg_excl_rows if is_excluded else reg_rows).append(row)

                pbar.update(1)
        pbar.close()

        # Persist after each seed so a kill mid-metrics keeps completed seeds (the rows
        # are merged into the on-disk CSV and the in-memory buffers reset).
        clf_existing = _save(clf_rows, clf_existing, clf_csv, force=clf_dirty)
        clf_rows.clear()
        reg_existing = _save(reg_rows, reg_existing, reg_csv, force=reg_dirty)
        reg_rows.clear()
        clf_excl_existing = _save(
            clf_excl_rows, clf_excl_existing, clf_excl_csv, force=clf_excl_dirty
        )
        clf_excl_rows.clear()
        reg_excl_existing = _save(
            reg_excl_rows, reg_excl_existing, reg_excl_csv, force=reg_excl_dirty
        )
        reg_excl_rows.clear()
        clf_dirty = reg_dirty = clf_excl_dirty = reg_excl_dirty = False


def _write_summary(df: pd.DataFrame, path: str):
    """Write mean ± std per (key, model) across seeds."""
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
