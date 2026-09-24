"""Read dashboard inputs from a single, read-only SQLite transaction."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from tabbench_bio.bio.datasets import BIO_DATASETS
from tabbench_bio.coverage import complete_folds, impute_failures
from tabbench_bio.leaderboard import (
    _metrics_from_sqlite,
    _open_results_sqlite,
    _validate_results_sqlite,
)
from tabbench_bio.metric_cache import MetricCache
from tabbench_bio.sample_fallback import record_is_memory_failure, resolve_metric_fallbacks
from tabbench_bio.seeds import get_seeds

METRICS = {
    "classification": ["f1_macro", "matthews_corrcoef", "balanced_accuracy", "roc_auc"],
    "regression": ["rmse", "mae", "r2"],
}
MODALITIES = {
    "Gene Expression": "Gene expression",
    "Metagenomic Functional Profile": "Metagenomics",
    "Metagenomic Marker Presence": "Metagenomics",
    "DNA Embedding": "GLM/PLM Embeddings",
    "Protein Embedding": "GLM/PLM Embeddings",
    "DNA Methylation": "DNA methylation",
    "Methylation": "DNA methylation",
    "SNP Genotype": "Genomic prediction",
    "Molecular Fingerprint": "Molecular properties",
    "Mass Spectrometry": "Other",
    "Misc": "Other",
}


def dataset_id(key: str, configured: set[str]) -> str:
    if key in configured:
        return key
    name, target = key.rsplit("_", 1)
    assert name in configured and target.isdigit(), f"Unconfigured target: {key}"
    return name


def dataset_metadata(configs):
    tasks = {}
    for config in configs.values():
        for task in METRICS:
            for name in config[f"datasets_{task}"]:
                assert name not in tasks or tasks[name] == task, f"Conflicting task: {name}"
                tasks[name] = task
    rows = []
    for name, task in sorted(tasks.items()):
        row = {
            "dataset_id": name,
            "display_name": name,
            "source": "custom",
            "fetch_id": "",
            "data_type": "Custom",
            "target": "target",
            "problem_type": task,
            "license": "Not specified",
            "modality": "Custom",
            "task": task.title(),
        }
        if name in BIO_DATASETS:
            spec = BIO_DATASETS[name]
            row.update(
                {
                    "display_name": spec.display_name or name,
                    "source": spec.source,
                    "fetch_id": spec.fetch_id,
                    "data_type": spec.data_type or "Other",
                    "target": spec.target or "target",
                    "problem_type": spec.problem_type,
                    "license": spec.license or "Not specified",
                    "modality": MODALITIES[spec.data_type]
                    if spec.data_type in MODALITIES
                    else (spec.data_type or "Other"),
                }
            )
        rows.append(row)
    return rows


def read_inputs(
    database: Path,
    cells: list[str] | None = None,
    *,
    workers: int = 1,
    metric_cache: Path | None = None,
):
    frames = {task: [] for task in METRICS}
    statuses = []
    assert workers > 0
    with ExitStack() as stack:
        connection = stack.enter_context(closing(_open_results_sqlite(database)))
        cache = (
            stack.enter_context(closing(MetricCache(metric_cache)))
            if metric_cache is not None
            else None
        )
        pool = (
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=workers, initializer=threadpool_limits, initargs=(1,)
                )
            )
            if workers > 1
            else None
        )
        connection.execute("BEGIN")
        _validate_results_sqlite(connection, database)
        configs = {
            name: json.loads(value)
            for name, value in connection.execute(
                "SELECT cell, config_json FROM cells ORDER BY cell"
            )
        }
        if cells is not None:
            assert set(cells) <= configs.keys(), f"Unknown cells: {set(cells) - configs.keys()}"
            configs = {name: configs[name] for name in cells}
        assert configs, "The database contains no configured cells"
        print(f"Computing fold metrics with {workers} worker(s)", flush=True)
        for index, (cell, config) in enumerate(configs.items(), start=1):
            print(f"Reading {cell} ({index}/{len(configs)} cells)", flush=True)
            reg, clf, status = _metrics_from_sqlite(
                connection,
                cell,
                executor=pool,
                max_pending=2 * workers,
                show_progress=True,
                cache=cache,
            )
            configured = set(config["datasets_classification"] + config["datasets_regression"])
            excluded = set(config["exclude_keys"]) if "exclude_keys" in config else set()
            excluded_datasets = (
                set(config["exclude_datasets"]) if "exclude_datasets" in config else set()
            )
            disabled = {
                name
                for name in configured
                if name in BIO_DATASETS and not BIO_DATASETS[name].enabled
            }
            config["datasets_classification"] = [
                name
                for name in config["datasets_classification"]
                if name not in disabled | excluded_datasets
            ]
            config["datasets_regression"] = [
                name
                for name in config["datasets_regression"]
                if name not in disabled | excluded_datasets
            ]
            for task, frame in (("classification", clf), ("regression", reg), ("status", status)):
                if frame.empty:
                    continue
                frame = frame.copy()
                frame["cell"] = cell
                frame["dataset"] = frame["key"].map(lambda key: dataset_id(key, configured))
                frame = frame[
                    ~frame["key"].isin(excluded)
                    & ~frame["dataset"].isin(disabled | excluded_datasets)
                ]
                if task == "status":
                    statuses.append(frame)
                else:
                    frame["max_features"] = config["bio_max_features"] or "full"
                    frame["n_train"] = config["train_subsample"] or "full"
                    frames[task].append(frame)
        attempt_count = connection.execute("SELECT count(*) FROM attempts").fetchone()[0]
    columns = ["cell", "seed", "key", "dataset", "model"]
    raw = {
        task: pd.concat(items, ignore_index=True)
        if items
        else pd.DataFrame(columns=[*columns, *METRICS[task]])
        for task, items in frames.items()
    }
    status = (
        pd.concat(statuses, ignore_index=True)
        if statuses
        else pd.DataFrame(columns=[*columns, "status", "reason", "ground_truth_sha256"])
    )
    for field in ("n_train_samples", "recommended_max_n_train", "train_time_s", "inference_time_s"):
        if field not in status:
            status[field] = float("nan")
    status["memory_failure"] = [
        record_is_memory_failure(record, Path("__missing_log__"))
        for record in status.to_dict("records")
    ]
    truths = {}
    for row in status.itertuples(index=False):
        if pd.isna(row.ground_truth_sha256):
            continue
        key = (row.cell, int(row.seed), row.key)
        assert key not in truths or truths[key] == row.ground_truth_sha256, (
            f"Conflicting targets: {key}"
        )
        truths[key] = row.ground_truth_sha256
    strict, adaptive, _ = resolve_metric_fallbacks(configs, raw, status, truths)
    views = {}
    for name, metrics in (("strict", strict), ("adaptive", adaptive), ("conditional", strict)):
        views[name] = {}
        for task, frame in metrics.items():
            output = []
            for cell, config in configs.items():
                cell_frame = (
                    frame[frame["cell"] == cell].copy() if not frame.empty else frame.copy()
                )
                cell_status = status[status["cell"] == cell]
                scored = (
                    impute_failures(cell_frame, cell_status)
                    if name != "conditional"
                    else cell_frame
                )
                if "imputed" not in scored:
                    scored = scored.assign(imputed=False)
                scored["imputed"] = scored["imputed"].fillna(False).astype(bool)
                output.append(complete_folds(scored, cell_status, get_seeds(config)))
            views[name][task] = pd.concat(output, ignore_index=True).reindex(
                columns=list(dict.fromkeys([*columns, *METRICS[task], *frame.columns, "imputed"]))
            )
    return configs, views, status, attempt_count


def progress_summary(configs, status):
    cells = []
    for cell, config in configs.items():
        frame = status[status["cell"] == cell]
        configured = set(config["datasets_classification"] + config["datasets_regression"])
        target_counts = frame.groupby("dataset")["key"].nunique()
        targets = sum(
            int(target_counts[name]) if name in target_counts else 1 for name in configured
        )
        expected = len(config["models"]) * len(get_seeds(config)) * targets
        counts = frame["status"].value_counts()
        counts = {key: int(counts[key]) if key in counts else 0 for key in ("pass", "skip", "fail")}
        recorded = sum(counts.values())
        assert recorded <= expected, f"More recorded units than configured in {cell}"
        cells.append(
            {
                "cell": cell,
                "expected": expected,
                "recorded": recorded,
                "fraction": recorded / expected if expected else 0,
                "status": counts,
                "status_complete": recorded == expected,
            }
        )
    expected = sum(cell["expected"] for cell in cells)
    recorded = sum(cell["recorded"] for cell in cells)
    return {
        "snapshot_utc": datetime.now(UTC).isoformat(),
        "expected": expected,
        "recorded": recorded,
        "fraction": recorded / expected if expected else 0,
        "status": {
            key: sum(cell["status"][key] for cell in cells) for key in ("pass", "skip", "fail")
        },
        "cells": cells,
        "cells_status_complete": sum(cell["status_complete"] for cell in cells),
    }
