"""Fold-level metrics read from result databases.

The fold-metric cache hashes this module, so it holds only the code that turns stored
predictions into metric values.
"""

import hashlib
import io
import json
import sqlite3
import zlib
from collections import deque
from concurrent.futures import Executor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tabbench_bio.dataset import TaskType
from tabbench_bio.metric_cache import MetricCache
from tabbench_bio.metrics import compute_metrics

_SQLITE_SCHEMA_VERSION = 1

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
                CASE WHEN status = 'skip' AND reason IN ('benchmark_exclusion', 'dummy_unavailable') THEN 2
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
    datasets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    ground_truth_cache: dict[str, pd.DataFrame] = {}
    assert max_pending > 0
    attempts = list(connection.execute(_CURRENT_SQLITE_ATTEMPTS, (cell,)))
    if datasets is not None:
        selected = set(datasets)
        attempts = [row for row in attempts if row[1].rsplit("_", 1)[0] in selected]
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
            y_proba = probability
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
