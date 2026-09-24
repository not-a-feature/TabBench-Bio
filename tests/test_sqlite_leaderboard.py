"""Read-only public API tests for published TabBench Bio SQLite bundles."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from tabbench_bio import Leaderboard
from tabbench_bio.leaderboard import _metrics_from_sqlite
from tabbench_bio.metric_cache import MetricCache


def test_metric_cache_persists_and_skips_blob_loading(tmp_path, monkeypatch):
    database = tmp_path / "results.sqlite"
    cache_path = tmp_path / "fold_metrics.sqlite"
    _build_release_bundle(database)
    checksum = hashlib.sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database) as connection:
        with closing(MetricCache(cache_path)) as cache, ProcessPoolExecutor(max_workers=2) as pool:
            cold = _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache, executor=pool)
        monkeypatch.setattr(
            "tabbench_bio.leaderboard._sqlite_frame",
            lambda *a: pytest.fail("Cached predictions should not be loaded"),
        )
        with closing(MetricCache(cache_path)) as cache:
            warm = _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache)
        for expected, actual in zip(cold, warm, strict=True):
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == checksum


def test_metric_cache_keeps_completed_batches_after_interruption(tmp_path, monkeypatch):
    database, cache_path = tmp_path / "results.sqlite", tmp_path / "cache.sqlite"
    _build_release_bundle(database)
    with sqlite3.connect(database) as connection:
        for seed in range(1, 20):
            connection.execute(
                "INSERT INTO attempts SELECT attempt_id || ?, cell, ?, dataset, model, "
                "status, reason, timestamp, record_json, prediction_sha256, probability_sha256, "
                "ground_truth_sha256, log_sha256 FROM attempts WHERE seed = 0",
                (f"-{seed}", seed),
            )
        connection.commit()
        original_read = MetricCache.read
        calls = 0

        def interrupted_read(self, key):
            nonlocal calls
            calls += 1
            if calls == 20:
                raise KeyboardInterrupt
            return original_read(self, key)

        with monkeypatch.context() as patch:
            patch.setattr(MetricCache, "read", interrupted_read)
            with closing(MetricCache(cache_path)) as cache, pytest.raises(KeyboardInterrupt):
                _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache)
        monkeypatch.setattr(
            "tabbench_bio.leaderboard._sqlite_frame",
            lambda *a: pytest.fail("Completed batches must remain cached after interruption"),
        )
        with closing(MetricCache(cache_path)) as cache:
            _, classification, _ = _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache)
            assert len(classification) == 40


@pytest.mark.parametrize("artifact", ["prediction", "probability", "ground_truth"])
def test_metric_cache_recomputes_only_changed_artifacts(tmp_path, capsys, artifact):
    database = tmp_path / "results.sqlite"
    _build_release_bundle(database)
    with (
        sqlite3.connect(database) as connection,
        closing(MetricCache(tmp_path / "cache.sqlite")) as cache,
    ):
        _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache)
        changed = (
            pd.DataFrame({0: [0.5] * 4, 1: [0.5] * 4}, index=[10, 11, 12, 13])
            if artifact == "probability"
            else pd.DataFrame({"target": [1, 1, 0, 0]}, index=[10, 11, 12, 13])
        )
        digest = _add_blob(connection, artifact, changed)
        connection.execute(
            f"UPDATE attempts SET {artifact}_sha256 = ? WHERE model = 'PERFECT'", (digest,)
        )
        connection.commit()
        expected = _metrics_from_sqlite(connection, "cap_10000_n100")
        actual = _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache, show_progress=True)
        assert "reused 1/2; computed 1" in capsys.readouterr().out
        for a, b in zip(expected, actual, strict=True):
            pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_metric_cache_invalidates_implementation_and_checks_integrity(
    tmp_path, monkeypatch, capsys
):
    database, cache_path = tmp_path / "results.sqlite", tmp_path / "cache.sqlite"
    _build_release_bundle(database)
    with sqlite3.connect(database) as connection:
        with closing(MetricCache(cache_path)) as cache:
            _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache)
        monkeypatch.setattr(
            "tabbench_bio.metric_cache.metric_algorithm_sha256", lambda: "new-version"
        )
        with closing(MetricCache(cache_path)) as cache:
            _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache, show_progress=True)
            assert "reused 0/2; computed 2" in capsys.readouterr().out
            cache.connection.execute("UPDATE fold_metrics SET metrics_json = '{}' ")
            cache.connection.commit()
            with pytest.raises(AssertionError, match="Corrupt fold-metric cache"):
                _metrics_from_sqlite(connection, "cap_10000_n100", cache=cache)


def test_parallel_metrics_match_serial_across_batches(tmp_path):
    database = tmp_path / "results.sqlite"
    _build_release_bundle(database)
    with sqlite3.connect(database) as connection:
        for seed in range(1, 40):
            connection.execute(
                "INSERT INTO attempts SELECT attempt_id || ?, cell, ?, dataset, model, "
                "status, reason, timestamp, record_json, prediction_sha256, probability_sha256, "
                "ground_truth_sha256, log_sha256 FROM attempts WHERE seed = 0",
                (f"-{seed}", seed),
            )
        connection.commit()
        serial = _metrics_from_sqlite(connection, "cap_10000_n100")
        with ProcessPoolExecutor(max_workers=2) as pool:
            parallel = _metrics_from_sqlite(
                connection, "cap_10000_n100", executor=pool, max_pending=2
            )
    assert len(serial[1]) == 80
    for expected, actual in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def _frame_payload(frame: pd.DataFrame) -> tuple[str, bytes, int]:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=True, lineterminator="\n")
    payload = buffer.getvalue().encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), zlib.compress(payload, level=6), len(payload)


def _add_blob(connection: sqlite3.Connection, kind: str, frame: pd.DataFrame) -> str:
    digest, compressed, size = _frame_payload(frame)
    connection.execute(
        "INSERT OR IGNORE INTO blobs VALUES (?, ?, 'zlib', ?, ?)",
        (digest, kind, size, compressed),
    )
    return digest


def _build_release_bundle(path: Path) -> None:
    truth = pd.DataFrame({"target": [0, 0, 1, 1]}, index=[10, 11, 12, 13])
    predictions = {
        "DUMMY": pd.DataFrame({"target": [0, 0, 0, 0]}, index=truth.index),
        "PERFECT": truth.copy(),
    }
    probabilities = {
        "DUMMY": pd.DataFrame({0: [0.5] * 4, 1: [0.5] * 4}, index=truth.index),
        "PERFECT": pd.DataFrame(
            {0: [0.99, 0.99, 0.01, 0.01], 1: [0.01, 0.01, 0.99, 0.99]},
            index=truth.index,
        ),
    }
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE cells (
                cell TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                config_sha256 TEXT NOT NULL
            );
            CREATE TABLE blobs (
                sha256 TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                encoding TEXT NOT NULL,
                uncompressed_bytes INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                cell TEXT NOT NULL REFERENCES cells(cell),
                seed INTEGER NOT NULL,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                record_json TEXT NOT NULL,
                prediction_sha256 TEXT REFERENCES blobs(sha256),
                probability_sha256 TEXT REFERENCES blobs(sha256),
                ground_truth_sha256 TEXT REFERENCES blobs(sha256),
                log_sha256 TEXT REFERENCES blobs(sha256)
            );
            """)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("schema_version", "1"), ("experiment_id", "test-release")],
        )
        connection.executemany(
            "INSERT INTO cells VALUES "
            '(?, \'{"cv_folds": null, "n_repetitions": null, "random_state": 0}\', ?)',
            [("cap_2000_n20", "first"), ("cap_10000_n100", "second")],
        )
        truth_digest = _add_blob(connection, "ground_truth", truth)
        for model in ("DUMMY", "PERFECT"):
            prediction_digest = _add_blob(connection, "prediction", predictions[model])
            probability_digest = _add_blob(connection, "probability", probabilities[model])
            connection.execute(
                "INSERT INTO attempts VALUES (?, ?, 0, 'synthetic_0', ?, 'pass', '', ?, "
                "'{}', ?, ?, ?, NULL)",
                (
                    f"attempt-{model}",
                    "cap_10000_n100",
                    model,
                    "2026-09-02T00:00:00Z",
                    prediction_digest,
                    probability_digest,
                    truth_digest,
                ),
            )


def test_sqlite_loader_is_read_only_and_ranks_selected_cell(tmp_path: Path) -> None:
    path = tmp_path / "tabbench-bio-results.sqlite"
    _build_release_bundle(path)
    before = (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())

    assert Leaderboard.sqlite_cells(path) == ["cap_10000_n100", "cap_2000_n20"]
    leaderboard = Leaderboard.from_sqlite(path, cell="cap_10000_n100")
    ranked = leaderboard.rank("classification")

    assert set(ranked["model_id"]) == {"PERFECT", "DUMMY"}
    assert ranked["Elo"].isna().all() and ranked["Rank"].isna().all()
    assert ranked["# Targets"].tolist() == [1, 1]
    after = (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
    assert after == before
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-journal").exists()


def test_sqlite_loader_requires_cell_for_multi_cell_bundle(tmp_path: Path) -> None:
    path = tmp_path / "tabbench-bio-results.sqlite"
    _build_release_bundle(path)

    with pytest.raises(ValueError, match="contains 2 cells"):
        Leaderboard.from_sqlite(path)


def test_sqlite_leaderboard_rejects_custom_model_on_different_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "tabbench-bio-results.sqlite"
    _build_release_bundle(sqlite_path)
    dataset_path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "feature_a": list(range(40)),
            "feature_b": [value % 3 for value in range(40)],
            "target": [0, 1] * 20,
        }
    ).to_csv(dataset_path, index=False)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "bio_id": "synthetic",
                    "source": "local",
                    "fetch_id": dataset_path.name,
                    "target": "target",
                    "problem_type": "binary",
                    "enabled": True,
                    "redistributable": True,
                    "license": "CC0-1.0",
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets_classification": ["synthetic"],
                "datasets_regression": [],
                "models": ["DUMMY"],
                "test_size": 0.2,
                "random_state": 0,
                "n_repetitions": 1,
                "cv_folds": None,
                "min_samples_per_class": 3,
                "group_regression_splits": False,
                "bio_max_features": None,
                "max_classes": None,
                "train_subsample": None,
                "model_limits": {},
                "model_overrides": {},
                "cache_dir": str(tmp_path / "cache"),
                "output_dir": str(tmp_path / "output"),
                "autogluon_time_limit": 30,
                "autogluon_presets": "medium_quality",
                "optimize": False,
                "ensemble": False,
                "num_hpo_trials": 0,
                "subsample": None,
                "nan_policy": None,
                "exclude_keys": [],
                "exclude_datasets": [],
                "exclude_targets": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TABBENCH_BIO_DATASETS", str(registry_path))
    monkeypatch.setenv("TABBENCH_BIO_LOCAL_DIR", str(tmp_path))

    from tabbench_bio.bio import datasets as dataset_registry

    dataset_registry.reload()
    try:
        leaderboard = Leaderboard.from_sqlite(sqlite_path, cell="cap_10000_n100")
        with pytest.raises(AssertionError, match="Reference test split"):
            leaderboard.evaluate_and_add(
                "USER_DUMMY",
                DummyClassifier(strategy="most_frequent"),
                config_path=str(config_path),
                seeds=1,
                task="classification",
            )
        assert "USER_DUMMY" not in leaderboard.rank("classification")["model_id"].tolist()
    finally:
        monkeypatch.undo()
        dataset_registry.reload()
