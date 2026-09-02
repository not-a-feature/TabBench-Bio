"""Read-only public API tests for published TabBench Bio SQLite bundles."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zlib
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from tabbench_bio import Leaderboard


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
        connection.executescript(
            """
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
            """
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("schema_version", "1"), ("experiment_id", "test-release")],
        )
        connection.executemany(
            "INSERT INTO cells VALUES (?, '{}', ?)",
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

    assert ranked["model_id"].tolist() == ["PERFECT", "DUMMY"]
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


def test_sqlite_leaderboard_accepts_a_user_dummy_model(
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
        metrics = leaderboard.evaluate_and_add(
            "USER_DUMMY",
            DummyClassifier(strategy="most_frequent"),
            config_path=str(config_path),
            seeds=1,
            task="classification",
        )

        assert metrics[["seed", "key"]].to_dict("records") == [
            {"seed": 0, "key": "synthetic_0"}
        ]
        assert "USER_DUMMY" in leaderboard.rank("classification")["model_id"].tolist()
    finally:
        monkeypatch.undo()
        dataset_registry.reload()
