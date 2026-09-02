"""Exercise the published SQLite API and add a tiny user model without shared state."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tempfile
import zlib
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier


def frame_payload(frame: pd.DataFrame) -> tuple[str, bytes, int]:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=True, lineterminator="\n")
    payload = buffer.getvalue().encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), zlib.compress(payload, level=6), len(payload)


def add_blob(connection: sqlite3.Connection, kind: str, frame: pd.DataFrame) -> str:
    digest, compressed, size = frame_payload(frame)
    connection.execute(
        "INSERT OR IGNORE INTO blobs VALUES (?, ?, 'zlib', ?, ?)",
        (digest, kind, size, compressed),
    )
    return digest


def build_release_bundle(path: Path) -> None:
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
            [("schema_version", "1"), ("experiment_id", "smoke-release")],
        )
        connection.execute("INSERT INTO cells VALUES ('cap_10000_n100', '{}', 'smoke')")
        truth_digest = add_blob(connection, "ground_truth", truth)
        for model in ("DUMMY", "PERFECT"):
            prediction_digest = add_blob(connection, "prediction", predictions[model])
            probability_digest = add_blob(connection, "probability", probabilities[model])
            connection.execute(
                "INSERT INTO attempts VALUES (?, 'cap_10000_n100', 0, 'synthetic_0', ?, "
                "'pass', '', '2026-09-02T00:00:00Z', '{}', ?, ?, ?, NULL)",
                (
                    f"attempt-{model}",
                    model,
                    prediction_digest,
                    probability_digest,
                    truth_digest,
                ),
            )


def write_inputs(root: Path) -> Path:
    dataset_path = root / "synthetic.csv"
    pd.DataFrame(
        {
            "feature_a": list(range(40)),
            "feature_b": [value % 3 for value in range(40)],
            "target": [0, 1] * 20,
        }
    ).to_csv(dataset_path, index=False)
    registry_path = root / "registry.json"
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
    config_path = root / "config.json"
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
                "cache_dir": str(root / "cache"),
                "output_dir": str(root / "output"),
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
    os.environ["TABBENCH_BIO_DATASETS"] = str(registry_path)
    os.environ["TABBENCH_BIO_LOCAL_DIR"] = str(root)
    return config_path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tabbench-bio-sqlite-smoke-") as temporary:
        root = Path(temporary)
        sqlite_path = root / "tabbench-bio-results.sqlite"
        build_release_bundle(sqlite_path)
        config_path = write_inputs(root)
        before = (sqlite_path.stat().st_mtime_ns, hashlib.sha256(sqlite_path.read_bytes()).hexdigest())

        from tabbench_bio import Leaderboard

        assert Leaderboard.sqlite_cells(sqlite_path) == ["cap_10000_n100"]
        leaderboard = Leaderboard.from_sqlite(sqlite_path)
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
        ranking = leaderboard.rank("classification")
        assert "USER_DUMMY" in ranking["model_id"].tolist()
        after = (sqlite_path.stat().st_mtime_ns, hashlib.sha256(sqlite_path.read_bytes()).hexdigest())
        assert after == before
        assert not Path(f"{sqlite_path}-wal").exists()
        assert not Path(f"{sqlite_path}-journal").exists()
        print("PASS: read-only SQLite load and USER_DUMMY evaluation")
        print(ranking[["Rank", "model_id", "Score"]].to_string(index=False))


if __name__ == "__main__":
    main()
