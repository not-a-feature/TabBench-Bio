"""Persistent, content-addressed cache for derived fold metrics."""

import hashlib
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path


def metric_algorithm_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256(b"fold-metrics-v1")
    for name in (
        "leaderboard.py",
        "metric_cache.py",
        "dataset.py",
        "metrics/__init__.py",
        "metrics/utils.py",
        "metrics/classification.py",
        "metrics/regression.py",
    ):
        digest.update((root / name).read_bytes())
    for dependency in ("numpy", "pandas", "scikit-learn", "scipy"):
        digest.update(f"{dependency}=={version(dependency)}\n".encode())
    return digest.hexdigest()


class MetricCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS fold_metrics ("
            "cache_key TEXT PRIMARY KEY, metrics_json TEXT NOT NULL, sha256 TEXT NOT NULL)"
        )
        self.algorithm = metric_algorithm_sha256()

    def key(self, prediction: str, probability: str | None, truth: str) -> str:
        payload = json.dumps((self.algorithm, prediction, probability, truth)).encode()
        return hashlib.sha256(payload).hexdigest()

    def read(self, key: str) -> dict | None:
        row = self.connection.execute(
            "SELECT metrics_json, sha256 FROM fold_metrics WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        payload, checksum = row
        assert hashlib.sha256(payload.encode()).hexdigest() == checksum, (
            "Corrupt fold-metric cache entry; delete the cache database and rebuild"
        )
        return json.loads(payload)

    def write(self, rows: list[tuple[str, dict]]) -> None:
        values = []
        for key, metrics in rows:
            payload = json.dumps(metrics, separators=(",", ":"))
            values.append((key, payload, hashlib.sha256(payload.encode()).hexdigest()))
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO fold_metrics VALUES (?, ?, ?)", values
            )

    def close(self) -> None:
        self.connection.close()
