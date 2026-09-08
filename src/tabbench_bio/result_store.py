"""Transactional benchmark result bundles.

Each physical writer host owns one SQLite bundle.  Bundles are consolidated into a
read-only canonical database for aggregation and publication; no database is written by
multiple hosts.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import socket
import sqlite3
import stat
import tempfile
import uuid
import zlib
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tabbench_bio.split_manifest import load_manifest, split_versions, validate_truth

SCHEMA_VERSION = 1
CANONICAL_FILENAME = "results.sqlite"
WRITER_DIRECTORY = "writers"
SNAPSHOT_DIRECTORY = "snapshots"
TERMINAL_STATES = ("pass", "skip", "fail")
STATUS_FIELDS = ("cell", "seed", "dataset", "model", "status", "reason")
RUN_STATS_FIELDS = (
    *STATUS_FIELDS,
    "n_train_samples",
    "n_test_samples",
    "train_time_s",
    "inference_time_s",
    "inference_time_per_sample_ms",
    "train_peak_memory_mb",
    "inference_peak_memory_mb",
    "n_models_trained",
    "n_base_models",
    "ag_total_fit_time_s",
    "ag_time_per_model_s",
    "train_gpu_energy_j",
    "train_cpu_energy_j",
    "inference_gpu_energy_j",
    "inference_cpu_energy_j",
)
_UNIT_KEY = ("cell", "seed", "dataset", "model")
_ARTIFACT_KINDS = ("prediction", "probability", "ground_truth", "log")
_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "output_dir",
        "cache_dir",
        "dataset_names_classification",
        "dataset_names_regression",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_writer_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    assert safe, f"Invalid writer id: {value!r}"
    return safe


def writer_id() -> str:
    """Return the explicit writer id, or the current physical host name."""
    value = (
        os.environ["TABBENCH_WRITER_ID"] if "TABBENCH_WRITER_ID" in os.environ else socket.getfqdn()
    )
    return _safe_writer_id(value)


def result_location(output_dir: str | os.PathLike[str]) -> tuple[Path, str]:
    """Return ``(result root, cell name)`` for a pipeline output directory."""
    output = Path(output_dir).resolve()
    if output.name.startswith("cap_"):
        return output.parent, output.name
    return output, "run"


def _normalise_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in _RUNTIME_CONFIG_KEYS}


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=True, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _bytes_frame(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), index_col=0)


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        assert path.is_file(), path
        uri_path = path.absolute().as_posix()
        if uri_path.startswith("//?/UNC/"):
            uri_path = "//" + uri_path[8:]
        elif uri_path.startswith("//?/"):
            uri_path = uri_path[4:]
        connection = sqlite3.connect(Path(uri_path).as_uri() + "?mode=ro", uri=True, timeout=60)
        connection.execute("PRAGMA query_only=ON")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=60)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _initialise(connection: sqlite3.Connection, *, experiment_id: str, bundle_writer: str) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cells (
            cell TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            config_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS blobs (
            sha256 TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            encoding TEXT NOT NULL,
            uncompressed_bytes INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY,
            cell TEXT NOT NULL REFERENCES cells(cell),
            seed INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pass', 'skip', 'fail')),
            reason TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            record_json TEXT NOT NULL,
            prediction_sha256 TEXT REFERENCES blobs(sha256),
            probability_sha256 TEXT REFERENCES blobs(sha256),
            ground_truth_sha256 TEXT REFERENCES blobs(sha256),
            log_sha256 TEXT REFERENCES blobs(sha256)
        );
        CREATE INDEX IF NOT EXISTS attempts_unit
            ON attempts(cell, seed, dataset, model, timestamp, attempt_id);
        """
    )
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "experiment_id": experiment_id,
        "writer_id": bundle_writer,
    }
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    for key, value in expected.items():
        if key in existing:
            assert existing[key] == value, (key, existing[key], value)
        else:
            connection.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (key, value))
    if "created_utc" not in existing:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", ("created_utc", _utc_now())
        )
    if "host" not in existing:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", ("host", socket.getfqdn())
        )
    if "platform" not in existing:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("platform", platform.platform()),
        )


def _register_cell(connection: sqlite3.Connection, cell: str, config: dict) -> None:
    config_json = _canonical_json(_normalise_config(config))
    config_hash = _sha256(config_json.encode("utf-8"))
    connection.execute(
        "INSERT OR IGNORE INTO cells(cell, config_json, config_sha256) VALUES (?, ?, ?)",
        (cell, config_json, config_hash),
    )
    existing = connection.execute(
        "SELECT config_json, config_sha256 FROM cells WHERE cell = ?", (cell,)
    ).fetchone()
    assert existing == (config_json, config_hash), f"Configuration drift for result cell {cell}"


def _insert_blob(connection: sqlite3.Connection, kind: str, payload: bytes | None) -> str | None:
    if payload is None:
        return None
    assert kind in _ARTIFACT_KINDS, kind
    digest = _sha256(payload)
    compressed = zlib.compress(payload, level=6)
    connection.execute(
        "INSERT OR IGNORE INTO blobs(sha256, kind, encoding, uncompressed_bytes, payload) "
        "VALUES (?, ?, 'zlib', ?, ?)",
        (digest, kind, len(payload), compressed),
    )
    stored = connection.execute(
        "SELECT uncompressed_bytes FROM blobs WHERE sha256 = ?", (digest,)
    ).fetchone()
    assert stored == (len(payload),), (digest, stored, len(payload))
    return digest


def _validate_frozen_splits(connection: sqlite3.Connection, manifest: dict | None) -> None:
    if manifest is None:
        return
    for seed, dataset, digest in connection.execute(
        "SELECT DISTINCT seed, dataset, ground_truth_sha256 FROM attempts "
        "WHERE ground_truth_sha256 IS NOT NULL"
    ):
        validate_truth(manifest, seed, dataset, digest)


@dataclass(frozen=True)
class StoredAttempt:
    attempt_id: str
    cell: str
    seed: int
    dataset: str
    model: str
    status: str
    reason: str
    timestamp: str
    record: dict
    prediction_sha256: str | None
    probability_sha256: str | None
    ground_truth_sha256: str | None
    log_sha256: str | None

    @property
    def key(self) -> tuple[str, int, str, str]:
        return self.cell, self.seed, self.dataset, self.model

    @property
    def artifact_hashes(self) -> tuple[str | None, ...]:
        return (
            self.prediction_sha256,
            self.probability_sha256,
            self.ground_truth_sha256,
        )


def _attempt_from_row(row: tuple) -> StoredAttempt:
    return StoredAttempt(
        attempt_id=row[0],
        cell=row[1],
        seed=int(row[2]),
        dataset=row[3],
        model=row[4],
        status=row[5],
        reason=row[6],
        timestamp=row[7],
        record=json.loads(row[8]),
        prediction_sha256=row[9],
        probability_sha256=row[10],
        ground_truth_sha256=row[11],
        log_sha256=row[12],
    )


_ATTEMPT_SELECT = (
    "SELECT attempt_id, cell, seed, dataset, model, status, reason, timestamp, record_json, "
    "prediction_sha256, probability_sha256, ground_truth_sha256, log_sha256 FROM attempts"
)


class ResultRepository:
    """Read all available bundles and write to this host's bundle."""

    def __init__(self, output_dir: str | os.PathLike[str], config: dict):
        self.root, self.cell = result_location(output_dir)
        self.experiment_id = self.root.name
        self.writer = writer_id()
        self.writer_path = self.root / WRITER_DIRECTORY / f"{self.writer}.sqlite"
        self.cell_filter = self.cell
        output_path = Path(output_dir).resolve()
        bundle_paths = []
        if (self.root / CANONICAL_FILENAME).is_file():
            bundle_paths.append(self.root / CANONICAL_FILENAME)
        bundle_paths.extend(sorted((self.root / WRITER_DIRECTORY).glob("*.sqlite")))
        legacy_exists = next(output_path.glob("seed_*/stats/*.json"), None) is not None
        cell_registered = False
        for bundle_path in bundle_paths:
            with closing(_connect(bundle_path, read_only=True)) as connection:
                cells_ready = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cells'"
                ).fetchone()
                if cells_ready is None:
                    assert bundle_path == self.writer_path, bundle_path
                    continue
                cell_registered = (
                    connection.execute(
                        "SELECT 1 FROM cells WHERE cell = ?", (self.cell,)
                    ).fetchone()
                    is not None
                )
            if cell_registered:
                break
        assert cell_registered or not legacy_exists, (
            f"Legacy result files under {output_path} have not been imported into the "
            f"result database. "
            f"Run `tabbench-bio results import-legacy --results-dir {self.root}` before "
            f"starting database writers."
        )
        config_path = output_path / "config.json"
        if config_path.is_file():
            self.cell_config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            self.cell_config = json.loads(json.dumps(config))
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(_connect(self.writer_path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            _initialise(connection, experiment_id=self.experiment_id, bundle_writer=self.writer)
            _register_cell(connection, self.cell, self.cell_config)
            connection.commit()
        self._attempts: dict[tuple[str, int, str, str], list[StoredAttempt]] = {}
        self._blob_sources: dict[str, Path] = {}
        self.reload()

    @classmethod
    def from_root(
        cls, results_root: str | os.PathLike[str], *, cell: str | None = None
    ) -> ResultRepository:
        """Open an experiment root without creating a writer bundle.

        Supplying a cell avoids loading unrelated attempts from a grid experiment.
        """
        repository = cls.__new__(cls)
        repository.root = Path(results_root).resolve()
        repository.cell = "run" if cell is None else cell
        repository.cell_filter = cell
        repository.experiment_id = repository.root.name
        repository.writer = writer_id()
        repository.writer_path = repository.root / WRITER_DIRECTORY / f"{repository.writer}.sqlite"
        repository.cell_config = None
        repository._attempts = {}
        repository._blob_sources = {}
        repository.reload()
        return repository

    def bundle_paths(self) -> list[Path]:
        """Return the canonical and live writer bundles currently in scope."""
        paths = []
        canonical = self.root / CANONICAL_FILENAME
        if canonical.is_file():
            paths.append(canonical)
        writers = self.root / WRITER_DIRECTORY
        if writers.is_dir():
            paths.extend(sorted(writers.glob("*.sqlite")))
        return paths

    def reload(self) -> None:
        self.split_manifest = load_manifest(self.root)
        self.split_manifest_sha256 = (
            None
            if self.split_manifest is None
            else _sha256(_canonical_json(self.split_manifest).encode())
        )
        attempts_by_id: dict[str, StoredAttempt] = {}
        blob_sources: dict[str, Path] = {}
        for path in self.bundle_paths():
            with closing(_connect(path, read_only=True)) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                assert int(metadata["schema_version"]) == SCHEMA_VERSION, path
                assert metadata["experiment_id"] == self.experiment_id, path
                query = _ATTEMPT_SELECT
                parameters: tuple = ()
                if self.cell_filter is not None:
                    query += " WHERE cell = ?"
                    parameters = (self.cell_filter,)
                for row in connection.execute(query, parameters):
                    attempt = _attempt_from_row(row)
                    if self.split_manifest is not None and attempt.ground_truth_sha256 is not None:
                        validate_truth(
                            self.split_manifest,
                            attempt.seed,
                            attempt.dataset,
                            attempt.ground_truth_sha256,
                        )
                    if attempt.attempt_id in attempts_by_id:
                        assert attempts_by_id[attempt.attempt_id] == attempt
                    else:
                        attempts_by_id[attempt.attempt_id] = attempt
                    for digest in (*attempt.artifact_hashes, attempt.log_sha256):
                        if digest is not None:
                            blob_sources.setdefault(digest, path)
        grouped: dict[tuple[str, int, str, str], list[StoredAttempt]] = {}
        for attempt in attempts_by_id.values():
            grouped.setdefault(attempt.key, []).append(attempt)
        for attempts in grouped.values():
            attempts.sort(key=lambda attempt: (attempt.timestamp, attempt.attempt_id))
            passes = [attempt for attempt in attempts if attempt.status == "pass"]
            if passes:
                hashes = {attempt.artifact_hashes for attempt in passes}
                assert len(hashes) == 1, f"Divergent passing results for {passes[0].key}: {hashes}"
        self._attempts = grouped
        self._blob_sources = blob_sources

    def current(
        self, seed: int, dataset: str, model: str, *, cell: str | None = None
    ) -> StoredAttempt | None:
        key = (self.cell if cell is None else cell, int(seed), dataset, model)
        attempts = self._attempts[key] if key in self._attempts else []
        passes = [attempt for attempt in attempts if attempt.status == "pass"]
        return passes[-1] if passes else (attempts[-1] if attempts else None)

    def attempts(self, *, cell: str | None = None) -> list[StoredAttempt]:
        selected = []
        for attempts in self._attempts.values():
            selected.extend(attempt for attempt in attempts if cell is None or attempt.cell == cell)
        return sorted(selected, key=lambda attempt: (*attempt.key, attempt.timestamp))

    def current_attempts(self, *, cell: str | None = None) -> list[StoredAttempt]:
        current = []
        for key in sorted(self._attempts):
            if cell is not None and key[0] != cell:
                continue
            attempt = self.current(key[1], key[2], key[3], cell=key[0])
            assert attempt is not None
            current.append(attempt)
        return current

    def current_frame(self, *, cell: str | None = None) -> pd.DataFrame:
        rows = []
        for attempt in self.current_attempts(cell=cell):
            row = dict(attempt.record)
            row.update(
                {
                    "cell": attempt.cell,
                    "seed": attempt.seed,
                    "dataset": attempt.dataset,
                    "model": attempt.model,
                    "status": attempt.status,
                    "reason": attempt.reason,
                    "attempt_id": attempt.attempt_id,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _blob(self, digest: str) -> bytes:
        path = self._blob_sources[digest]
        with closing(_connect(path, read_only=True)) as connection:
            row = connection.execute(
                "SELECT encoding, uncompressed_bytes, payload FROM blobs WHERE sha256 = ?",
                (digest,),
            ).fetchone()
        assert row is not None, digest
        encoding, size, compressed = row
        assert encoding == "zlib", encoding
        payload = zlib.decompress(compressed)
        assert len(payload) == size
        assert _sha256(payload) == digest
        return payload

    def dataframe(self, attempt: StoredAttempt, kind: str) -> pd.DataFrame | None:
        assert kind in {"prediction", "probability", "ground_truth"}, kind
        digest = getattr(attempt, f"{kind}_sha256")
        return None if digest is None else _bytes_frame(self._blob(digest))

    def log_text(self, attempt: StoredAttempt) -> str:
        return "" if attempt.log_sha256 is None else self._blob(attempt.log_sha256).decode("utf-8")

    def write(
        self,
        record: dict,
        *,
        seed: int,
        prediction: pd.DataFrame | pd.Series | None = None,
        probability: pd.DataFrame | None = None,
        ground_truth: pd.DataFrame | None = None,
        log_text: str = "",
    ) -> StoredAttempt:
        assert record["status"] in TERMINAL_STATES, record["status"]
        assert record["dataset"] and record["model"]
        if record["status"] == "pass":
            assert prediction is not None and ground_truth is not None
            if isinstance(prediction, pd.Series):
                assert prediction.name == "target", prediction.name
                prediction = prediction.to_frame()
            assert "target" in prediction and "target" in ground_truth
            assert prediction.index.equals(ground_truth.index), (
                record["dataset"],
                record["model"],
            )
            if probability is not None:
                assert probability.index.equals(ground_truth.index), (
                    record["dataset"],
                    record["model"],
                )
            cell_config = self.cell_config
            classification = (
                record["task_type"] == "classification" if "task_type" in record else False
            )
            if "task_type" not in record and "datasets_classification" in cell_config:
                dataset_name = record["dataset"].rsplit("_", 1)[0]
                configured = cell_config["datasets_classification"]
                classification = configured is not None and dataset_name in set(configured)
            if classification:
                assert probability is not None, (
                    f"Passing classification unit lacks probabilities: "
                    f"{record['dataset']}/{record['model']}"
                )
        timestamp = record["timestamp"] if "timestamp" in record else _utc_now()
        payload = dict(record)
        payload["timestamp"] = timestamp
        payload["writer_id"] = self.writer
        payload["host"] = socket.getfqdn()
        if self.split_manifest is not None:
            payload["split_manifest_sha256"] = self.split_manifest_sha256
            payload["split_versions"] = split_versions()
        stable = {
            key: value
            for key, value in payload.items()
            if key not in {"timestamp", "writer_id", "host"}
        }
        prior = self.current(seed, payload["dataset"], payload["model"])
        if prior is not None and prior.status != "pass" and payload["status"] != "pass":
            prior_stable = {
                key: value
                for key, value in prior.record.items()
                if key not in {"timestamp", "writer_id", "host"}
            }
            if prior_stable == stable:
                return prior

        attempt_id = uuid.uuid4().hex
        if self.split_manifest is not None and ground_truth is not None:
            validate_truth(
                self.split_manifest, seed, payload["dataset"], _sha256(_frame_bytes(ground_truth))
            )
        with closing(_connect(self.writer_path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            _initialise(connection, experiment_id=self.experiment_id, bundle_writer=self.writer)
            _register_cell(connection, self.cell, self.cell_config)
            hashes = {
                "prediction": _insert_blob(
                    connection,
                    "prediction",
                    None if prediction is None else _frame_bytes(prediction),
                ),
                "probability": _insert_blob(
                    connection,
                    "probability",
                    None if probability is None else _frame_bytes(probability),
                ),
                "ground_truth": _insert_blob(
                    connection,
                    "ground_truth",
                    None if ground_truth is None else _frame_bytes(ground_truth),
                ),
                "log": _insert_blob(
                    connection, "log", log_text.encode("utf-8") if log_text else None
                ),
            }
            if payload["status"] == "pass":
                unit_key = (self.cell, int(seed), payload["dataset"], payload["model"])
                existing_attempts = self._attempts[unit_key] if unit_key in self._attempts else []
                existing_passes = [
                    attempt for attempt in existing_attempts if attempt.status == "pass"
                ]
                artifact_hashes = (
                    hashes["prediction"],
                    hashes["probability"],
                    hashes["ground_truth"],
                )
                assert all(
                    attempt.artifact_hashes == artifact_hashes for attempt in existing_passes
                ), (
                    f"Divergent passing results for {(self.cell, seed, payload['dataset'], payload['model'])}"
                )
            connection.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt_id,
                    self.cell,
                    int(seed),
                    payload["dataset"],
                    payload["model"],
                    payload["status"],
                    payload["reason"] if "reason" in payload else "",
                    timestamp,
                    _canonical_json(payload),
                    hashes["prediction"],
                    hashes["probability"],
                    hashes["ground_truth"],
                    hashes["log"],
                ),
            )
            connection.commit()
        stored = StoredAttempt(
            attempt_id=attempt_id,
            cell=self.cell,
            seed=int(seed),
            dataset=payload["dataset"],
            model=payload["model"],
            status=payload["status"],
            reason=payload["reason"] if "reason" in payload else "",
            timestamp=timestamp,
            record=payload,
            prediction_sha256=hashes["prediction"],
            probability_sha256=hashes["probability"],
            ground_truth_sha256=hashes["ground_truth"],
            log_sha256=hashes["log"],
        )
        attempts = self._attempts.setdefault(stored.key, [])
        attempts.append(stored)
        attempts.sort(key=lambda attempt: (attempt.timestamp, attempt.attempt_id))
        for digest in hashes.values():
            if digest is not None:
                self._blob_sources[digest] = self.writer_path
        return stored


def snapshot_database(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> Path:
    """Create and atomically install a consistent SQLite backup."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    assert source_path.is_file(), source_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent, prefix=f".{destination_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            closing(_connect(source_path, read_only=True)) as source_connection,
            closing(sqlite3.connect(temporary)) as destination_connection,
        ):
            source_connection.backup(destination_connection)
        with closing(_connect(temporary, read_only=True)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        temporary.chmod(stat.S_IMODE(source_path.stat().st_mode) | stat.S_IRGRP)
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def snapshot_writers(
    results_root: str | os.PathLike[str], *, exclude_writers: tuple[str, ...] = ()
) -> Path:
    """Snapshot every live writer bundle and publish one checksum manifest."""
    root = Path(results_root).resolve()
    excluded = {_safe_writer_id(value) for value in exclude_writers}
    writers = [
        path
        for path in sorted((root / WRITER_DIRECTORY).glob("*.sqlite"))
        if path.stem not in excluded
    ]
    assert writers, f"No writer bundles found under {root / WRITER_DIRECTORY}"
    snapshot_dir = root / SNAPSHOT_DIRECTORY
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in writers:
        destination = snapshot_database(source, snapshot_dir / source.name)
        entries.append(
            {
                "file": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    included = {entry["file"] for entry in entries}
    for stale in snapshot_dir.glob("*.sqlite"):
        if stale.name not in included:
            stale.unlink()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": root.name,
        "created_utc": _utc_now(),
        "bundles": entries,
    }
    destination = snapshot_dir / "manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=snapshot_dir, prefix=".manifest.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_name, 0o640)
    os.replace(temporary_name, destination)
    return destination


def install_snapshots(
    results_root: str | os.PathLike[str], snapshot_dir: str | os.PathLike[str]
) -> list[Path]:
    """Validate transferred snapshots and atomically install them as writer bundles."""
    root = Path(results_root).resolve()
    source_dir = Path(snapshot_dir).resolve()
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["experiment_id"] == root.name
    filenames = [entry["file"] for entry in manifest["bundles"]]
    assert len(filenames) == len(set(filenames)), "Snapshot manifest repeats a bundle"
    split_manifest = load_manifest(root)
    installed = []
    for entry in manifest["bundles"]:
        assert Path(entry["file"]).name == entry["file"], entry["file"]
        source = source_dir / entry["file"]
        assert source.stat().st_size == entry["bytes"], source
        assert _sha256_file(source) == entry["sha256"], source
        with closing(_connect(source, read_only=True)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            assert metadata["experiment_id"] == root.name
            _validate_frozen_splits(connection, split_manifest)
        destination = root / WRITER_DIRECTORY / source.name
        if destination.is_file():
            with closing(_connect(destination, read_only=True)) as existing_connection:
                existing_ids = {
                    row[0] for row in existing_connection.execute("SELECT attempt_id FROM attempts")
                }
            with closing(_connect(source, read_only=True)) as incoming_connection:
                incoming_ids = {
                    row[0] for row in incoming_connection.execute("SELECT attempt_id FROM attempts")
                }
            assert existing_ids.issubset(incoming_ids), (
                f"Incoming snapshot would discard {len(existing_ids - incoming_ids)} attempt(s) "
                f"from {destination.name}; transfer a newer append-only snapshot."
            )
        installed.append(snapshot_database(source, destination))
    return installed


def _copy_database_rows(source: Path, destination: sqlite3.Connection) -> None:
    with closing(_connect(source, read_only=True)) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert int(metadata["schema_version"]) == SCHEMA_VERSION, source
        destination_metadata = dict(destination.execute("SELECT key, value FROM metadata"))
        assert metadata["experiment_id"] == destination_metadata["experiment_id"], source
        for cell, config_json, config_hash in connection.execute(
            "SELECT cell, config_json, config_sha256 FROM cells"
        ):
            existing = destination.execute(
                "SELECT config_json, config_sha256 FROM cells WHERE cell = ?", (cell,)
            ).fetchone()
            assert existing is None or existing == (config_json, config_hash), (
                source,
                cell,
                existing,
                config_hash,
            )
            destination.execute(
                "INSERT OR IGNORE INTO cells VALUES (?, ?, ?)", (cell, config_json, config_hash)
            )
        for row in connection.execute(
            "SELECT sha256, kind, encoding, uncompressed_bytes, payload FROM blobs"
        ):
            existing = destination.execute(
                "SELECT sha256, kind, encoding, uncompressed_bytes, payload "
                "FROM blobs WHERE sha256 = ?",
                (row[0],),
            ).fetchone()
            if existing is not None:
                assert existing[2] == row[2] == "zlib", (source, row[0])
                existing_payload = zlib.decompress(existing[4])
                incoming_payload = zlib.decompress(row[4])
                assert len(existing_payload) == existing[3], (source, row[0])
                assert len(incoming_payload) == row[3], (source, row[0])
                assert _sha256(existing_payload) == row[0], (source, row[0])
                assert incoming_payload == existing_payload, (source, row[0])
            destination.execute("INSERT OR IGNORE INTO blobs VALUES (?, ?, ?, ?, ?)", row)
        for row in connection.execute(_ATTEMPT_SELECT):
            existing = destination.execute(
                _ATTEMPT_SELECT + " WHERE attempt_id = ?", (row[0],)
            ).fetchone()
            assert existing is None or existing == row, (source, row[0])
            destination.execute(
                "INSERT OR IGNORE INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )


def _assert_consistent_passes(connection: sqlite3.Connection) -> None:
    passing_artifacts: dict[tuple[str, int, str, str], tuple[str | None, ...]] = {}
    for row in connection.execute(_ATTEMPT_SELECT + " WHERE status = 'pass'"):
        attempt = _attempt_from_row(row)
        previous = passing_artifacts.setdefault(attempt.key, attempt.artifact_hashes)
        assert previous == attempt.artifact_hashes, (
            f"Divergent passing results for {attempt.key}: {previous} != {attempt.artifact_hashes}"
        )


def consolidate_results(results_root: str | os.PathLike[str]) -> Path:
    """Merge writer bundles into a validated canonical database atomically."""
    root = Path(results_root).resolve()
    writer_paths = sorted((root / WRITER_DIRECTORY).glob("*.sqlite"))
    assert writer_paths, f"No writer bundles found under {root / WRITER_DIRECTORY}"
    experiment_id = root.name
    destination = root / CANONICAL_FILENAME
    sources = ([destination] if destination.is_file() else []) + writer_paths
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=f".{CANONICAL_FILENAME}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with closing(sqlite3.connect(temporary, timeout=60)) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            _initialise(connection, experiment_id=experiment_id, bundle_writer="canonical")
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            for source in sources:
                _copy_database_rows(source, connection)
            connection.commit()
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            _assert_consistent_passes(connection)
            _validate_frozen_splits(connection, load_manifest(root))
        temporary.chmod(0o640)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def import_legacy_results(
    results_root: str | os.PathLike[str], *, bundle_writer: str = "legacy"
) -> Path:
    """Import the authoritative current JSON/CSV tree into one writer bundle.

    Source files are read only.  Stable content-derived attempt ids make the import
    idempotent, including after an interrupted import.
    """
    root = Path(results_root).resolve()
    assert root.is_dir(), root
    split_manifest = load_manifest(root)
    bundle_writer = _safe_writer_id(bundle_writer)
    destination = root / WRITER_DIRECTORY / f"{bundle_writer}.sqlite"
    cells = sorted(path.parent for path in root.glob("cap_*/config.json"))
    if not cells and (root / "config.json").is_file():
        cells = [root]
    assert cells, f"No configured result cells found under {root}"

    with closing(_connect(destination)) as connection:
        _initialise(connection, experiment_id=root.name, bundle_writer=bundle_writer)
        connection.commit()
        for cell_dir in cells:
            cell = cell_dir.name if cell_dir != root else "run"
            config = json.loads((cell_dir / "config.json").read_text(encoding="utf-8"))
            connection.execute("BEGIN IMMEDIATE")
            _register_cell(connection, cell, config)
            for stats_path in sorted(cell_dir.glob("seed_*/stats/*.json")):
                seed_dir = stats_path.parent.parent
                seed = int(seed_dir.name.removeprefix("seed_"))
                record = json.loads(stats_path.read_text(encoding="utf-8"))
                assert record["status"] in TERMINAL_STATES, stats_path
                dataset = record["dataset"]
                model = record["model"]
                predictions = seed_dir / "predictions"
                paths = {
                    "prediction": predictions / f"{dataset}_{model}_predictions.csv",
                    "probability": predictions / f"{dataset}_{model}_proba.csv",
                    "ground_truth": predictions / f"{dataset}_ground_truth.csv",
                    "log": seed_dir / "logs" / f"{dataset}_{model}.log",
                }
                if record["status"] == "pass":
                    assert paths["prediction"].is_file(), paths["prediction"]
                    assert paths["ground_truth"].is_file(), paths["ground_truth"]
                    if "datasets_classification" in config:
                        classification = set(config["datasets_classification"])
                        dataset_name = dataset.rsplit("_", 1)[0]
                        if dataset_name in classification:
                            assert paths["probability"].is_file(), paths["probability"]
                artifact_payloads = {
                    kind: path.read_bytes() if path.is_file() else None
                    for kind, path in paths.items()
                }
                if record["status"] == "pass":
                    prediction_frame = _bytes_frame(artifact_payloads["prediction"])
                    ground_truth_frame = _bytes_frame(artifact_payloads["ground_truth"])
                    assert prediction_frame.index.equals(ground_truth_frame.index), stats_path
                    if artifact_payloads["probability"] is not None:
                        probability_frame = _bytes_frame(artifact_payloads["probability"])
                        assert probability_frame.index.equals(ground_truth_frame.index), stats_path
                hashes = {
                    kind: _insert_blob(connection, kind, payload)
                    for kind, payload in artifact_payloads.items()
                }
                if split_manifest is not None and hashes["ground_truth"] is not None:
                    validate_truth(split_manifest, seed, dataset, hashes["ground_truth"])
                timestamp = (
                    record["timestamp"]
                    if "timestamp" in record
                    else datetime.fromtimestamp(stats_path.stat().st_mtime, UTC)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                payload = dict(record)
                payload["timestamp"] = timestamp
                payload["writer_id"] = bundle_writer
                payload["host"] = "legacy-import"
                identity = _canonical_json(
                    {
                        "source": stats_path.relative_to(root).as_posix(),
                        "record": payload,
                        "artifacts": hashes,
                    }
                ).encode("utf-8")
                attempt_id = _sha256(identity)
                row = (
                    attempt_id,
                    cell,
                    seed,
                    dataset,
                    model,
                    payload["status"],
                    payload["reason"] if "reason" in payload else "",
                    timestamp,
                    _canonical_json(payload),
                    hashes["prediction"],
                    hashes["probability"],
                    hashes["ground_truth"],
                    hashes["log"],
                )
                existing = connection.execute(
                    _ATTEMPT_SELECT + " WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
                assert existing is None or existing == row, stats_path
                connection.execute(
                    "INSERT OR IGNORE INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            connection.commit()
    return destination
