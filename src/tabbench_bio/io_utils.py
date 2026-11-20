"""Crash-safe writes for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

RESULT_BLOB_ENCODING = "zlib+blake2b-128"
ATTEMPT_SUMMARY_FIELDS = (
    "task_type",
    "n_train_samples",
    "n_test_samples",
    "n_features",
    "train_time_s",
    "inference_time_s",
    "inference_time_per_sample_ms",
    "inference_batch_size",
    "inference_oom_retries",
    "train_peak_memory_mb",
    "inference_peak_memory_mb",
    "train_gpu_energy_j",
    "train_cpu_energy_j",
    "inference_gpu_energy_j",
    "inference_cpu_energy_j",
    "hardware",
    "slurm_cpus_per_task",
    "slurm_memory_mb",
    "allocated_cpus",
    "resource_profile",
    "recommended_max_n_train",
    "memory_failure_source",
    "memory_failure_scope",
    "memory_failure_stage",
    "memory_retry_version",
    "log_memory_failure",
    "error",
)


def archive_existing(
    path: str | os.PathLike[str], *, history_dir_name: str = "history"
) -> Path | None:
    """Copy an existing artifact into a timestamped sibling history directory."""
    source = Path(path)
    if not source.is_file():
        return None
    history = source.parent / history_dir_name
    history.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    destination = history / (f"{source.stem}.{timestamp}.{uuid.uuid4().hex[:8]}{source.suffix}")
    shutil.copy2(source, destination)
    return destination


def sha256_file(path: Path) -> str:
    """Hash a file without reading its contents into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def result_blob_digest(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def decode_result_blob(encoding: str, size: int, compressed: bytes, digest: str) -> bytes:
    """Decode a result blob and verify the digest declared by its encoding."""
    assert encoding == RESULT_BLOB_ENCODING, encoding
    payload = zlib.decompress(compressed)
    assert len(payload) == size and result_blob_digest(payload) == digest, digest
    return payload


def read_result_blob(connection: sqlite3.Connection, digest: str) -> bytes:
    row = connection.execute(
        "SELECT encoding, uncompressed_bytes, payload FROM blobs WHERE hash = ?", (digest,)
    ).fetchone()
    assert row is not None, f"Missing result blob {digest}"
    encoding, size, compressed = row
    return decode_result_blob(encoding, int(size), compressed, digest)


def pack_attempt_record(text: str) -> tuple[str, bytes]:
    """Return SQL-readable metadata and a lossless copy of the full JSON."""
    record = json.loads(text)
    summary = {key: record[key] for key in ATTEMPT_SUMMARY_FIELDS if key in record}
    return (
        json.dumps(summary, separators=(",", ":"), ensure_ascii=False),
        zlib.compress(text.encode("utf-8"), level=6),
    )


def unpack_attempt_record(compressed: bytes) -> str:
    return zlib.decompress(compressed).decode("utf-8")


@contextmanager
def _atomic_path(path: str | os.PathLike[str]):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        # Windows requires a writable descriptor for fsync.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o660
        temporary.chmod(mode | stat.S_IRGRP | stat.S_IWGRP)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: str | os.PathLike[str], payload: object, *, indent: int | None = 2
) -> None:
    """Write JSON beside its destination and atomically replace on success."""
    with _atomic_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload, handle, indent=indent, separators=(",", ":") if indent is None else None
            )
            handle.write("\n")


def atomic_to_pickle(frame: object, path: str | os.PathLike[str]) -> None:
    """Publish a complete cache object, including when workers prepare concurrently."""
    with _atomic_path(path) as temporary:
        with temporary.open("wb") as handle:
            pd.to_pickle(frame, handle)


def _verify_checksum(path: Path, expected: str, algorithm: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(
            handle, lambda: hashlib.new(algorithm, usedforsecurity=False)
        ).hexdigest()
    label = "SHA-256" if algorithm == "sha256" else algorithm.upper()
    assert actual == expected, (
        f"{path}: {label} mismatch (expected {expected}, found {actual}). "
        "Remove this file and retry downloading it."
    )


def download_file(
    url: str,
    destination: Path,
    *,
    checksum: str,
    algorithm: str = "sha256",
    timeout: int = 60,
) -> Path:
    """Reuse a verified cache file or stream and verify a download before publishing it."""
    if destination.is_file():
        _verify_checksum(destination, checksum, algorithm)
        return destination
    with _atomic_path(destination) as temporary:
        with temporary.open("wb") as target, urlopen(url, timeout=timeout) as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        _verify_checksum(temporary, checksum, algorithm)
    return destination


def atomic_to_csv(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    index: bool,
) -> None:
    """Write a dataframe atomically so an interrupted writer leaves the old CSV intact."""
    with _atomic_path(path) as temporary:
        frame.to_csv(temporary, index=index)


def atomic_to_parquet(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    index: bool,
) -> None:
    """Write a dataframe atomically so the prior Parquet file survives interruption."""
    with _atomic_path(path) as temporary:
        frame.to_parquet(temporary, index=index)
