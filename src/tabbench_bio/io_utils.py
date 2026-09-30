"""Crash-safe writes for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def atomic_write_json(path: str | os.PathLike[str], payload: object, *, indent: int = 2) -> None:
    """Write JSON beside its destination and atomically replace on success."""
    with _atomic_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
            handle.write("\n")


def atomic_to_pickle(frame: object, path: str | os.PathLike[str]) -> None:
    """Publish a complete cache object, including when workers prepare concurrently."""
    with _atomic_path(path) as temporary:
        with temporary.open("wb") as handle:
            pd.to_pickle(frame, handle)


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
