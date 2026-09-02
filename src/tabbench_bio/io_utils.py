"""Crash-safe writes for benchmark artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
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


def _temporary_sibling(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    return descriptor, Path(name)


def _commit_temporary(path: Path, temporary: Path) -> None:
    os.replace(temporary, path)


def atomic_write_json(path: str | os.PathLike[str], payload: object, *, indent: int = 2) -> None:
    """Write JSON beside its destination and atomically replace on success."""
    destination = Path(path)
    descriptor, temporary = _temporary_sibling(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _commit_temporary(destination, temporary)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_to_csv(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    index: bool,
) -> None:
    """Write a dataframe atomically so an interrupted writer leaves the old CSV intact."""
    destination = Path(path)
    descriptor, temporary = _temporary_sibling(destination)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=index)
        # Windows requires a writable descriptor for fsync; Linux accepts read-only
        # descriptors, which hid this portability issue in the cluster workflow.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _commit_temporary(destination, temporary)
    finally:
        temporary.unlink(missing_ok=True)
