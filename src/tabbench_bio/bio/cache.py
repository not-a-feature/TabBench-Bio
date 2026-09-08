"""Caching for fetched bio datasets.

Two layers of caching keep the loaders from re-downloading:

1. **Source caches** (per loader): the TCGA full matrix (``tcga_raw/``) and the GEO SOFT
   file (``geo_raw/``) are cached by the loaders themselves; OpenML and Kaggle reuse
   their own native caches (``~/.cache/openml``, ``~/.cache/kagglehub``).
2. **Unified dataset cache** (this module): the assembled :class:`BioRawDataset`
   (``datasets/<bio_id>.pkl``) so a second run skips fetching *and* re-assembling.

All bio caches live under one root (``default_bio_cache_dir()`` or an explicit
``cache_dir`` passed through from :class:`~tabbench_bio.benchmark.TabBenchBio`).
Pickle is used to match the split-cache convention (no extra deps).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tabbench_bio.bio.loaders.base import BioRawDataset

#: Env var to override the bio cache root.
_CACHE_ENV = "TABBENCH_BIO_CACHE"


def default_bio_cache_dir() -> Path:
    """Bio cache root used when no explicit ``cache_dir`` is provided.

    Resolves ``$TABBENCH_BIO_CACHE`` (or ``~/.cache/tabbench_bio``) and appends ``bio``.
    """
    base = os.environ.get(_CACHE_ENV) or str(Path.home() / ".cache" / "tabbench_bio")
    return Path(base) / "bio"


def _safe(bio_id: str) -> str:
    """Filesystem-safe form of a ``bio_id`` for use as a path component."""
    return bio_id.replace("/", "_").replace("\\", "_")


def dataset_cache_path(root: str | Path, bio_id: str) -> Path:
    """Path to the unified per-dataset cache file under ``root/datasets``."""
    return Path(root) / "datasets" / f"{_safe(bio_id)}.pkl"


def load_cached_raw(root: str | Path, bio_id: str) -> BioRawDataset | None:
    """Load a cached :class:`BioRawDataset`, or ``None`` if absent/unreadable."""
    path = dataset_cache_path(root, bio_id)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        # A corrupt/partial cache should not be fatal — drop it and re-fetch.
        path.unlink(missing_ok=True)
        return None


def save_cached_raw(root: str | Path, raw: BioRawDataset) -> Path:
    """Persist a :class:`BioRawDataset` to the unified cache and return its path."""
    path = dataset_cache_path(root, raw.bio_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(raw, path)
    return path
