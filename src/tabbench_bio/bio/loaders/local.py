"""Local table loader with optional checksum-pinned release downloads.

For datasets that are not fetched from a remote source but ship as a curated file in
the repo (or live anywhere on disk). ``spec.fetch_id`` is a path to a tabular file
(``.csv`` / ``.tsv`` / ``.parquet``): an absolute path is used as-is, a relative path is
resolved against the local-data dir — ``$TABBENCH_BIO_LOCAL_DIR`` or the bundled
``bio/data/local/`` directory. Missing tables with a registered download URL are
cached under the bio cache's ``local_raw/`` directory, or in the explicitly selected
local-data directory.

The file is framed with the **curated** ``spec.target`` (no heuristic target detection):
that column is the label, every other column is a feature. When ``spec.embedding_column``
is set, that single column is expected to hold a per-row embedding as one comma-separated
string of floats (e.g. a DNA/protein language-model embedding) and is exploded into one
numeric feature column per dimension (``"<embedding_column>_<i>"``).

Parquet tables require the ``bio`` extra's PyArrow dependency.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

import numpy as np
import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec

#: Env var to override the local-data root (where relative ``fetch_id`` paths resolve).
_LOCAL_ENV = "TABBENCH_BIO_LOCAL_DIR"

#: Bundled local-data dir, shipped as package data alongside the registry JSON.
_BUNDLED_LOCAL = Path(__file__).parents[1] / "data" / "local"

#: Tabular file extensions we know how to load.
_TABLE_SUFFIXES = (".csv", ".tsv", ".parquet")


def _local_root() -> Path:
    """Root for relative ``fetch_id`` paths ($TABBENCH_BIO_LOCAL_DIR or the bundled dir)."""
    override = os.environ.get(_LOCAL_ENV)
    return Path(override) if override else _BUNDLED_LOCAL


def _resolve_path(fetch_id: str) -> Path:
    """Resolve ``fetch_id`` to a file path (absolute as-is, relative under the local root)."""
    path = Path(fetch_id)
    return path if path.is_absolute() else _local_root() / fetch_id


def _read_table(path: Path) -> pd.DataFrame:
    """Load a tabular file by extension; strip stray whitespace from column names."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # skipinitialspace handles a space after the delimiter; strip below covers the rest.
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _verify_download(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == expected, (
        f"{path}: SHA-256 mismatch (expected {expected}, found {actual}). "
        "Remove this file and retry downloading it."
    )


def _ensure_download(path: Path, spec: BioDatasetSpec) -> None:
    if path.is_file():
        _verify_download(path, spec.download_sha256)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    temporary = Path(name)
    try:
        with (
            os.fdopen(descriptor, "wb") as target,
            urlopen(spec.download_url, timeout=60) as source,
        ):
            shutil.copyfileobj(source, target, length=1024 * 1024)
        _verify_download(temporary, spec.download_sha256)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _explode_embedding(series: pd.Series, column: str, bio_id: str) -> pd.DataFrame:
    """Explode a column of comma-separated float strings into a wide numeric DataFrame.

    Each row's string is parsed into a float vector; all rows must share one width.
    Columns are named ``"<column>_<i>"`` and the original row index is preserved.
    """
    if series.isna().any():
        raise ValueError(
            f"{bio_id}: embedding column {column!r} has {int(series.isna().sum())} missing value(s)."
        )
    vectors = [np.fromstring(str(s), sep=",", dtype=np.float32) for s in series]
    widths = {v.shape[0] for v in vectors}
    if widths == {0}:
        raise ValueError(
            f"{bio_id}: embedding column {column!r} parsed to empty vectors (wrong separator?)."
        )
    if len(widths) != 1:
        raise ValueError(
            f"{bio_id}: embedding column {column!r} has ragged widths {sorted(widths)}; expected one."
        )
    matrix = np.vstack(vectors)
    cols = [f"{column}_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=cols, index=series.index)


class LocalLoader:
    """Load a curated tabular file from disk and frame it with the curated target."""

    def __init__(
        self,
        *,
        embedding_column: str | None = None,
        group_column: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the loader.

        Parameters
        ----------
        embedding_column : str | None
            Name of a column holding a per-row comma-separated embedding string to
            explode into numeric features. ``None`` treats every non-target column as a
            plain feature.
        group_column : str | None
            Biological independence-unit column to remove from the features and
            preserve for grouped splitting.
        cache_dir : Path | None
            Directory for downloaded tables when no local table or directory override
            is supplied. Defaults to ``local_raw`` under the bio cache.
        """
        self.embedding_column = embedding_column
        self.group_column = group_column
        self.cache_dir = (
            cache_dir if cache_dir is not None else default_bio_cache_dir() / "local_raw"
        )

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        """Load the local file and return it as a :class:`BioRawDataset`."""
        if spec.target is None:
            raise ValueError(f"{spec.bio_id}: local loader needs a curated target.")
        path = _resolve_path(spec.fetch_id)
        if spec.download_url is not None:
            if (
                not path.is_file()
                and not (_LOCAL_ENV in os.environ and os.environ[_LOCAL_ENV])
                and not Path(spec.fetch_id).is_absolute()
            ):
                path = self.cache_dir / Path(spec.fetch_id).name
            _ensure_download(path, spec)
        if not path.exists():
            raise FileNotFoundError(
                f"{spec.bio_id}: local file {path} not found (fetch_id={spec.fetch_id!r})."
            )
        if path.suffix.lower() not in _TABLE_SUFFIXES:
            raise ValueError(
                f"{spec.bio_id}: unsupported local file type {path.suffix!r} (expected {_TABLE_SUFFIXES})."
            )

        df = _read_table(path)
        if spec.target not in df.columns:
            raise ValueError(
                f"{spec.bio_id}: curated target {spec.target!r} not in {path.name} "
                f"(columns: {list(df.columns[:8])}...).",
            )
        y = df[spec.target]
        group_column = self.group_column or spec.group_column
        if group_column is not None and group_column not in df.columns:
            raise ValueError(
                f"{spec.bio_id}: group_column {group_column!r} not in {path.name}; "
                "regenerate the dataset with biological group identifiers before benchmarking."
            )
        groups = df[group_column] if group_column is not None else None
        drop_cols = [spec.target] + ([group_column] if group_column is not None else [])
        features = df.drop(columns=drop_cols)

        if self.embedding_column is not None:
            if self.embedding_column not in features.columns:
                raise ValueError(
                    f"{spec.bio_id}: embedding_column {self.embedding_column!r} not in {path.name} "
                    f"(columns: {list(features.columns[:8])}...).",
                )
            exploded = _explode_embedding(
                features[self.embedding_column], self.embedding_column, spec.bio_id
            )
            others = features.drop(columns=[self.embedding_column]).reset_index(drop=True)
            X = (
                exploded.reset_index(drop=True)
                if others.empty
                else pd.concat([others, exploded.reset_index(drop=True)], axis=1)
            )
            y = y.reset_index(drop=True)
            if groups is not None:
                groups = groups.reset_index(drop=True)
        else:
            X = features

        problem_type = spec.problem_type or (
            "multiclass" if y.nunique(dropna=True) > 2 else "binary"
        )

        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=spec.license or "unknown (local file; set spec.license)",
            source_url=spec.download_url or str(path),
            citation=f"Local dataset {spec.bio_id} ({path.name}).",
            metadata={
                "local_path": str(path),
                "table_file": path.name,
                "embedding_column": self.embedding_column,
                "n_features": int(X.shape[1]),
                "target": spec.target,
            },
            groups=groups,
        )
