"""Turn a fetched :class:`BioRawDataset` into a native :class:`~tabbench_bio.dataset.Dataset`.

A gene-expression / feature matrix is exactly the wide-matrix shape the benchmark
consumes, so a bio dataset becomes a :class:`Dataset` (``features`` = the matrix,
``feature_names`` = the real gene/probe identifiers, ``targets`` = labels) with no
further changes downstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir, load_cached_raw, save_cached_raw
from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.dataset import Dataset, DatasetInfo, TaskType

if TYPE_CHECKING:
    from tabbench_bio.bio.loaders.base import BioRawDataset

logger = logging.getLogger(__name__)

#: Benchmark-wide default feature cap. Datasets wider than this (e.g. TCGA RNA-seq at
#: ~60660 genes) are truncated to a uniform random subset of columns to bound compute time
#: while preserving the HDLSS (high-dim, low-sample) character of the benchmark. The cap
#: is configurable per run (``bio_max_features`` in the config) and per dataset
#: (``max_features`` in the registry); set it to ``None`` to keep all features.
DEFAULT_MAX_FEATURES = 30_000


def cap_features(
    X: pd.DataFrame,
    *,
    max_features: int | None = DEFAULT_MAX_FEATURES,
    random_state: int = 0,
) -> pd.DataFrame:
    """Truncate a wide feature matrix to a random subset of ``max_features`` columns.

    No-op when ``max_features`` is ``None`` or ``X`` already has ``<= max_features``
    columns. Otherwise ``max_features`` columns are drawn uniformly at random (without
    replacement, seeded by ``random_state`` for reproducibility); survivors retain their
    original left-to-right order so the matrix stays readable.

    The draw is uniform over columns — it does not rank by variance. On expression data
    (TPM, microarray intensity) variance scales with mean expression, so a variance rank
    would preferentially retain highly-expressed genes and bias the retained set by
    modality; a random subset is unbiased with respect to expression level and, depending
    only on column indices rather than values, carries no risk of leakage.
    """
    if max_features is None or X.shape[1] <= max_features:
        return X
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.permutation(X.shape[1])[:max_features])
    return X.iloc[:, idx]


def _numeric_features(raw: BioRawDataset) -> pd.DataFrame:
    """Keep only the numeric feature columns (bio matrices are numeric)."""
    numeric = raw.X.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        raise ValueError(f"{raw.bio_id}: no numeric feature columns to build a matrix from.")
    if numeric.shape[1] < raw.X.shape[1]:
        logger.warning(
            "%s: dropped %d non-numeric feature column(s); kept %d numeric.",
            raw.bio_id,
            raw.X.shape[1] - numeric.shape[1],
            numeric.shape[1],
        )
    return numeric


def bio_raw_to_dataset(
    raw: BioRawDataset, *, max_features: int | None = DEFAULT_MAX_FEATURES
) -> Dataset:
    """Adapt a :class:`BioRawDataset` to a native :class:`~tabbench_bio.dataset.Dataset`.

    Parameters
    ----------
    raw : BioRawDataset
        The fetched dataset.
    max_features : int | None
        Feature cap (see :func:`cap_features`). ``None`` keeps all features.
    """
    is_classification = raw.problem_type in ("binary", "multiclass")

    X = _numeric_features(raw)
    y = raw.y
    groups = getattr(raw, "groups", None)
    # Loader-layer guarantee: drop rows whose target is missing — they cannot be
    # supervised, so no adapted Dataset should carry them. (Done here, the single
    # chokepoint all source loaders flow through, rather than per loader.)
    target_missing = y.isna().to_numpy()
    if target_missing.any():
        keep = ~target_missing
        logger.info(
            "%s: dropped %d row(s) with missing target (%d remain).",
            raw.bio_id,
            int(target_missing.sum()),
            int(keep.sum()),
        )
        X = X.iloc[keep]
        y = y.iloc[keep]
        if groups is not None:
            groups = groups.iloc[keep]

    capped = cap_features(X, max_features=max_features)
    if capped.shape[1] < X.shape[1]:
        logger.info(
            "%s: capped features %d -> %d (max_features=%s).",
            raw.bio_id,
            X.shape[1],
            capped.shape[1],
            max_features,
        )

    features = capped.to_numpy(dtype=np.float32)
    feature_names = [str(c) for c in capped.columns]  # real gene/probe identifiers

    targets = y.to_numpy()
    group_values = groups.to_numpy() if groups is not None else None
    target_name = y.name or raw.metadata.get("target") or "target"
    task_type = TaskType.Classification if is_classification else TaskType.Regression

    info = DatasetInfo(
        id=raw.bio_id,
        name=raw.bio_id,
        task_type=task_type,
        metadata=dict(raw.metadata),
    )
    return Dataset(
        features=features,
        targets=targets,
        feature_names=feature_names,
        target_names=[str(target_name)],
        info=info,
        groups=group_values,
    )


def load_bio_dataset(
    bio_id: str,
    *,
    cache_dir: str | None = None,
    force_refetch: bool = False,
) -> BioRawDataset:
    """Return the fetched :class:`BioRawDataset` for ``bio_id``, using the unified cache.

    On a cache miss the dataset is fetched via its source loader and cached so later
    runs skip both the download and the re-assembly.
    """
    root = cache_dir or str(default_bio_cache_dir())
    spec = get_spec(bio_id)

    if not force_refetch:
        cached = load_cached_raw(root, bio_id)
        # Old caches predate biological group metadata. Do not silently reuse them for
        # sources whose rows can contain repeated measurements of one independence unit.
        needs_groups = (
            spec.source in {"tcga", "tdc", "chembl", "geo_matrix", "fusionai"}
            or spec.group_column is not None
        )
        if cached is not None and (not needs_groups or getattr(cached, "groups", None) is not None):
            return cached

    if not spec.enabled:
        raise ValueError(
            f"{bio_id}: dataset is disabled in the registry; enable it before running."
        )
    if not spec.is_curated:
        raise ValueError(
            f"{bio_id}: target/problem_type not curated; set them in the registry before running."
        )

    raw = get_loader(spec, cache_dir=root).fetch(spec)
    save_cached_raw(root, raw)
    return raw


def load_bio_as_dataset(
    bio_id: str,
    *,
    cache_dir: str | None = None,
    max_features: int | None = DEFAULT_MAX_FEATURES,
) -> Dataset:
    """Fetch (cached) and adapt a bio dataset to a native :class:`~tabbench_bio.dataset.Dataset`.

    The per-dataset registry ``max_features`` override takes precedence over the
    ``max_features`` argument when set.
    """
    raw = load_bio_dataset(bio_id, cache_dir=cache_dir)
    spec = get_spec(bio_id)
    cap = spec.max_features if spec.max_features is not None else max_features
    return bio_raw_to_dataset(raw, max_features=cap)
