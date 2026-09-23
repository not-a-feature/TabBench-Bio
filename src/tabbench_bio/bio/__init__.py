"""HDLSS biological datasets for the TabBench-Bio benchmark.

This subpackage fetches and curates high-dimensional, low-sample-size (HDLSS)
biological datasets (GEO, TCGA, Kaggle, OpenML).  The genuinely source-specific
work lives in :mod:`~tabbench_bio.bio.loaders` and the configurable registry
:mod:`~tabbench_bio.bio.datasets`; everything downstream (cross-validation,
AutoGluon fitting, metrics, leaderboard) is the standard pipeline, reached via the
:class:`~tabbench_bio.bio.loaders.base.BioRawDataset` ->
:class:`~tabbench_bio.dataset.Dataset` adapter.

Nothing heavy (the per-source fetch libraries) is imported at package import time,
so ``import tabbench_bio.bio`` stays cheap and dependency-light.
"""

from __future__ import annotations

from tabbench_bio.bio.adapter import (
    DEFAULT_MAX_FEATURES,
    bio_raw_to_dataset,
    cap_features,
    load_bio_as_dataset,
    load_bio_dataset,
)
from tabbench_bio.bio.datasets import (
    BIO_DATASETS,
    BioDatasetSpec,
    bio_dataset_names,
    get_spec,
    is_bio_dataset,
    reload,
    runnable_specs,
)

__all__ = [
    "BIO_DATASETS",
    "DEFAULT_MAX_FEATURES",
    "BioDatasetSpec",
    "bio_dataset_names",
    "bio_raw_to_dataset",
    "cap_features",
    "get_spec",
    "is_bio_dataset",
    "load_bio_as_dataset",
    "load_bio_dataset",
    "reload",
    "runnable_specs",
]
