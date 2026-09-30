"""Identity of a dataset specification and its loader behavior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from tabbench_bio.bio.loaders.metagenomics import COHORT_VERSION
from tabbench_bio.bio.loaders.mgnify import GROUPING_VERSION, grouping_digest

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec
    from tabbench_bio.bio.loaders.base import BioRawDataset

# Increment when loader behavior changes the assembled matrix, labels or groups.
LOADER_CACHE_VERSION = 1
DATA_FIELDS = {
    "source",
    "fetch_id",
    "target",
    "problem_type",
    "data_file",
    "embedding_column",
    "group_column",
    "source_max_features",
    "download_url",
    "download_sha256",
    "train_prevalence_filter",
}


def spec_fingerprint(spec: BioDatasetSpec) -> str:
    payload = {
        "spec": {
            key: value
            for key, value in asdict(spec).items()
            if key in DATA_FIELDS and value is not None
        },
        "loader_version": LOADER_CACHE_VERSION,
    }
    if spec.source == "metagenomics":
        payload["cohort_version"] = COHORT_VERSION
    if spec.source == "mgnify":
        payload.update(grouping_version=GROUPING_VERSION, grouping_sha256=grouping_digest())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_cached_spec(raw: BioRawDataset, spec: BioDatasetSpec) -> None:
    assert raw.bio_id == spec.bio_id and (
        "spec_sha256" in raw.metadata and raw.metadata["spec_sha256"] == spec_fingerprint(spec)
    ), (
        f"{spec.bio_id}: stale dataset cache; refetch with force_refetch=True, or explicitly adopt an unversioned cache with cache-adopt"
    )
