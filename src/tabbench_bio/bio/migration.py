"""Explicit adoption of assembled caches that predate registry fingerprints."""

import hashlib
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tabbench_bio.bio.cache import load_cached_raw, save_cached_raw
from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.bio.fingerprint import spec_fingerprint
from tabbench_bio.bio.loaders.metagenomics import COHORT_VERSION
from tabbench_bio.bio.loaders.mgnify import GROUPING_VERSION, grouping_digest

logger = logging.getLogger(__name__)


def adopt_cached_dataset(
    bio_id: str, root: str | Path, *, reason: str, manifest: Path | None = None
) -> Path:
    """Stamp an unversioned cache after the caller verifies its source and task."""
    assert reason.strip(), "Cache adoption requires a reason documenting the provenance check"
    spec = get_spec(bio_id)
    raw = load_cached_raw(root, bio_id)
    assert raw is not None and raw.bio_id == bio_id, f"Missing cache: {bio_id}"
    assert "spec_sha256" not in raw.metadata, (
        "Only unversioned caches can be adopted; refetch changed specs"
    )
    assert raw.problem_type == spec.problem_type, "Cached task type differs from the registry"
    needs_groups = spec.group_column is not None or spec.source in {
        "tcga",
        "tdc",
        "chembl",
        "geo_matrix",
        "fusionai",
        "mgnify",
        "metagenomics",
    }
    if needs_groups:
        assert raw.groups is not None, "Cache predates required biological groups; refetch it"
    if spec.source == "metagenomics":
        assert raw.metadata["cohort_version"] == COHORT_VERSION, "Obsolete cohort cache"
    if spec.source == "mgnify":
        assert raw.metadata["grouping_version"] == GROUPING_VERSION
        assert raw.metadata["grouping_sha256"] == grouping_digest()
    audit = {"reason": reason, "timestamp": datetime.now(UTC).isoformat()}
    if manifest is not None:
        payload = manifest.read_bytes()
        frozen = json.loads(payload)
        assert frozen["schema_version"] == 1
        key = f"{bio_id}_0"
        target = pd.DataFrame({"target": raw.y.dropna().to_numpy()})
        checked = 0
        for identity, expected in frozen["target_fingerprints"].items():
            dataset, repeat = json.loads(identity)
            if dataset != key:
                continue
            indices = sorted(
                {
                    idx
                    for unit, record in frozen["units"].items()
                    if json.loads(unit)[1] == key
                    and json.loads(unit)[0] // frozen["cv_folds"] == repeat
                    for idx in record["test_indices"]
                }
            )
            actual = hashlib.sha256(
                target.loc[indices].to_csv(index=True, lineterminator="\n").encode()
            ).hexdigest()
            assert actual == expected, f"Frozen target fingerprint differs: {identity}"
            checked += 1
        assert checked, f"No frozen target fingerprint for {key}"
        audit["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    adopted = replace(
        raw,
        metadata={**raw.metadata, "spec_sha256": spec_fingerprint(spec), "cache_adoption": audit},
    )
    path = save_cached_raw(root, adopted)
    logger.warning(
        "Adopted existing cache %s: %s (manifest checked: %s)", bio_id, reason, manifest is not None
    )
    return path
