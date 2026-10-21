"""Stable source-file identities for the published TCGA cohorts."""

import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

from tabbench_bio.bio.loaders.base import BioRawDataset


@cache
def sample_orders() -> dict:
    record = json.loads((Path(__file__).parent / "data/tcga_sample_order.json").read_text())
    assert record["schema_version"] == 1
    return record["datasets"]


def sample_order_fingerprint(bio_id: str) -> str | None:
    orders = sample_orders()
    if bio_id not in orders:
        return None
    return hashlib.sha256(json.dumps(orders[bio_id]["row_ids"]).encode()).hexdigest()


def align_reference_rows(raw: BioRawDataset) -> BioRawDataset:
    orders = sample_orders()
    if raw.bio_id not in orders:
        return raw
    row_ids = orders[raw.bio_id]["row_ids"]
    assert len(row_ids) == len(set(row_ids)) and raw.X.index.is_unique
    assert set(raw.X.index) == set(row_ids), (
        f"{raw.bio_id}: source cohort differs from the published file-ID manifest"
    )
    assert (
        raw.y.index.equals(raw.X.index)
        and raw.groups is not None
        and raw.groups.index.equals(raw.X.index)
    ), f"{raw.bio_id}: source labels and patient IDs must align with file IDs"
    return replace(
        raw,
        X=raw.X.loc[row_ids],
        y=raw.y.loc[row_ids],
        groups=raw.groups.loc[row_ids],
        metadata={**raw.metadata, "sample_order_sha256": sample_order_fingerprint(raw.bio_id)},
    )
