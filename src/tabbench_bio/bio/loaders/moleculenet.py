"""Curated Tox21 assays with scaffold and identical-input separation."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.bio.loaders.tdc import prepare_tdc_table
from tabbench_bio.io_utils import download_file

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec

ENDPOINTS = {"SR-MMP", "NR-AhR"}


def scaffold_input_groups(X: pd.DataFrame, scaffolds: pd.Series) -> tuple[pd.Series, dict]:
    """Join scaffold families transitively when they share a fingerprint vector."""
    assert len(X) == len(scaffolds) and not scaffolds.isna().any()
    values = X.to_numpy(dtype=np.uint8)
    assert np.isin(X.to_numpy(), [0, 1]).all()
    scaffold_codes, scaffold_names = pd.factorize(scaffolds, sort=True)
    fingerprints, input_codes = np.unique(values, axis=0, return_inverse=True)
    n_rows, n_scaffolds = len(X), len(scaffold_names)
    rows = np.tile(np.arange(n_rows), 2)
    columns = np.concatenate([n_rows + scaffold_codes, n_rows + n_scaffolds + input_codes])
    size = n_rows + n_scaffolds + len(fingerprints)
    graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(size, size)).tocsr()
    _, labels = connected_components(graph, directed=False)
    groups = pd.Series(
        [f"molecular_component:{label}" for label in labels[:n_rows]],
        name="scaffold_input_component",
    )
    return groups, {
        "independence_groups": int(groups.nunique()),
        "unique_fingerprints": len(fingerprints),
        "duplicate_fingerprint_rows": n_rows - len(fingerprints),
        "grouping": "Connected components of Bemis-Murcko scaffolds and identical 2048-bit inputs",
    }


def prepare_tox21_table(
    table: pd.DataFrame, *, endpoint: str, bio_id: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Select one measured assay; missing labels never become inactive labels."""
    assert endpoint in ENDPOINTS, f"Unsupported Tox21 assay: {endpoint}"
    assert {"smiles", endpoint} <= set(table.columns)
    target = pd.to_numeric(table[endpoint], errors="raise")
    assert set(target.dropna().unique()) == {0, 1}, f"{bio_id}: expected binary assay labels"
    labelled = table.loc[target.notna(), ["smiles"]].rename(columns={"smiles": "Drug"})
    labelled["Y"] = target.loc[target.notna()]
    X, y, scaffolds, metadata = prepare_tdc_table(labelled, bio_id=bio_id, problem_type="binary")
    groups, grouping = scaffold_input_groups(X, scaffolds)
    y.name = endpoint
    metadata.update(grouping)
    metadata.update(
        source_rows=len(table),
        unlabelled_rows_dropped=int(target.isna().sum()),
        endpoint=endpoint,
        endpoint_family="Tox21",
        rdkit_version=version("rdkit"),
        row_order="canonical parent SMILES, lexicographic",
    )
    return X, y, groups, metadata


class MoleculeNetLoader:
    """Download the pinned original Tox21 table and construct one curated endpoint."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or (default_bio_cache_dir() / "moleculenet_raw"))

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        assert spec.fetch_id in ENDPOINTS and spec.target == spec.fetch_id, (
            f"{spec.bio_id}: fetch_id and target must name the same curated Tox21 assay"
        )
        assert spec.problem_type == "binary"
        assert spec.download_url and spec.download_sha256, (
            f"{spec.bio_id}: missing download configuration"
        )
        path = download_file(
            spec.download_url,
            self.cache_dir / "tox21.csv.gz",
            checksum=spec.download_sha256,
        )
        X, y, groups, metadata = prepare_tox21_table(
            pd.read_csv(path), endpoint=spec.fetch_id, bio_id=spec.bio_id
        )
        metadata["source_sha256"] = spec.download_sha256
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            groups=groups,
            problem_type="binary",
            license=spec.license,
            source_url=spec.source_url,
            citation=spec.citation,
            metadata=metadata,
        )
