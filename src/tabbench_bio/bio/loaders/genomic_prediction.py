"""Frozen plant genomic-prediction panels from Azodi et al.

Each panel pairs a SNP genotype matrix (plant lines by markers) with one
quantitative phenotype.  The files are the CC0 Dryad release mirrored by its
Zenodo record; URLs and checksums are pinned in the registry so a benchmark run cannot
silently consume changed source data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.io_utils import download_file

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


PANELS = {"maize": "FT", "rice": "FT", "soy": "YLD", "switchgrass": "HT"}


class GenomicPredictionLoader:
    """Fetch one curated SNP-to-quantitative-trait genomic-prediction panel."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or (default_bio_cache_dir() / "gp_raw"))

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        crop = spec.fetch_id.strip().lower()
        if crop not in PANELS:
            raise ValueError(
                f"{spec.bio_id}: unsupported genomic-prediction crop {spec.fetch_id!r}; "
                f"curated panels are {sorted(PANELS)}."
            )
        trait = PANELS[crop]
        if spec.target != trait:
            raise ValueError(
                f"{spec.bio_id}: registry target={spec.target!r} disagrees with the curated "
                f"{crop} trait {trait!r}."
            )
        if spec.problem_type is not None and spec.problem_type != "regression":
            raise ValueError(
                f"{spec.bio_id}: genomic prediction requires problem_type='regression', "
                f"not {spec.problem_type!r}."
            )

        assert spec.download_files is not None, f"{spec.bio_id}: missing download configuration"
        genotype_file = f"{crop}_geno.csv"
        phenotype_file = f"{crop}_pheno.csv"
        assert set(spec.download_files) == {genotype_file, phenotype_file}, (
            f"{spec.bio_id}: expected genotype and phenotype source files"
        )
        genotype_path = download_file(
            destination=self.cache_dir / genotype_file,
            timeout=300,
            **spec.download_files[genotype_file],
        )
        phenotype_path = download_file(
            destination=self.cache_dir / phenotype_file,
            timeout=300,
            **spec.download_files[phenotype_file],
        )
        X = pd.read_csv(genotype_path, index_col=0)
        phenotypes = pd.read_csv(phenotype_path, index_col=0)

        if X.index.has_duplicates:
            raise ValueError(
                f"{spec.bio_id}: genotype matrix has duplicate plant-line identifiers."
            )
        if phenotypes.index.has_duplicates:
            raise ValueError(
                f"{spec.bio_id}: phenotype table has duplicate plant-line identifiers."
            )
        if trait not in phenotypes.columns:
            raise ValueError(f"{spec.bio_id}: phenotype table is missing curated trait {trait!r}.")
        non_numeric = X.select_dtypes(exclude="number").columns
        if len(non_numeric):
            raise ValueError(
                f"{spec.bio_id}: genotype matrix has non-numeric marker columns "
                f"{non_numeric[:5].tolist()}."
            )

        target = pd.to_numeric(phenotypes[trait], errors="coerce")
        keep = X.index.isin(target.index[target.notna()])
        X = X.loc[keep]
        y = target.reindex(X.index)
        if len(X) == 0:
            raise ValueError(f"{spec.bio_id}: no plant lines have both genotypes and phenotype.")
        if y.isna().any():
            raise ValueError(
                f"{spec.bio_id}: target alignment introduced missing phenotype values."
            )

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True).rename(trait)
        metadata = {
            "crop": crop,
            "trait": trait,
            "split_unit": "plant line",
            "source_doi": "10.5061/dryad.xksn02vb9",
            "source_record": spec.download_files[genotype_file]["url"].split("/files/", 1)[0],
            "genotype_file": genotype_file,
            "genotype_checksum": spec.download_files[genotype_file]["checksum"],
            "genotype_checksum_algorithm": spec.download_files[genotype_file]["algorithm"],
            "phenotype_file": phenotype_file,
            "phenotype_checksum": spec.download_files[phenotype_file]["checksum"],
            "phenotype_checksum_algorithm": spec.download_files[phenotype_file]["algorithm"],
            "n_features": int(X.shape[1]),
        }
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type="regression",
            license=spec.license or "CC0-1.0",
            source_url=spec.source_url,
            citation=spec.citation,
            metadata=metadata,
        )
