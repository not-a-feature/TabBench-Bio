"""Frozen plant genomic-prediction panels from Azodi et al.

Each panel pairs a SNP genotype matrix (plant lines by markers) with one
quantitative phenotype.  The files are the CC0 Dryad release mirrored by its
Zenodo record; byte sizes and checksums are pinned so a benchmark run cannot
silently consume changed source data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


ZENODO_RECORD = "https://zenodo.org/api/records/4980429"
ZENODO_FILE_URL = ZENODO_RECORD + "/files/{filename}/content"
SOURCE_URL = "https://doi.org/10.5061/dryad.xksn02vb9"
CITATION = (
    "Azodi CB et al. (2019), Benchmarking Parametric and Machine Learning Models "
    "for Genomic Prediction of Complex Traits, G3: Genes|Genomes|Genetics, "
    "doi:10.1534/g3.119.400498."
)


@dataclass(frozen=True)
class SourceFile:
    """Pinned identity of one file in the public Zenodo record."""

    filename: str
    n_bytes: int
    md5: str


@dataclass(frozen=True)
class GenomicPanel:
    """The curated phenotype and two source files for one crop panel."""

    trait: str
    genotype: SourceFile
    phenotype: SourceFile


PANELS: dict[str, GenomicPanel] = {
    "maize": GenomicPanel(
        trait="FT",
        genotype=SourceFile("maize_geno.csv", 217_174_436, "2e2d6fbcbe478105077ed0063fe2401a"),
        phenotype=SourceFile("maize_pheno.csv", 23_462, "1f8eadf2595faec317af5665f8be4afc"),
    ),
    "rice": GenomicPanel(
        trait="FT",
        genotype=SourceFile("rice_geno.csv", 50_527_158, "3584b09824b1b4cfa8d44869b91e3fd4"),
        phenotype=SourceFile("rice_pheno.csv", 19_797, "f6f14948fc9cbede9d08cc3edb78c2d7"),
    ),
    "soy": GenomicPanel(
        trait="YLD",
        genotype=SourceFile("soy_geno.csv", 50_038_368, "211c2c45f1d38d479471cf803a4dfa3f"),
        phenotype=SourceFile("soy_pheno.csv", 327_918, "bdfaa0ceacf42b0197cbb8ee8347f56c"),
    ),
    "switchgrass": GenomicPanel(
        trait="HT",
        genotype=SourceFile(
            "switchgrass_geno.csv", 237_057_149, "99d6c544cab1d93073c5e86a6855a0f0"
        ),
        phenotype=SourceFile("switchgrass_pheno.csv", 33_780, "5d2fbd31a9fde9edffec23f1d13f34d7"),
    ),
}


def _md5(path: Path) -> str:
    """Calculate a source-identity checksum without loading a large panel into RAM."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(source: SourceFile, destination: Path) -> Path:
    """Download and verify one source file, using an atomic cache write."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = ZENODO_FILE_URL.format(filename=source.filename)
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            partial.replace(destination)
        finally:
            if partial.exists():
                partial.unlink()

    n_bytes = destination.stat().st_size
    checksum = _md5(destination)
    if n_bytes != source.n_bytes or checksum != source.md5:
        raise ValueError(
            f"{source.filename}: cached/downloaded genomic-prediction file failed identity "
            f"check (bytes={n_bytes}, md5={checksum}); expected bytes={source.n_bytes}, "
            f"md5={source.md5}. Remove only this cache file and retry."
        )
    return destination


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
        panel = PANELS[crop]
        if spec.target != panel.trait:
            raise ValueError(
                f"{spec.bio_id}: registry target={spec.target!r} disagrees with the curated "
                f"{crop} trait {panel.trait!r}."
            )
        if spec.problem_type is not None and spec.problem_type != "regression":
            raise ValueError(
                f"{spec.bio_id}: genomic prediction requires problem_type='regression', "
                f"not {spec.problem_type!r}."
            )

        genotype_path = _download(panel.genotype, self.cache_dir / panel.genotype.filename)
        phenotype_path = _download(panel.phenotype, self.cache_dir / panel.phenotype.filename)
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
        if panel.trait not in phenotypes.columns:
            raise ValueError(
                f"{spec.bio_id}: phenotype table is missing curated trait {panel.trait!r}."
            )
        non_numeric = X.select_dtypes(exclude="number").columns
        if len(non_numeric):
            raise ValueError(
                f"{spec.bio_id}: genotype matrix has non-numeric marker columns "
                f"{non_numeric[:5].tolist()}."
            )

        target = pd.to_numeric(phenotypes[panel.trait], errors="coerce")
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
        y = y.reset_index(drop=True).rename(panel.trait)
        metadata = {
            "crop": crop,
            "trait": panel.trait,
            "split_unit": "plant line",
            "source_doi": "10.5061/dryad.xksn02vb9",
            "source_record": ZENODO_RECORD,
            "genotype_file": panel.genotype.filename,
            "genotype_md5": panel.genotype.md5,
            "phenotype_file": panel.phenotype.filename,
            "phenotype_md5": panel.phenotype.md5,
            "n_features": int(X.shape[1]),
        }
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type="regression",
            license=spec.license or "CC0-1.0",
            source_url=SOURCE_URL,
            citation=CITATION,
            metadata=metadata,
        )
