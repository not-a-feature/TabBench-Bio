"""Network-free tests for the curated genomic-prediction panels."""

from __future__ import annotations

import pandas as pd
import pytest

from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.loaders import genomic_prediction as gp
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.bio.loaders.genomic_prediction import GenomicPredictionLoader


def _spec() -> BioDatasetSpec:
    return BioDatasetSpec(
        bio_id="gp-maize-FT",
        source="gp",
        fetch_id="maize",
        target="FT",
        problem_type="regression",
    )


def test_genomic_prediction_loader_aligns_lines_and_drops_missing_targets(tmp_path, monkeypatch):
    genotype = tmp_path / "maize_geno.csv"
    phenotype = tmp_path / "maize_pheno.csv"
    pd.DataFrame(
        {"marker_1": [0, 1, -1], "marker_2": [1, 0, 1]},
        index=["line_b", "line_a", "line_c"],
    ).to_csv(genotype)
    pd.DataFrame(
        {"FT": [3.0, 1.0, None], "unused": [4, 5, 6]},
        index=["line_c", "line_a", "line_b"],
    ).to_csv(phenotype)
    paths = {"maize_geno.csv": genotype, "maize_pheno.csv": phenotype}
    monkeypatch.setattr(gp, "_download", lambda source, destination: paths[source.filename])

    raw = GenomicPredictionLoader(cache_dir=tmp_path / "cache").fetch(_spec())

    assert raw.X.to_dict(orient="list") == {"marker_1": [1, -1], "marker_2": [0, 1]}
    assert raw.y.tolist() == [1.0, 3.0]
    assert raw.y.name == "FT"
    assert raw.groups is None
    assert raw.metadata["split_unit"] == "plant line"


def test_genomic_prediction_dispatch_is_lazy_and_curated():
    assert isinstance(get_loader(_spec()), GenomicPredictionLoader)


def test_genomic_prediction_rejects_registry_target_mismatch(tmp_path):
    spec = BioDatasetSpec(
        bio_id="bad",
        source="gp",
        fetch_id="maize",
        target="YLD",
        problem_type="regression",
    )
    loader = GenomicPredictionLoader(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="disagrees with the curated maize trait"):
        loader.fetch(spec)
