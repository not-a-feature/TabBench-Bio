"""Network-free tests for the curated genomic-prediction panels."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace

import pytest

from tabbench_bio import io_utils
from tabbench_bio.bio.datasets import BioDatasetSpec, get_spec
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.bio.loaders.genomic_prediction import GenomicPredictionLoader


def _spec() -> BioDatasetSpec:
    return get_spec("gp-maize-FT")


@pytest.fixture
def sources():
    payloads = {
        "maize_geno.csv": b",marker_1,marker_2\nline_b,0,1\nline_a,1,0\nline_c,-1,1\n",
        "maize_pheno.csv": b",FT,unused\nline_c,3,4\nline_a,1,5\nline_b,,6\n",
    }
    spec = replace(
        _spec(),
        download_files={
            name: {
                "url": f"https://example.org/{name}",
                "checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                "algorithm": "md5",
            }
            for name, payload in payloads.items()
        },
    )
    return spec, payloads


def test_genomic_prediction_loader_aligns_lines_and_drops_missing_targets(
    tmp_path, monkeypatch, sources
):
    spec, payloads = sources
    calls = []

    def download(url, timeout):
        calls.append(url)
        assert timeout == 300
        return io.BytesIO(payloads[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(io_utils, "urlopen", download)
    loader = GenomicPredictionLoader(cache_dir=tmp_path)
    raw = loader.fetch(spec)
    reused = loader.fetch(spec)
    assert calls == [asset["url"] for asset in spec.download_files.values()]
    assert reused.y.equals(raw.y)

    assert raw.X.to_dict(orient="list") == {"marker_1": [1, -1], "marker_2": [0, 1]}
    assert raw.y.tolist() == [1.0, 3.0]
    assert raw.y.name == "FT"
    assert raw.groups is None
    assert raw.metadata["split_unit"] == "plant line"
    assert raw.source_url == spec.source_url and raw.citation == spec.citation
    assert raw.metadata["genotype_checksum"] == spec.download_files["maize_geno.csv"]["checksum"]


@pytest.mark.parametrize("cached", [False, True])
def test_genomic_prediction_rejects_corrupt_source(tmp_path, monkeypatch, sources, cached):
    spec, payloads = sources
    genotype = tmp_path / "maize_geno.csv"
    phenotype = tmp_path / "maize_pheno.csv"
    genotype.write_bytes(payloads[genotype.name])
    if cached:
        phenotype.write_bytes(b"corrupt")
    monkeypatch.setattr(io_utils, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
    with pytest.raises(AssertionError, match="MD5 mismatch"):
        GenomicPredictionLoader(cache_dir=tmp_path).fetch(spec)
    assert phenotype.exists() == cached
    assert {p.name for p in tmp_path.iterdir()} == (
        {genotype.name, phenotype.name} if cached else {genotype.name}
    )


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
