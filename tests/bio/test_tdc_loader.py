"""Network-free tests for molecular preparation and TDC dispatch."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from tabbench_bio import io_utils
from tabbench_bio.bio.datasets import BioDatasetSpec, get_spec
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.bio.loaders.tdc import FINGERPRINT_BITS, TdcLoader, prepare_tdc_table


def test_prepare_tdc_table_deduplicates_and_builds_scaffold_groups():
    table = pd.DataFrame(
        {
            "Drug": ["CCO", "OCC", "c1ccccc1O", "c1ccccc1N"],
            "Y": [0, 0, 1, 1],
        }
    )
    X, y, groups, metadata = prepare_tdc_table(
        table,
        bio_id="test",
        problem_type="binary",
    )
    assert X.shape == (3, FINGERPRINT_BITS)
    assert y.tolist() == [0, 1, 1]
    assert len(groups) == 3
    assert groups.iloc[1] == groups.iloc[2]  # phenyl scaffold
    assert metadata["duplicate_rows"] == 1
    assert metadata["invalid_smiles"] == 0


def test_prepare_tdc_table_rejects_many_invalid_smiles():
    table = pd.DataFrame({"Drug": ["not a smiles", "CCO"], "Y": [0, 1]})
    with pytest.raises(ValueError, match="SMILES are invalid"):
        prepare_tdc_table(table, bio_id="bad", problem_type="binary")


def test_prepare_tdc_table_accepts_legacy_x_smiles_column():
    table = pd.DataFrame({"X": ["CCO", "CCN"], "Y": [1.2, 2.3]})
    X, y, _, _ = prepare_tdc_table(table, bio_id="legacy", problem_type="regression")
    assert X.shape == (2, FINGERPRINT_BITS)
    assert y.tolist() == [2.3, 1.2]  # deterministic canonical-SMILES ordering


def test_tdc_dispatch_is_lazy_and_curated():
    spec = BioDatasetSpec(
        bio_id="TDC-BBB-Martins",
        source="tdc",
        fetch_id="bbb_martins",
        target="Y",
        problem_type="binary",
    )
    assert isinstance(get_loader(spec), TdcLoader)


@pytest.mark.parametrize(
    "bio_id",
    [
        "TDC-BBB-Martins",
        "TDC-AMES",
        "TDC-CYP3A4-Veith",
        "TDC-AqSolDB",
        "TDC-Lipophilicity-AstraZeneca",
        "TDC-LD50-Zhu",
    ],
)
def test_tdc_fetch_uses_registry_source_and_shared_cache(tmp_path, monkeypatch, bio_id):
    payload = b"Drug\tY\nCCO\t0\nCCN\t1\n"
    spec = replace(
        get_spec(bio_id),
        download_url="https://example.org/mirror/endpoint.tab",
        download_sha256=hashlib.sha256(payload).hexdigest(),
    )
    calls = []

    def download(url, timeout):
        calls.append(url)
        assert timeout == 180
        return io.BytesIO(payload)

    monkeypatch.setattr(io_utils, "urlopen", download)
    loader = TdcLoader(cache_dir=tmp_path)
    raw = loader.fetch(spec)
    reused = loader.fetch(spec)
    assert calls == [spec.download_url]
    assert reused.y.equals(raw.y)
    assert raw.X.shape == (2, FINGERPRINT_BITS)
    assert raw.problem_type == spec.problem_type
    assert raw.source_url == spec.source_url and raw.citation == spec.citation
    assert raw.metadata["source_sha256"] == spec.download_sha256
