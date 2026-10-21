"""Network-free unit tests for the TCGA loader's parsing helpers."""

from __future__ import annotations

import io
import tarfile
from types import SimpleNamespace

import pandas as pd
import pytest

from tabbench_bio.bio.loaders.tcga import (
    _CATEGORY_CONFIG,
    TcgaLoader,
    parse_star_counts,
    sample_type_to_binary,
)

# A minimal STAR-Counts file: leading `# gene-model` comment, header row, the four
# `N_*` alignment-summary rows, then three gene rows.
_STAR_COUNTS_SAMPLE = (
    "# gene-model: GENCODE v36\n"
    "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second"
    "\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
    "N_unmapped\t\t\t1000\t1000\t1000\t\t\t\n"
    "N_multimapping\t\t\t2000\t2000\t2000\t\t\t\n"
    "N_noFeature\t\t\t3000\t3000\t3000\t\t\t\n"
    "N_ambiguous\t\t\t4000\t4000\t4000\t\t\t\n"
    "ENSG00000000003.15\tTSPAN6\tprotein_coding\t500\t250\t250\t12.5\t8.0\t9.0\n"
    "ENSG00000000005.6\tTNMD\tprotein_coding\t0\t0\t0\t0.0\t0.0\t0.0\n"
    "ENSG00000000419.13\tDPM1\tprotein_coding\t800\t400\t400\t30.25\t20.0\t22.0\n"
)


def test_parse_star_counts_drops_comment_and_summary_rows():
    s = parse_star_counts(
        _STAR_COUNTS_SAMPLE, value_column="tpm_unstranded", index_column="gene_id"
    )
    assert list(s.index) == ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13"]
    assert s.dtype == "float32"
    assert s["ENSG00000000003.15"] == 12.5
    assert s["ENSG00000000419.13"] == 30.25
    assert not any(g.startswith("N_") for g in s.index)


def test_sample_type_to_binary():
    assert sample_type_to_binary("Solid Tissue Normal") == "Normal"
    assert sample_type_to_binary("Blood Derived Normal") == "Normal"
    assert sample_type_to_binary("Primary Tumor") == "Tumor"
    assert sample_type_to_binary("Metastatic") == "Tumor"
    assert sample_type_to_binary("Additional - New Primary") == "Tumor"


def test_query_files_returns_patient_groups(monkeypatch):
    payload = {
        "data": {
            "hits": [
                {
                    "file_id": "file-tumor",
                    "cases": [
                        {
                            "submitter_id": "TCGA-XX-0001",
                            "samples": [{"sample_type": "Primary Tumor"}],
                        }
                    ],
                },
                {
                    "file_id": "file-normal",
                    "cases": [
                        {
                            "submitter_id": "TCGA-XX-0001",
                            "samples": [{"sample_type": "Solid Tissue Normal"}],
                        }
                    ],
                },
            ],
            "pagination": {"total": 2},
        }
    }

    class Response:
        def json(self):
            return payload

    loader = TcgaLoader()
    monkeypatch.setattr(loader, "_post_with_retries", lambda *args, **kwargs: Response())
    sample_types, patients = loader._query_files(
        object(),
        project="TCGA-TEST",
        cfg=_CATEGORY_CONFIG["Gene-Expression-Quantification"],
    )

    assert sample_types == {
        "file-tumor": "Primary Tumor",
        "file-normal": "Solid Tissue Normal",
    }
    assert patients == {
        "file-tumor": "TCGA-XX-0001",
        "file-normal": "TCGA-XX-0001",
    }


def _archive():
    stream = io.BytesIO()
    data = _STAR_COUNTS_SAMPLE.encode()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in ("file1", "file2"):
            member = tarfile.TarInfo(f"{name}/counts.tsv")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def test_truncated_archive_is_retried_before_a_batch_is_published(tmp_path, monkeypatch):
    payload = _archive()
    responses = iter([payload[: len(payload) // 2], payload])
    loader = TcgaLoader(cache_dir=tmp_path, batch_size=2, max_retries=2)
    monkeypatch.setattr(
        loader, "_post_with_retries", lambda *a, **kw: SimpleNamespace(content=next(responses))
    )
    monkeypatch.setattr("tabbench_bio.bio.loaders.tcga.time.sleep", lambda _: None)
    matrix = loader._download_matrix(
        object(),
        file_ids=["file1", "file2"],
        cfg=_CATEGORY_CONFIG["Gene-Expression-Quantification"],
    )
    assert list(matrix.index) == ["file1", "file2"]
    assert matrix.loc["file1", "ENSG00000000003.15"] == 12.5
    assert len(list((tmp_path / "batches").glob("*.pkl"))) == 1


def test_completed_batches_survive_interruption_and_are_reused(tmp_path, monkeypatch):
    loader = TcgaLoader(cache_dir=tmp_path, batch_size=2)
    calls = []

    def interrupted(session, *, file_ids, cfg):
        calls.append(file_ids)
        if file_ids == ["file3"]:
            raise ConnectionError("interrupted transfer")
        return {name: pd.Series([1.0], index=["gene"], dtype="float32") for name in file_ids}

    monkeypatch.setattr(loader, "_download_batch", interrupted)
    cfg = _CATEGORY_CONFIG["Gene-Expression-Quantification"]
    with pytest.raises(ConnectionError, match="interrupted"):
        loader._download_matrix(object(), file_ids=["file1", "file2", "file3"], cfg=cfg)
    assert len(list((tmp_path / "batches").glob("*.pkl"))) == 1

    def resumed(session, *, file_ids, cfg):
        calls.append(file_ids)
        assert file_ids == ["file3"], "Completed files must come from the source cache"
        return {"file3": pd.Series([1.0], index=["gene"], dtype="float32")}

    monkeypatch.setattr(loader, "_download_batch", resumed)
    matrix = loader._download_matrix(object(), file_ids=["file1", "file2", "file3"], cfg=cfg)
    assert list(matrix.index) == ["file1", "file2", "file3"]
    assert calls == [["file1", "file2"], ["file3"], ["file3"]]
