"""Network-free tests for curated ChEMBL bioactivity endpoints."""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.loaders import chembl, get_loader
from tabbench_bio.bio.loaders.chembl import ChemblLoader, prepare_chembl_activities
from tabbench_bio.bio.loaders.tdc import FINGERPRINT_BITS


def _activity(
    smiles: str,
    value: float,
    *,
    relation: str = "=",
    validity: str | None = None,
    assay_type: str = "B",
) -> dict:
    return {
        "canonical_smiles": smiles,
        "pchembl_value": value,
        "standard_relation": relation,
        "data_validity_comment": validity,
        "assay_type": assay_type,
    }


def _spec() -> BioDatasetSpec:
    return BioDatasetSpec(
        bio_id="CHEMBL206",
        source="chembl",
        fetch_id="CHEMBL206",
        target="pchembl_value",
        problem_type="regression",
    )


def test_prepare_chembl_activities_filters_deduplicates_and_groups_scaffolds():
    activities = [
        _activity("CCO", 6.0),
        _activity("OCC", 8.0),
        _activity("c1ccccc1O", 7.0),
        _activity("c1ccccc1N", 7.5, assay_type="F"),
        _activity("CCN", 5.0, relation=">"),
        _activity("CCC", 4.0, validity="Outside typical range"),
    ]

    X, y, groups, metadata = prepare_chembl_activities(activities, bio_id="test")

    assert X.shape == (3, FINGERPRINT_BITS)
    assert sorted(y.tolist()) == [7.0, 7.0, 7.5]
    assert y.name == "pchembl_value"
    assert groups.str.startswith(("murcko:", "acyclic:")).all()
    assert metadata["duplicate_rows"] == 1
    assert metadata["non_exact_relation_dropped"] == 1
    assert metadata["flagged_validity_dropped"] == 1
    assert metadata["assay_type_counts"] == {"B": 5, "F": 1}


def test_chembl_loader_checks_release_and_fetches_curated_endpoint(monkeypatch):
    activities = [
        _activity("CCO", 6.0),
        _activity("c1ccccc1O", 7.0),
    ]

    def fake_get_json(url, *, params=None):
        if url == chembl.STATUS_URL:
            return {"chembl_db_version": chembl.EXPECTED_CHEMBL_VERSION}
        assert url == chembl.ACTIVITY_URL
        assert params["target_chembl_id"] == "CHEMBL206"
        return {
            "activities": activities,
            "page_meta": {"total_count": len(activities), "next": None},
        }

    monkeypatch.setattr(chembl, "_get_json", fake_get_json)
    raw = ChemblLoader().fetch(_spec())

    assert raw.X.shape == (2, FINGERPRINT_BITS)
    assert raw.groups is not None
    assert raw.metadata["chembl_db_version"] == chembl.EXPECTED_CHEMBL_VERSION
    assert raw.metadata["target_name"] == "Estrogen receptor"


def test_chembl_activity_pagination_follows_relative_next_link(monkeypatch):
    first = _activity("CCO", 6.0)
    second = _activity("CCN", 7.0)
    calls = []

    def fake_get_json(url, *, params=None):
        calls.append((url, params))
        if len(calls) == 1:
            return {
                "activities": [first],
                "page_meta": {
                    "total_count": 2,
                    "next": "/chembl/api/data/activity.json?limit=1000&offset=1000",
                },
            }
        return {
            "activities": [second],
            "page_meta": {"total_count": 2, "next": None},
        }

    monkeypatch.setattr(chembl, "_get_json", fake_get_json)
    activities, total = chembl._activities("CHEMBL206")

    assert activities == [first, second]
    assert total == 2
    assert calls[0][1]["target_chembl_id"] == "CHEMBL206"
    assert calls[1] == (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json?limit=1000&offset=1000",
        None,
    )


def test_chembl_rejects_unreviewed_release(monkeypatch):
    monkeypatch.setattr(
        chembl,
        "_get_json",
        lambda url, params=None: {"chembl_db_version": "ChEMBL_38"},
    )
    with pytest.raises(ValueError, match="pinned to 'ChEMBL_37'"):
        ChemblLoader().fetch(_spec())


def test_chembl_dispatch_is_lazy_and_curated():
    assert isinstance(get_loader(_spec()), ChemblLoader)
