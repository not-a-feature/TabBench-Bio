"""Source identity, clinical-field exclusion and discovery/validation separation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.bio.fingerprint import spec_fingerprint
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.bio.loaders import metabolomics as mw


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "Samples": [
                "7122",
                "7147",
                "7500",
                "7600",
                "8000",
                "UMCGIBD001",
                "UMCGIBD002",
                "LLDeep_01",
            ],
            "group": [
                "Diagnosis:CD",
                "Diagnosis:UC",
                "Diagnosis:CD",
                "Diagnosis:UC",
                "Diagnosis:Control",
                "Diagnosis:UC",
                "Diagnosis:CD",
                "Diagnosis:Control",
            ],
            "123.45_1.23": [np.nan, 2, 3, 4, 5, 6, 7, 8],
            "234.56_2.34": [np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8],
        }
    )


def test_st001000_excludes_controls_validation_and_ids_without_fitting(table):
    X, y, groups, metadata = mw.prepare_st001000(table)
    assert X.shape == (4, 2)
    assert X.columns.tolist() == ["123.45_1.23", "234.56_2.34"]
    assert y.tolist() == [1, 0, 1, 0]
    assert groups.tolist() == ["PRISM|7122", "PRISM|7147", "PRISM|7500", "PRISM|7600"]
    assert X.iloc[:, 1].isna().all()  # retain even globally empty discovery features
    assert np.isclose(X.iloc[1, 0], np.log1p(2))
    assert metadata["class_names"] == {"0": "UC", "1": "CD"}
    external, external_y, external_groups, _ = mw.prepare_st001000(table, cohort="Validation")
    assert external.shape == (2, 2) and external_y.tolist() == [0, 1]
    assert set(groups).isdisjoint(external_groups)


@pytest.mark.parametrize("field", ["row_idx", "Age", "Diagnosis"])
def test_st001000_rejects_administrative_or_clinical_predictor_columns(table, field):
    table[field] = 1
    with pytest.raises(AssertionError, match="measurement columns"):
        mw.prepare_st001000(table)


def test_st001000_rejects_duplicate_or_unrecognized_participants(table):
    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        mw.prepare_st001000(duplicate)
    table.loc[0, "Samples"] = "unknown_patient"
    with pytest.raises(AssertionError, match="cohort membership"):
        mw.prepare_st001000(table)


@pytest.mark.parametrize(
    "bio_id", ["MW-ST001000-CD-UC", "MoleculeNet-Tox21-SR-MMP", "MoleculeNet-Tox21-NR-AhR"]
)
def test_registry_source_revision_changes_cache_identity(bio_id):
    spec = get_spec(bio_id)
    previous = spec_fingerprint(spec)
    assert spec_fingerprint(replace(spec, download_sha256="0" * 64)) != previous
    assert spec_fingerprint(replace(spec, download_url="https://example.org/revised")) != previous


def test_curated_loader_dispatch():
    spec = get_spec("MW-ST001000-CD-UC")
    assert isinstance(get_loader(spec), mw.MetabolomicsLoader)
