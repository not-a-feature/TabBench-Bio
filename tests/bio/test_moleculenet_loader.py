"""Tox21 label handling and transitive scaffold/fingerprint independence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.bio.loaders.moleculenet import (
    MoleculeNetLoader,
    prepare_tox21_table,
    scaffold_input_groups,
)


def test_fingerprint_collision_joins_entire_scaffold_families_transitively():
    X = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    scaffold = pd.Series(["A", "A", "B", "B", "C"])
    groups, metadata = scaffold_input_groups(X, scaffold)
    assert groups.iloc[:4].nunique() == 1
    assert groups.iloc[4] != groups.iloc[0]
    assert metadata["independence_groups"] == 2
    assert metadata["duplicate_fingerprint_rows"] == 1


def test_tox21_missing_labels_are_excluded_not_encoded_inactive():
    pytest.importorskip("rdkit")
    table = pd.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN", "CCC"],
            "NR-AhR": [1, 1, 0, np.nan],
            "SR-MMP": [0, 0, 1, 1],
        }
    )
    X, y, groups, metadata = prepare_tox21_table(table, endpoint="NR-AhR", bio_id="toy")
    assert X.shape == (2, 2048) and set(y) == {0, 1}
    assert len(groups) == 2
    assert metadata["unlabelled_rows_dropped"] == 1
    assert metadata["duplicate_rows"] == 1
    assert y.name == "NR-AhR"
    assert all(c.startswith("ecfp4_") for c in X)


def test_endpoint_mismatch_fails_before_download():
    spec = get_spec("MoleculeNet-Tox21-NR-AhR")
    assert isinstance(get_loader(spec), MoleculeNetLoader)
    with pytest.raises(AssertionError, match="same curated Tox21 assay"):
        MoleculeNetLoader().fetch(replace(spec, target="SR-MMP"))
