from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from tabbench_bio.bio.adapter import load_bio_dataset
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.bio.loaders.mgnify import (
    GROUPING_VERSION,
    biological_groups,
    grouping_digest,
)


def test_replicates_and_duplicate_profiles_merge_entire_participants():
    X = pd.DataFrame({"x": [1, 2, 2, 3, 4, 5]})
    samples = ["s1", "s1", "s2", "s3", "s4", "s5"]
    mapping = {"s1": "p1", "s2": "p2", "s3": "p2", "s4": "p3", "s5": "p4"}
    groups = biological_groups(X, samples, mapping)
    assert groups.iloc[:4].nunique() == 1
    assert groups.nunique() == 3
    assert X.x.tolist() == [1, 2, 2, 3, 4, 5]
    y = [0, 1, 1, 0, 0, 1]
    for train, test in StratifiedGroupKFold(2).split(X, y, groups):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
        assert not set(X.x.iloc[train]) & set(X.x.iloc[test])


def test_unknown_samples_cannot_silently_become_independent():
    with pytest.raises(AssertionError, match="lack curated biological groups"):
        biological_groups(pd.DataFrame({"x": [1]}), ["unknown"], {})


@pytest.mark.parametrize(
    "metadata,groups",
    [
        ({}, None),
        ({}, pd.Series(["p1"])),
        ({"grouping_version": GROUPING_VERSION, "grouping_sha256": "old"}, pd.Series(["p1"])),
    ],
)
def test_obsolete_mgnify_cache_is_rebuilt(metadata, groups):
    old = BioRawDataset(
        "MGYS00005384",
        pd.DataFrame({"x": [1]}),
        pd.Series(["male"]),
        "binary",
        "public",
        "url",
        "citation",
        metadata,
        groups,
    )
    fresh = replace(
        old,
        metadata={"grouping_version": GROUPING_VERSION, "grouping_sha256": grouping_digest()},
        groups=pd.Series(["p1"]),
    )
    with (
        patch("tabbench_bio.bio.adapter.load_cached_raw", return_value=old),
        patch("tabbench_bio.bio.adapter.get_loader") as loader,
        patch("tabbench_bio.bio.adapter.save_cached_raw") as save,
    ):
        loader.return_value.fetch.return_value = fresh
        assert load_bio_dataset(old.bio_id) is fresh
        save.assert_called_once()
