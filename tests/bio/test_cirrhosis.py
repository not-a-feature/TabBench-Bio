import bz2
import sys
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch

import pandas as pd
import pytest

from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.bio.adapter import load_bio_dataset
from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.bio.loaders.metagenomics import (
    COHORT,
    COHORT_VERSION,
    MARKER_URL,
    MetagenomicsLoader,
)


def source(tmp_path, *, bad_site=False):
    rows = {
        "dataset_name": [COHORT] * 232 + ["other"] * 2,
        "sampleID": [f"sample{i}" for i in range(234)],
        "subjectID": [f"person{i}" for i in range(234)],
        "bodysite": ["stool"] * 232 + ["skin", "oral"],
        "disease": ["cirrhosis"] * 118 + ["n"] * 116,
        "numeric_metadata": ["1"] * 234,
        "gi|common": ["1"] * 234,
        "gi|rare": ["1"] + ["0"] * 233,
        "gi|test_only": ["0"] * 232 + ["1", "1"],
    }
    if bad_site:
        rows["bodysite"][0] = "oral"
    with bz2.open(tmp_path / "marker_presence.txt.bz2", "wt") as handle:
        for name, values in rows.items():
            handle.write(name + "\t" + "\t".join(values) + "\n")
    return MetagenomicsLoader(cache_dir=tmp_path)


def test_matched_cohort_retains_unfiltered_markers_and_subjects(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    raw = source(tmp_path).fetch(get_spec("gut-cirrhosis"))
    assert raw.X.shape == (232, 3)
    assert raw.y.value_counts().to_dict() == {"disease": 118, "healthy": 114}
    assert raw.groups.nunique() == 232
    assert raw.metadata["cohort_version"] == COHORT_VERSION
    assert "gi|rare" in raw.X and raw.X["gi|test_only"].sum() == 0


def test_download_works_without_requests(tmp_path, monkeypatch):
    loader = source(tmp_path)
    path = tmp_path / "marker_presence.txt.bz2"
    payload = path.read_bytes()
    path.unlink()
    monkeypatch.setitem(sys.modules, "requests", None)
    with patch(
        "tabbench_bio.bio.loaders.metagenomics.urlopen", return_value=BytesIO(payload)
    ) as download:
        raw = loader.fetch(get_spec("gut-cirrhosis"))
    download.assert_called_once_with(MARKER_URL, timeout=180)
    assert path.read_bytes() == payload
    assert raw.X.shape == (232, 3)


def test_non_stool_sample_in_curated_cohort_is_rejected(tmp_path):
    with pytest.raises(AssertionError):
        source(tmp_path, bad_site=True).fetch(get_spec("gut-cirrhosis"))


def test_prevalence_uses_training_only_and_precedes_feature_cap(tmp_path):
    bench = TabBenchBio(
        dataset_names_classification=["gut-cirrhosis"],
        dataset_names_regression=[],
        cache_dir=str(tmp_path),
        bio_max_features=1,
    )
    train = pd.DataFrame({"train_marker": [1] * 10, "test_marker": [0] * 10, "target": [0, 1] * 5})
    test = pd.DataFrame({"train_marker": [0, 0], "test_marker": [1, 1], "target": [0, 1]})
    actual_train, actual_test = bench._fit_apply_features(train, test, "gut-cirrhosis_0")
    assert actual_train.columns.tolist() == ["train_marker", "target"]
    assert actual_test.columns.tolist() == actual_train.columns.tolist()


def test_obsolete_cohort_cache_is_not_reused(tmp_path):
    fresh = source(tmp_path).fetch(get_spec("gut-cirrhosis"))
    stale = replace(fresh, metadata={"disease": "cirrhosis"})
    with (
        patch("tabbench_bio.bio.adapter.load_cached_raw", return_value=stale),
        patch("tabbench_bio.bio.adapter.get_loader") as loader,
        patch("tabbench_bio.bio.adapter.save_cached_raw"),
    ):
        loader.return_value.fetch.return_value = fresh
        assert load_bio_dataset("gut-cirrhosis") is fresh
