"""Fresh source ordering must not change published sample identities or patient splits."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.bio import sample_order
from tabbench_bio.bio.adapter import bio_raw_to_dataset
from tabbench_bio.bio.loaders.base import BioRawDataset


def test_adapter_aligns_fresh_features_labels_and_groups_without_changing_raw(monkeypatch):
    monkeypatch.setattr(
        sample_order, "sample_orders", lambda: {"toy": {"row_ids": ["a", "b", "c"]}}
    )
    index = ["c", "a", "b"]
    raw = BioRawDataset(
        "toy",
        pd.DataFrame({"x": [30, 10, 20]}, index=index),
        pd.Series([1, 0, 1], index=index, name="y"),
        "binary",
        "",
        "",
        "",
        {},
        groups=pd.Series(["p2", "p1", "p2"], index=index),
    )
    adapted = bio_raw_to_dataset(raw)
    assert adapted.features[:, 0].tolist() == [10, 20, 30]
    assert adapted.targets.tolist() == [0, 1, 1]
    assert adapted.groups.tolist() == ["p1", "p2", "p2"]
    assert raw.X.index.tolist() == index
    assert "sample_order_sha256" not in raw.metadata
    with pytest.raises(AssertionError, match="source cohort differs"):
        sample_order.align_reference_rows(replace(raw, X=raw.X.drop(index="b")))


def test_row_order_changes_invalidate_only_affected_prepared_cache(tmp_path, monkeypatch):
    manifest = {"OpenML-1138": {"row_ids": ["a", "b"]}}
    monkeypatch.setattr(sample_order, "sample_orders", lambda: manifest)
    bench = TabBenchBio([], [], cache_dir=str(tmp_path))
    before = bench._get_cache_paths("OpenML-1138_0")
    other = bench._get_cache_paths("OpenML-1083_0")
    manifest["OpenML-1138"]["row_ids"].reverse()
    assert bench._get_cache_paths("OpenML-1138_0") != before
    assert bench._get_cache_paths("OpenML-1083_0") == other


def test_tied_class_counts_preserve_first_occurrence_without_platform_sort(tmp_path, monkeypatch):
    bench = TabBenchBio(
        dataset_names_classification=["OpenML-1106"],
        dataset_names_regression=[],
        train_subsample=100,
        max_classes=4,
        cache_dir=str(tmp_path),
        random_state=0,
    )
    names = [f"class_{i}" for i in range(14)]
    counts = [24, 18, 16, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8]
    frame = pd.DataFrame({"target": np.repeat(names, counts)})
    original = pd.Series.value_counts

    def unsorted_counts(series, *args, **kwargs):
        assert kwargs["sort"] is False, (
            "Default frequency sort can reorder tied classes across CPUs"
        )
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "value_counts", unsorted_counts)
    sampled = bench._subsample_train(frame, "OpenML-1106_0")
    assert len(sampled) == 100 and sampled.index.is_unique
    assert (sampled.target == "class_3").sum() == 7
    assert (sampled.target == "class_6").sum() == 6
    pd.testing.assert_frame_equal(bench._subsample_train(frame, "OpenML-1106_0"), sampled)
    assert set(bench._limit_classes(frame, "OpenML-1106_0").target) == set(names[:4])
