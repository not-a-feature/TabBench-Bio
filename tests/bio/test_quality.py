"""Unit tests for explicit dataset quality gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.bio.quality import QualityThresholds, assess_raw_dataset


def _raw() -> BioRawDataset:
    rng = np.random.default_rng(7)
    return BioRawDataset(
        bio_id="candidate",
        X=pd.DataFrame(rng.normal(size=(40, 20))),
        y=pd.Series([0, 1] * 20),
        problem_type="binary",
        license="test",
        source_url="test",
        citation="test",
        metadata={},
        groups=pd.Series([f"g{i}" for i in range(40)]),
    )


def test_quality_report_passes_good_small_fixture_with_explicit_thresholds():
    thresholds = QualityThresholds(
        min_samples=20,
        min_features=10,
        min_nonconstant_features=10,
        min_class_samples=10,
        min_groups=10,
        min_groups_per_class=5,
    )
    report = assess_raw_dataset(_raw(), thresholds)
    assert report["status"] == "pass"
    assert report["n_groups"] == 40


def test_quality_report_explains_failure():
    raw = _raw()
    raw.X.iloc[0, 0] = np.nan
    report = assess_raw_dataset(raw)
    assert report["status"] == "fail"
    assert any("non-finite feature" in issue for issue in report["issues"])
