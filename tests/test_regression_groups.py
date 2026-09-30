from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split

from tabbench_bio.benchmark import TabBenchBio


@pytest.mark.parametrize("folds", [None, 4])
def test_equal_targets_do_not_define_groups(tmp_path, folds):
    frame = pd.DataFrame({"x": range(80), "target": np.tile([40, 45, 50, 55], 20)})
    bench = TabBenchBio(
        [],
        ["toy"],
        cache_dir=str(tmp_path),
        cv_folds=folds,
        random_state=0,
    )
    train, test = bench._split(frame, "toy", SimpleNamespace(groups=None))
    if folds is None:
        expected_train, expected_test = train_test_split(frame, test_size=0.2, random_state=0)
    else:
        train_idx, test_idx = next(KFold(4, shuffle=True, random_state=0).split(frame))
        expected_train, expected_test = frame.iloc[train_idx], frame.iloc[test_idx]
    pd.testing.assert_frame_equal(train, expected_train)
    pd.testing.assert_frame_equal(test, expected_test)
    assert set(train.target) == set(test.target)


@pytest.mark.parametrize("folds", [None, 4])
def test_explicit_regression_groups_remain_disjoint(tmp_path, folds):
    frame = pd.DataFrame({"x": range(80), "target": [45] * 80})
    groups = np.repeat(range(20), 4)
    bench = TabBenchBio([], ["toy"], cache_dir=str(tmp_path), cv_folds=folds)
    train, test = bench._split(frame, "toy", SimpleNamespace(groups=groups))
    assert set(groups[train.index]).isdisjoint(groups[test.index])


def test_legacy_target_grouping_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match="group_regression_splits must be False"):
        TabBenchBio([], [], cache_dir=str(tmp_path), group_regression_splits=True)
