from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.seeds import get_seeds


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_folds_match_the_published_seed_zero_partition(tmp_path, task):
    frame = pd.DataFrame({"x": range(60), "target": np.tile([0, 1, 2], 20)})
    names = (["toy"], []) if task == "classification" else ([], ["toy"])
    splitter = (
        StratifiedKFold(5, shuffle=True, random_state=0)
        if task == "classification"
        else KFold(5, shuffle=True, random_state=0)
    )
    expected = list(splitter.split(frame, frame["target"]))
    tested = []
    for fold in range(5):
        bench = TabBenchBio(*names, cache_dir=str(tmp_path), cv_folds=5, random_state=fold)
        train, test = bench._split(frame, "toy", SimpleNamespace(groups=None))
        pd.testing.assert_frame_equal(test, frame.iloc[expected[fold][1]])
        pd.testing.assert_frame_equal(train, frame.iloc[expected[fold][0]])
        tested.extend(test.index)
    assert sorted(tested) == list(range(60))


def test_fold_index_outside_the_partition_is_rejected(tmp_path):
    frame = pd.DataFrame({"x": range(20), "target": np.tile([0, 1], 10)})
    bench = TabBenchBio(["toy"], [], cache_dir=str(tmp_path), cv_folds=5, random_state=5)
    with pytest.raises(AssertionError, match="CV fold index 5"):
        bench._split(frame, "toy", SimpleNamespace(groups=None))


@pytest.mark.parametrize("repetitions", [None, 1])
def test_cv_seeds_are_fold_indices(repetitions):
    config = {"cv_folds": 5, "n_repetitions": repetitions, "random_state": 42}
    assert get_seeds(config) == [0, 1, 2, 3, 4]


def test_repeated_cv_is_rejected():
    with pytest.raises(AssertionError, match="Repeated k-fold CV"):
        get_seeds({"cv_folds": 5, "n_repetitions": 2, "random_state": 42})
