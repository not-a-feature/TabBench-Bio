"""Biological metadata must reach validation without becoming model input."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.bio import datasets as registry
from tabbench_bio.dataset import Dataset, DatasetInfo, TaskType


@pytest.fixture
def model_module(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.device_count.return_value = 0
    monkeypatch.setitem(sys.modules, "torch", torch)
    common = ModuleType("autogluon.common")
    common.TabularDataset = pd.DataFrame
    tabular = ModuleType("autogluon.tabular")
    tabular.TabularPredictor = MagicMock()
    monkeypatch.setitem(sys.modules, "autogluon.common", common)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    path = importlib.util.find_spec("tabbench_bio.model").origin
    spec = importlib.util.spec_from_file_location("group_validation_model_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_resolve_hyperparameters", lambda *args: {"RF": {}})
    return module


@pytest.mark.parametrize("native", [False, True])
def test_groups_reach_native_validation_but_are_ignored_features(model_module, tmp_path, native):
    frame = pd.DataFrame({"f": range(40), "target": [0, 1] * 20}, index=range(100, 140))
    groups = pd.Series(np.repeat(range(10), 4), index=frame.index)
    model = model_module.AutoGluonModel(
        ["AUTOGLUON" if native else "RF"],
        ensemble=native,
        optimize=False,
        task_type=TaskType.Classification,
        autogluon_path=str(tmp_path),
    )
    model.fit(frame, groups=groups)
    init = model_module.TabularPredictor.call_args.kwargs
    fit = model.predictor.fit.call_args
    column = fit.kwargs["validation_structure"]["group_on"]
    assert init["learner_kwargs"]["ignored_columns"] == [column]
    pd.testing.assert_series_equal(fit.args[0][column], groups, check_names=False)
    assert column not in frame
    if not native:
        assert fit.kwargs["num_bag_folds"] == 0
        assert fit.kwargs["refit_full"] is True


@pytest.mark.parametrize("scheduler_cpus,model_cpus,expected", [(18, "32", 32), (32, None, 32)])
def test_model_threads_can_differ_from_scheduler_cores(
    model_module, monkeypatch, tmp_path, scheduler_cpus, model_cpus, expected
):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(scheduler_cpus))
    if model_cpus is None:
        monkeypatch.delenv("TABBENCH_MODEL_CPUS", raising=False)
    else:
        monkeypatch.setenv("TABBENCH_MODEL_CPUS", model_cpus)
    model = model_module.AutoGluonModel(
        ["RF"],
        ensemble=False,
        optimize=False,
        task_type=TaskType.Classification,
        autogluon_path=str(tmp_path),
    )
    model.fit(pd.DataFrame({"f": range(40), "target": [0, 1] * 20}))
    assert model.predictor.fit.call_args.kwargs["num_cpus"] == expected


def test_unique_groups_leave_iid_protocol_unchanged(model_module, tmp_path):
    frame = pd.DataFrame({"f": range(40), "target": [0, 1] * 20})
    model = model_module.AutoGluonModel(
        ["RF"],
        ensemble=False,
        optimize=False,
        task_type=TaskType.Classification,
        autogluon_path=str(tmp_path),
    )
    model.fit(frame, groups=pd.Series(range(40)))
    assert "validation_structure" not in model.predictor.fit.call_args.kwargs
    assert "learner_kwargs" not in model_module.TabularPredictor.call_args.kwargs


def test_grouped_mitra_does_not_create_a_random_holdout_during_refit(
    model_module, monkeypatch, tmp_path
):
    limits = ModuleType("tabbench_bio.models.mitra_limits")
    limits.mitra_actual_sample_limits = nullcontext
    monkeypatch.setitem(sys.modules, "tabbench_bio.models.mitra_limits", limits)
    frame = pd.DataFrame({"f": range(40), "target": [0, 1] * 20})
    model = model_module.AutoGluonModel(
        ["MITRA"],
        ensemble=False,
        optimize=False,
        task_type=TaskType.Classification,
        autogluon_path=str(tmp_path),
    )
    model.fit(frame, groups=pd.Series(np.repeat(range(10), 4)))
    args = model.predictor.fit.call_args.kwargs
    assert args["refit_full"] is False
    assert args["set_best_to_refit_full"] is False
    assert "validation_structure" in args


def test_misaligned_or_single_class_group_is_rejected(model_module, tmp_path):
    frame = pd.DataFrame({"f": range(40), "target": [0] * 20 + [1] * 20})
    model = model_module.AutoGluonModel(
        ["RF"],
        ensemble=False,
        optimize=False,
        task_type=TaskType.Classification,
        autogluon_path=str(tmp_path),
    )
    with pytest.raises(AssertionError, match="misaligned"):
        model.fit(frame, groups=pd.Series(range(40), index=range(1, 41)))
    with pytest.raises(AssertionError, match="each class"):
        model.fit(frame, groups=pd.Series([0] * 20 + [1] * 20))


def test_groups_align_after_arbitrary_subsampling_and_load_once(monkeypatch, tmp_path):
    dataset = Dataset(
        features=np.ones((6, 2)),
        targets=np.arange(6),
        feature_names=["a", "b"],
        target_names=["y"],
        info=DatasetInfo("toy", "toy", TaskType.Regression),
        groups=np.array(["a", "a", "b", "b", "c", "c"]),
    )
    bench = TabBenchBio([], [], cache_dir=str(tmp_path))
    loader = MagicMock(return_value=dataset)
    monkeypatch.setattr(bench, "_load_dataset", loader)
    for rows in ([5, 0, 3], [2, 1]):
        train = dataset.to_dataframe().loc[rows]
        groups = bench.training_groups("toy_0", train)
        assert groups.index.tolist() == rows
        assert groups.tolist() == dataset.groups[rows].tolist()
    assert loader.call_count == 1


def test_new_benchmark_reloads_registry_and_excludes_cached_disabled_dataset(monkeypatch, tmp_path):
    path = tmp_path / "registry.json"
    entry = {
        "bio_id": "toy",
        "source": "local",
        "fetch_id": "unused.csv",
        "target": "y",
        "problem_type": "binary",
        "enabled": True,
    }
    path.write_text(json.dumps([entry]))
    monkeypatch.setenv("TABBENCH_BIO_DATASETS", str(path))
    monkeypatch.setattr(registry, "BIO_DATASETS", {})
    first = TabBenchBio(["toy"], [], cache_dir=str(tmp_path / "cache"))
    assert first.dataset_names_classification == ["toy"]
    frame = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
    first._save_dataset("toy_0", frame, frame)
    entry["enabled"] = False
    path.write_text(json.dumps([entry]))
    second = TabBenchBio(["toy"], [], cache_dir=str(tmp_path / "cache"))
    assert second._has_dataset_in_cache("toy_0")
    second.init_datasets()
    assert second.dataset_names_classification == []
    assert len(second) == 0
