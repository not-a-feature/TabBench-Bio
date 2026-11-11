"""Exercise the pinned LightPFN weights through AutoGluon on CPU."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("lightpfn")

import torch

from tabbench_bio.dataset import TaskType
from tabbench_bio.model import AutoGluonModel, _resolve_hyperparameters
from tabbench_bio.models import lightpfn
from tabbench_bio.models.lightpfn import CHECKPOINT_REVISION, LightPFNModel


def _data(classes=0, features=6):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(size=(44, features)),
        columns=[f"x{i}" for i in range(features)],
    )
    X["kind"] = ["a", "b"] * 20 + ["unseen", "b", "a", "unseen"]
    X["integer_category"] = np.arange(44) % 3
    X.loc[2, "x0"] = np.nan
    y = pd.Series(np.arange(40) % classes if classes else 100 + X.x1.iloc[:40] * 20)
    return X, y


@pytest.mark.parametrize(
    "task,classes", [("binary", 2), ("multiclass", 3), ("multiclass", 10), ("regression", 0)]
)
def test_released_checkpoint_fit_predict_and_restore(tmp_path, task, classes):
    assert version("lightpfn") == "1.5.1"
    X, y = _data(classes)
    model = LightPFNModel(path=str(tmp_path), name="LightPFN", problem_type=task)
    model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)
    prediction = model.predict_proba(X.iloc[40:])
    assert prediction.shape == ((4, classes) if classes > 2 else (4,))
    assert np.isfinite(prediction).all()
    assert model.get_device() == "cpu"
    assert model.model.device == "cpu"
    assert model.model.n_threads == 1
    assert model.model.revision == CHECKPOINT_REVISION
    assert model.model.n_estimators == (4 if classes else 2)
    assert model.model.max_context == (20_000 if classes else 50_000)
    assert model.model.precision == "auto"
    assert model.model.cache_context is True
    if classes:
        np.testing.assert_array_equal(model.model.classes_, np.arange(classes))
        if classes > 2:
            np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-5)
        categories = model.model.categories_
        assert model.model.cat_[-1]  # Detect integer categories from training rows.
    else:
        categories = model.model.regressor_.categories_
        assert model.model.log_skew == 2.0
    assert "unseen" not in categories[6]
    restored = LightPFNModel.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[40:]), prediction, atol=1e-6)


@pytest.mark.parametrize("task,classes", [("multiclass", 3), ("regression", 0)])
def test_predictions_are_independent_of_other_query_rows(tmp_path, task, classes):
    X, y = _data(classes)
    model = LightPFNModel(path=str(tmp_path), name="LightPFN", problem_type=task)
    model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)
    first = model.predict_proba(X.iloc[40:41])
    X.loc[41:, "x0"] = 1e6
    np.testing.assert_allclose(model.predict_proba(X.iloc[40:])[:1], first, atol=1e-5)


def test_wide_inputs_reach_the_estimator_without_feature_selection(tmp_path):
    X, y = _data(2, features=501)
    model = LightPFNModel(path=str(tmp_path), name="LightPFN", problem_type="binary")
    model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)
    assert model.model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.predict_proba(X.iloc[40:])).all()


def test_class_limit_fails_before_loading_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(lightpfn, "LightPFNClassifier", lambda **kw: pytest.fail("Loaded weights"))
    X, y = _data(11)
    model = LightPFNModel(path=str(tmp_path), name="LightPFN", problem_type="multiclass")
    with pytest.raises(ValueError, match="2-10 classes"):
        model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)


def test_explicit_gpu_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        LightPFNModel(problem_type="binary")._fit(None, None, num_gpus=1)


def test_wrong_package_version_fails_before_loading_weights(monkeypatch):
    monkeypatch.setattr(lightpfn, "version", lambda package: "1.5.0")
    with pytest.raises(RuntimeError, match="lightpfn==1.5.1"):
        LightPFNModel()._fit(None, None)


@pytest.mark.parametrize("task,classes", [(TaskType.Classification, 3), (TaskType.Regression, 0)])
def test_benchmark_wrapper_refits_and_preserves_output_units(tmp_path, monkeypatch, task, classes):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    X, y = _data(classes)
    train = X.iloc[:40].copy()
    train["target"] = y.map({0: "z", 1: "a", 2: "m"}) if classes else y
    wrapper = AutoGluonModel(
        models=["LIGHTPFN"],
        task_type=task,
        ensemble=False,
        optimize=False,
        autogluon_presets="medium_quality",
        autogluon_time_limit=60,
        autogluon_path=str(tmp_path),
    )
    wrapper.fit(train)
    fitted = wrapper.predictor._trainer.load_model(wrapper.predictor.model_best)
    assert fitted.name.endswith("_FULL")
    prediction = wrapper.predict(X.iloc[40:])
    assert prediction.shape == (4,)
    if classes:
        probability = wrapper.predict_proba(X.iloc[40:])
        assert list(probability.columns) == ["a", "m", "z"]
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
        np.testing.assert_array_equal(prediction, probability.idxmax(axis=1))
    else:
        processed = wrapper.predictor.transform_features(X.iloc[40:])
        expected = fitted.predict(processed) * wrapper._target_std + wrapper._target_mean
        np.testing.assert_allclose(prediction, expected, rtol=1e-5)


def test_registry_resolves_adapter():
    assert _resolve_hyperparameters(["LIGHTPFN"], 1) == {LightPFNModel: [{"ag.num_gpus": 1}]}
