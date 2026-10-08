"""Exercise GeneICL's bundled seed-0 weights and AutoGluon adapter on CPU."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("geneicl")

import torch
from geneicl import CKPT

from tabbench_bio.dataset import TaskType
from tabbench_bio.model import AutoGluonModel, _resolve_hyperparameters
from tabbench_bio.models import geneicl
from tabbench_bio.models.geneicl import GeneICLModel


def _data(classes=0):
    X = pd.DataFrame(np.random.default_rng(42).normal(5, 1, (44, 6)), columns=list("abcdef"))
    X["category"] = pd.Series(["a", "b"] * 20 + ["unseen", "b", "a", "unseen"], dtype="category")
    X.loc[2, "a"] = np.nan
    X.loc[41, "b"] = np.inf
    y = pd.Series(np.arange(40) % classes if classes else 100 + X.c.iloc[:40] * 20)
    return X, y


@pytest.mark.parametrize(
    "task,classes", [("binary", 2), ("multiclass", 3), ("multiclass", 10), ("regression", 0)]
)
def test_released_checkpoint_fit_predict_and_restore(tmp_path, task, classes):
    X, y = _data(classes)
    model = GeneICLModel(path=str(tmp_path), name="GeneICL", problem_type=task)
    model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)
    categories = model._encoder.named_transformers_["category"].categories_[0].copy()
    support = model.model.X_.copy()
    prediction = model.predict_proba(X.iloc[40:])
    assert prediction.shape == ((4, classes) if classes > 2 else (4,))
    assert np.isfinite(prediction).all()
    if classes:
        np.testing.assert_array_equal(model.model.classes_, np.arange(classes))
        if classes > 2:
            np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(categories, ["a", "b"])
    np.testing.assert_array_equal(
        model._encoder.named_transformers_["category"].categories_[0], categories
    )
    np.testing.assert_array_equal(model.model.X_, support)
    assert model.model._runner.device == "cpu"
    assert model.model.ensemble_seed is None
    assert CKPT.name == "trm_segmented8_s0.pt"
    restored = GeneICLModel.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[40:]), prediction, atol=1e-6)


@pytest.mark.parametrize("task,classes", [("multiclass", 3), ("regression", 0)])
def test_query_rows_do_not_change_other_predictions(tmp_path, task, classes):
    X, y = _data(classes)
    model = GeneICLModel(path=str(tmp_path), name="GeneICL", problem_type=task)
    model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)
    first = model.predict_proba(X.iloc[40:41])
    query = X.iloc[40:].copy()
    query.loc[41:, list("abcdef")] = 1e6
    np.testing.assert_allclose(model.predict_proba(query)[:1], first, rtol=1e-5, atol=1e-5)
    model.model._runner.query_chunk = 1
    np.testing.assert_allclose(model.predict_proba(query)[:1], first, rtol=1e-5, atol=1e-5)


def test_regression_preserves_target_units(tmp_path):
    X, y = _data()
    predictions = []
    for offset in [0, 500]:
        model = GeneICLModel(path=str(tmp_path), name=f"GeneICL{offset}", problem_type="regression")
        model.fit(X=X.iloc[:40], y=y + offset, num_cpus=1, num_gpus=0)
        predictions.append(model.predict(X.iloc[40:]))
    np.testing.assert_allclose(predictions[1] - predictions[0], 500, atol=1e-4)


def test_more_than_ten_classes_fails_before_loading_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(geneicl, "GeneICL", lambda **kw: pytest.fail("Loaded weights"))
    X, y = _data(11)
    model = GeneICLModel(path=str(tmp_path), name="GeneICL", problem_type="multiclass")
    with pytest.raises(ValueError, match="2–10 classes"):
        model.fit(X=X.iloc[:40], y=y, num_cpus=1, num_gpus=0)


def test_explicit_gpu_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        GeneICLModel()._fit(None, None, num_gpus=1)


def test_wrong_checkpoint_fails_before_loading_weights(monkeypatch):
    monkeypatch.setattr(geneicl, "CHECKPOINT_SHA256", "wrong-checkpoint")
    with pytest.raises(AssertionError, match="seed-0 checkpoint"):
        GeneICLModel(problem_type="regression")._fit(None, None)


@pytest.mark.parametrize("task,classes", [(TaskType.Classification, 3), (TaskType.Regression, 0)])
def test_benchmark_wrapper_refits_all_training_rows(tmp_path, monkeypatch, task, classes):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    X, y = _data(classes)
    train = X.iloc[:40].copy()
    train["target"] = y.map({0: "z", 1: "a", 2: "m"}) if classes else y
    wrapper = AutoGluonModel(
        models=["GENEICL"],
        task_type=task,
        ensemble=False,
        optimize=False,
        autogluon_presets="medium_quality",
        autogluon_time_limit=60,
        autogluon_path=str(tmp_path),
    )
    wrapper.fit(train)
    fitted = wrapper.predictor._trainer.load_model(wrapper.predictor.model_best)
    assert len(fitted.model.X_) == len(train)
    prediction = wrapper.predict(X.iloc[40:])
    assert prediction.shape == (4,)
    if classes:
        probabilities = wrapper.predict_proba(X.iloc[40:])
        assert list(probabilities.columns) == ["a", "m", "z"]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        np.testing.assert_array_equal(prediction, probabilities.idxmax(axis=1))
    else:
        processed = wrapper.predictor.transform_features(X.iloc[40:])
        expected = fitted.predict(processed) * wrapper._target_std + wrapper._target_mean
        np.testing.assert_allclose(prediction, expected, rtol=1e-5)


def test_registry_resolves_adapter():
    assert _resolve_hyperparameters(["GENEICL"], 1) == {GeneICLModel: [{"ag.num_gpus": 1}]}
