"""Adapter contracts and an opt-in smoke test using the released LimiX-2 weights."""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("limix")

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models import limix_2
from tabbench_bio.models.limix_2 import (
    LimiX2Model,
    _Predictor,
    _TrainOnlyEncoder,
    _TrainOnlyFilter,
    _TrainOnlyInteractions,
)


class _Backend:
    def __init__(self, **kwargs):
        self.settings = kwargs

    def predict(self, X, y, query, task_type):
        assert list(query.columns) == list(X.columns)
        if task_type == "Regression":
            return np.full(len(query), np.mean(y))
        self.classes, counts = np.unique(y, return_counts=True)
        return np.tile(counts / counts.sum(), (len(query), 1))


def _data(task):
    X = pd.DataFrame(np.random.default_rng(8).normal(size=(40, 6)), columns=list("abcdef"))
    classes = 2 if task == "binary" else 3
    y = pd.Series(np.arange(len(X)) % classes if task != "regression" else X.a * 20 + 100)
    return X, y


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_adapter_contract_and_reload(tmp_path, monkeypatch, task):
    downloads = []
    monkeypatch.setattr(limix_2, "_Predictor", _Backend)

    def download(**kwargs):
        downloads.append(kwargs)
        return "LimiX-2.ckpt"

    monkeypatch.setattr(limix_2, "hf_hub_download", download)
    X, y = _data(task)
    model = LimiX2Model(path=str(tmp_path), name="LimiX2", problem_type=task)
    model.fit(X=X.iloc[:32], y=y.iloc[:32], num_cpus=1, num_gpus=0)
    query = X.iloc[32:]
    predictions = model.predict_proba(query)
    assert predictions.shape == ((8, 3) if task == "multiclass" else (8,))
    assert np.isfinite(predictions).all()
    if task == "regression":
        np.testing.assert_allclose(predictions, y.iloc[:32].mean())
    assert downloads == [
        {
            "repo_id": "stable-ai/LimiX-2",
            "filename": "LimiX-2.ckpt",
            "revision": limix_2.CHECKPOINT_REVISION,
        }
    ]
    assert model.model.settings["seed"] == 0
    assert model.model.settings["preprocess_num_jobs"] == 1
    assert model.model.settings["use_data_cache"] is False
    assert model.model.settings["inference_with_DDP"] is False
    assert (
        model.model.settings["inference_config"]["adaptive_svd"]["retry_on_cuda_resource_error"]
        is False
    )
    restored = LimiX2Model.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(query), predictions)


def test_filter_ignores_query_distribution():
    train = np.array([[1.0, 2.0, np.nan], [1.0, 3.0, np.nan]])
    query = np.array([[5.0, np.nan, 6.0]])
    filtered, _ = _TrainOnlyFilter().fit_transform(
        np.concatenate([train, query]), [], 0, y=np.array([0, 1])
    )
    np.testing.assert_allclose(filtered, [[2.0], [3.0], [np.nan]])


@pytest.mark.parametrize("strategy", ["ordinal_shuffled", "onehot"])
def test_encoder_fits_training_categories_only(strategy):
    train = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0], [1.0, 4.0]])
    query = np.array([[7.0, 5.0], [0.0, 6.0]])
    encoder = _TrainOnlyEncoder(encoding_strategy=strategy)
    values, _ = encoder.fit_transform(np.concatenate([train, query]), [0], 0, y=np.arange(4))
    transformer = encoder.transformer.transformers_[0][1]
    np.testing.assert_array_equal(transformer.categories_[0], [0.0, 1.0])
    if strategy == "ordinal_shuffled":
        assert np.isnan(values[4, 0])
        assert values[5, 0] == values[0, 0]
    assert np.isfinite(values[:, -1]).all()


def test_rejects_more_than_ten_classes_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(limix_2, "hf_hub_download", lambda **kw: pytest.fail("Downloaded weights"))
    X = pd.DataFrame({"x": np.arange(44)})
    model = LimiX2Model(path=str(tmp_path), name="LimiX2", problem_type="multiclass")
    with pytest.raises(ValueError, match="2–10 classes"):
        model.fit(X=X, y=pd.Series(np.arange(44) % 11), num_cpus=1, num_gpus=0)


def test_interaction_scaler_ignores_query_values():
    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    step = _TrainOnlyInteractions(max_interaction_features=2)
    step.fit_transform(np.vstack([train, [[1e6, 1e6]]]), [], 0, y=np.arange(3))
    np.testing.assert_allclose(step.feature_normalizer.mean_, train.mean(axis=0))


def test_regression_categories_are_fitted_on_training_only():
    predictor = object.__new__(_Predictor)
    predictor._train_rows = 3
    frame = pd.DataFrame({"cat": ["a", "b", "a", "new"], "numeric": [1.0, 2.0, 3.0, 4.0]})
    values = predictor.convert_category2num(frame)
    assert np.isnan(values[-1, 0])
    np.testing.assert_allclose(values[:3, 0], [0.0, 1.0, 0.0])


def test_gpu_request_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(limix_2.torch.cuda, "is_available", lambda: False)
    X, y = _data("binary")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        LimiX2Model()._fit(X, y, num_gpus=1)


def test_registry_resolves_gpu_adapter():
    assert _resolve_hyperparameters(["LIMIX-2"], 1) == {LimiX2Model: [{"ag.num_gpus": 1}]}


@pytest.mark.skipif(
    os.environ.get("TABBENCH_LIMIX_SMOKE") != "1", reason="Downloads LimiX-2 weights"
)
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_released_checkpoint(tmp_path, task):
    X, y = _data(task)
    X["category"] = pd.Series(["a", "b"] * 19 + ["a", "unseen"], dtype="category")
    model = LimiX2Model(path=str(tmp_path), name="LimiX2", problem_type=task)
    model.fit(X=X.iloc[:32], y=y.iloc[:32], num_cpus=4, num_gpus=1)
    prediction = model.predict_proba(X.iloc[32:])
    assert prediction.shape == ((8, 3) if task == "multiclass" else (8,))
    assert np.isfinite(prediction).all()
    restored = LimiX2Model.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[32:]), prediction, atol=1e-5)
