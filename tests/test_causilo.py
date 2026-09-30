"""Exercise the real Causilo backend with tiny, local test weights on CPU."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("causilo")

import torch
from causilo import checkpoints
from causilo.model import Model, ModelConfig

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models import causilo
from tabbench_bio.models.causilo import CausiloModel


def _test_weights(task):
    with torch.random.fork_rng():
        torch.manual_seed(12)
        model = Model(
            ModelConfig(
                task=task,
                width=16,
                expansion=2,
                group_size=3,
                frequencies=4,
                column_latents=2,
                column_heads=2,
                column_depths=(1, 1),
                row_heads=2,
                row_latents=2,
                row_depths=(1, 1),
                prediction_heads=2,
                prediction_depth=1,
                outputs=10 if task == "classification" else 9,
            )
        )
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
    return model.eval()


@pytest.fixture
def checkpoint_tasks(monkeypatch):
    tasks = []

    def load(task):
        tasks.append(task)
        return _test_weights(task)

    monkeypatch.setattr(checkpoints, "load_pretrained_model", load)
    return tasks


@pytest.mark.parametrize(
    "task,classes", [("binary", 2), ("multiclass", 3), ("multiclass", 12), ("regression", 0)]
)
def test_fit_predict_and_restore(tmp_path, checkpoint_tasks, task, classes):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(52, 4)), columns=list("abcd"))
    X["category"] = pd.Series(["a", "b"] * 25 + ["unseen", "b"], dtype="category")
    X.loc[2, "a"] = np.nan
    y = pd.Series(np.arange(48) % classes if classes else 100 + X.b.iloc[:48] * 20)
    model = CausiloModel(path=str(tmp_path), name="Causilo", problem_type=task)
    model.fit(X=X.iloc[:48], y=y, num_cpus=1, num_gpus=0)
    state = model.model._engine.state.dataset
    categories = state.encoder.encoder.categories_[0].copy()
    prediction = model.predict_proba(X.iloc[48:])
    assert prediction.shape == ((4, classes) if classes > 2 else (4,))
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(state.encoder.encoder.categories_[0], categories)
    assert "unseen" not in categories
    if classes:
        np.testing.assert_array_equal(model.model.classes_, np.arange(classes))
        if classes > 2:
            np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-5)
    restored = CausiloModel.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[48:]), prediction, atol=1e-6)
    expected_task = "classification" if classes else "regression"
    assert checkpoint_tasks == [expected_task, expected_task]
    assert model.model.n_estimators == 8
    assert model.model.random_state == 42
    assert model.model.device == "cpu"
    assert not model.model.use_kv_cache
    assert model.model.retain_preprocessing


def test_regression_preserves_target_units(tmp_path, checkpoint_tasks):
    X = pd.DataFrame(np.random.default_rng(8).normal(size=(24, 3)), columns=list("abc"))
    y = pd.Series(np.arange(20, dtype=float))
    predictions = []
    for offset in [0, 500]:
        model = CausiloModel(path=str(tmp_path), name=f"Causilo{offset}", problem_type="regression")
        model.fit(X=X.iloc[:20], y=y + offset, num_cpus=1, num_gpus=0)
        predictions.append(model.predict(X.iloc[20:]))
    np.testing.assert_allclose(predictions[1] - predictions[0], 500, atol=1e-4)


def test_explicit_gpu_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        CausiloModel()._fit(None, None, num_gpus=1)


def test_wrong_checkpoint_revision_fails_before_fitting(monkeypatch):
    monkeypatch.setattr(causilo, "RELEASE_COMMIT", "wrong-revision")
    with pytest.raises(AssertionError, match="checkpoint revision"):
        CausiloModel()._fit(None, None)


def test_registry_resolves_adapter():
    assert _resolve_hyperparameters(["CAUSILO"], 1) == {CausiloModel: [{"ag.num_gpus": 1}]}
