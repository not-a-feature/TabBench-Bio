"""Exercise both adapters with the real backend and small local checkpoints."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("neuralk_oss")

import torch
from autogluon.tabular import TabularPredictor
from neuralk_oss import Seldon
from neuralk_oss.model import seldon as backend
from neuralk_oss.model.config import SeldonConfig

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models.seldon import CHECKPOINT_REVISIONS, SeldonMiniModel, SeldonModel


@pytest.fixture
def local_weights(tmp_path, monkeypatch):
    config = SeldonConfig(
        embed_dim=8,
        use_rmsnorm=True,
        head={"task": "classification", "max_classes": 10},
        col={"num_blocks": 1, "nhead": 2, "bias_free_ln": True},
        row={"num_blocks": 1, "nhead": 2, "num_cls": 2, "bias_free_ln": True},
        icl={"num_blocks": 1, "nhead": 2, "bias_free_ln": True},
        num_inducing_points=4,
        num_frequencies=4,
        icl_channels=16,
        num_icl_key_value_heads_for_query=None,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = Seldon(config)
        model.save_pretrained(tmp_path / "checkpoint")

    def download(repo_id, revision, **kwargs):
        name = repo_id.split("/")[-1]
        assert revision == CHECKPOINT_REVISIONS[name]
        return str(tmp_path / "checkpoint")

    monkeypatch.setattr(backend, "snapshot_download", download)


@pytest.mark.parametrize("adapter,members", [(SeldonModel, 8), (SeldonMiniModel, 4)])
@pytest.mark.parametrize("classes", [2, 3, 12])
def test_predict_restore_and_query_independence(tmp_path, local_weights, adapter, members, classes):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(28, 4)), columns=list("abcd"))
    X["category"] = pd.Series(["a", "b"] * 12 + ["unseen", "b", "a", "b"], dtype="category")
    X.loc[2, "a"] = np.nan
    task = "binary" if classes == 2 else "multiclass"
    model = adapter(path=str(tmp_path), name="Seldon", problem_type=task)
    model.fit(X=X.iloc[:24], y=pd.Series(np.arange(24) % classes), num_cpus=1, num_gpus=0)
    assert model.model.n_estimators == members
    assert model.model.estimator.recipe.half_shuffle is True
    assert not model.model._subsampled
    prediction = model.predict_proba(X.iloc[24:])
    assert prediction.shape == ((4,) if classes == 2 else (4, classes))
    assert np.isfinite(prediction).all()
    if classes > 2:
        np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(model.predict_proba(X.iloc[24:25]), prediction[:1], atol=1e-5)
    restored = adapter.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[24:]), prediction, atol=1e-5)


@pytest.mark.parametrize("adapter", [SeldonModel, SeldonMiniModel])
def test_all_features_reach_the_network(tmp_path, local_weights, monkeypatch, adapter):
    widths = []
    original = Seldon.encode_context

    def record(self, X, *args, **kwargs):
        widths.append(X.shape[-1])
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(Seldon, "encode_context", record)
    X = pd.DataFrame(np.random.default_rng(5).normal(size=(14, 513))).add_prefix("f")
    model = adapter(path=str(tmp_path), name="WideSeldon", problem_type="binary")
    model.fit(X=X.iloc[:12], y=pd.Series([0, 1] * 6), num_cpus=1, num_gpus=0)
    assert np.isfinite(model.predict_proba(X.iloc[12:])).all()
    assert widths and set(widths) == {513}


@pytest.mark.parametrize("adapter", [SeldonModel, SeldonMiniModel])
def test_gpu_request_requires_cuda(monkeypatch, adapter):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        adapter(problem_type="binary")._fit(None, None, num_gpus=1)


def test_both_models_resolve_together():
    assert _resolve_hyperparameters(["SELDON", "SELDON-MINI"], 1) == {
        SeldonModel: [{"ag.num_gpus": 1}],
        SeldonMiniModel: [{"ag.num_gpus": 1}],
    }


@pytest.mark.parametrize("adapter", [SeldonModel, SeldonMiniModel])
def test_autogluon_full_refit(tmp_path, local_weights, adapter):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(30, 4))).add_prefix("f")
    train = X.iloc[:24].copy()
    train["target"] = np.arange(24) % 2
    predictor = TabularPredictor(label="target", path=str(tmp_path / "predictor"))
    predictor.fit(
        train_data=train,
        hyperparameters={adapter: {}},
        num_cpus=1,
        num_gpus=0,
        num_bag_folds=0,
        fit_weighted_ensemble=False,
        refit_full=True,
        set_best_to_refit_full=True,
    )
    assert predictor.model_best.endswith("_FULL")
    model = predictor._trainer.load_model(predictor.model_best)
    assert not model.model._subsampled
    prediction = predictor.predict_proba(X.iloc[24:]).to_numpy()
    restored = TabularPredictor.load(predictor.path)
    np.testing.assert_allclose(restored.predict_proba(X.iloc[24:]), prediction, atol=1e-5)
