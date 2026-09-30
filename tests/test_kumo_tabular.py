"""Exercise the real Kumo backend using small, local test checkpoints."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("sdm")

import sdm
import torch
from autogluon.tabular import TabularPredictor
from sdm.models.kumo.tabular.model import MODEL_KWARGS

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models import kumo_tabular
from tabbench_bio.models.kumo_tabular import CHECKPOINT_REVISION, KumoTabularMediumModel


@pytest.fixture
def local_weights(tmp_path, monkeypatch):
    config = {
        **MODEL_KWARGS["medium"],
        "cell_channels": 8,
        "num_embedding_layers": 1,
        "num_embedding_heads": 2,
        "num_inducing_points": 4,
        "num_frequencies": 4,
        "icl_channels": 32,
        "num_icl_layers": 1,
        "num_icl_heads": 2,
    }
    monkeypatch.setitem(MODEL_KWARGS, "medium", config)
    for task, filename in [("classification", "classifier"), ("regression", "regressor")]:
        with torch.random.fork_rng():
            torch.manual_seed(12)
            model = sdm.models.KumoTabular(task=task, size="medium", pretrained=False)
            for parameter in model.parameters():
                if not parameter.any():
                    torch.nn.init.normal_(parameter, std=0.02)
        torch.save(model.models[task].state_dict(), tmp_path / f"{filename}.pt")

    def download(repo_id, filename, revision):
        assert repo_id == "nvidia/Kumo-Tabular"
        assert revision == CHECKPOINT_REVISION
        assert filename.startswith("medium/")
        return str(tmp_path / filename.split("/")[1])

    monkeypatch.setattr(kumo_tabular, "hf_hub_download", download)


@pytest.mark.parametrize("task,classes", [("binary", 2), ("multiclass", 12), ("regression", 0)])
def test_fit_predict_restore_and_unseen_categories(tmp_path, local_weights, task, classes):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(28, 4)), columns=list("abcd"))
    X["category"] = pd.Series(["a", "b"] * 12 + ["unseen", "b", "a", "b"], dtype="category")
    X.loc[2, "a"] = np.nan
    y = pd.Series(np.arange(24) % classes if classes else 100 + X.b.iloc[:24] * 20)
    model = KumoTabularMediumModel(path=str(tmp_path), name="Kumo", problem_type=task)
    model.fit(X=X.iloc[:24], y=y, num_cpus=1, num_gpus=0)
    assert model.model._cache is None
    assert len(model._contexts) == 8
    assert all(context.x.device.type == "cpu" for context in model._contexts)
    prediction = model.predict_proba(X.iloc[24:])
    assert prediction.shape == ((4, classes) if classes > 2 else (4,))
    assert np.isfinite(prediction).all()
    if classes > 2:
        np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-5)
    # A query's prediction must not depend on the other query rows.
    np.testing.assert_allclose(model.predict_proba(X.iloc[24:25]), prediction[:1], atol=1e-5)
    restored = KumoTabularMediumModel.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[24:]), prediction, atol=1e-5)


def test_more_than_500_features_reach_the_network(tmp_path, local_weights, monkeypatch):
    from sdm.models.kumo.tabular.row_embedding import RowEmbedding

    widths = []
    original = RowEmbedding.forward

    def record(self, x, *args, **kwargs):
        widths.append(x.shape[-1])
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(RowEmbedding, "forward", record)
    X = pd.DataFrame(np.random.default_rng(5).normal(size=(14, 513))).add_prefix("f")
    model = KumoTabularMediumModel(path=str(tmp_path), name="WideKumo", problem_type="binary")
    model.fit(X=X.iloc[:12], y=pd.Series([0, 1] * 6), num_cpus=1, num_gpus=0)
    assert np.isfinite(model.predict_proba(X.iloc[12:])).all()
    assert widths and set(widths) == {513}


def test_regression_preserves_target_units(tmp_path, local_weights):
    X = pd.DataFrame(np.random.default_rng(8).normal(size=(14, 3)), columns=list("abc"))
    predictions = []
    for offset in [0, 500]:
        model = KumoTabularMediumModel(
            path=str(tmp_path), name=f"Kumo{offset}", problem_type="regression"
        )
        model.fit(
            X=X.iloc[:12], y=pd.Series(np.arange(12, dtype=float) + offset), num_cpus=1, num_gpus=0
        )
        predictions.append(model.predict(X.iloc[12:]))
    np.testing.assert_allclose(predictions[1] - predictions[0], 500, atol=1e-4)


def test_explicit_gpu_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        KumoTabularMediumModel()._fit(None, None, num_gpus=1)


def test_registry_and_foundation_model_category():
    from tabbench_bio import dashboard, site

    key = "KUMO-TABULAR-MEDIUM"
    assert _resolve_hyperparameters([key], 1) == {KumoTabularMediumModel: [{"ag.num_gpus": 1}]}
    assert dashboard.MODEL_CATEGORY[key] == site.MODEL_CATEGORY[key] == "Tabular Foundation"


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_autogluon_refits_all_training_rows(tmp_path, local_weights, task):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(30, 5))).add_prefix("f")
    train = X.iloc[:24].copy()
    train["target"] = np.arange(24) % 2 if task == "binary" else np.arange(24, dtype=float)
    predictor = TabularPredictor(label="target", problem_type=task, path=str(tmp_path / task))
    predictor.fit(
        train_data=train,
        hyperparameters={KumoTabularMediumModel: {}},
        num_cpus=1,
        num_gpus=0,
        num_bag_folds=0,
        fit_weighted_ensemble=False,
        refit_full=True,
        set_best_to_refit_full=True,
    )
    assert predictor.model_best.endswith("_FULL")
    model = predictor._trainer.load_model(predictor.model_best)
    assert model.model._cache is None
    assert all(len(context.x) == len(train) for context in model._contexts)
    assert model._recipe_execution is not None
    prediction = predictor.predict(X.iloc[24:]).to_numpy()
    assert np.isfinite(prediction).all()
    restored = TabularPredictor.load(predictor.path)
    np.testing.assert_allclose(restored.predict(X.iloc[24:]), prediction, atol=1e-5)
