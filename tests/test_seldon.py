"""Exercise both adapters with the real backend and small local checkpoints."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("neuralk_oss")

import torch
from autogluon.tabular import TabularPredictor
from neuralk_oss import Seldon, SeldonRegressor
from neuralk_oss.model import seldon as backend
from neuralk_oss.model.config import SeldonConfig

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models.seldon import (
    CHECKPOINT_REVISIONS,
    REGRESSION_CHECKPOINT_REVISIONS,
    SeldonMiniModel,
    SeldonModel,
    _FullFeatureRecipe,
)


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
    for task, head in (
        ("classification", {"task": "classification", "max_classes": 10}),
        ("regression", {"task": "regression", "num_quantiles": 7}),
    ):
        settings = config.model_copy(
            update={"head": type(config).model_validate({**config.model_dump(), "head": head}).head}
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            Seldon(settings).save_pretrained(tmp_path / task)

    resolve_checkpoint = backend.resolve_checkpoint

    def resolve(name_or_path, *, task=None, revision=None, **kwargs):
        name = (
            str(name_or_path).split("/")[-1].removesuffix("-classifier").removesuffix("-regressor")
        )
        revisions = (
            REGRESSION_CHECKPOINT_REVISIONS if task == "regression" else CHECKPOINT_REVISIONS
        )
        assert revision == revisions[name]
        return resolve_checkpoint(tmp_path / task, task=task)

    monkeypatch.setattr(backend, "resolve_checkpoint", resolve)


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
@pytest.mark.parametrize("budget", [0, 4096, 2**30])
def test_regression_matches_released_cache_and_restores(tmp_path, local_weights, adapter, budget):
    torch.set_num_threads(1)
    X = pd.DataFrame(np.random.default_rng(12).normal(size=(32, 6))).add_prefix("f")
    X["category"] = pd.Categorical(["a", "b"] * 16)
    X.iloc[2, 0] = np.nan
    y = pd.Series(np.linspace(-3, 5, 28) + np.random.default_rng(4).normal(size=28))
    reference = SeldonRegressor.from_pretrained(
        adapter.checkpoint,
        revision=REGRESSION_CHECKPOINT_REVISIONS[adapter.checkpoint],
        device="cpu",
        estimator_batch_size=1,
        query_chunk_size=16,
        max_cache_gb=budget / 2**30,
    )
    recipe = reference.estimator.recipe
    reference.estimator.recipe = _FullFeatureRecipe(**vars(recipe))
    model = adapter(path=str(tmp_path), name="SeldonRegressor", problem_type="regression")
    model.fit(X=X.iloc[:28], y=y, num_cpus=1, num_gpus=0)
    model.model.estimator.cache_budget = budget
    reference.fit(X.iloc[:28], y)
    expected = reference.predict(X.iloc[28:])
    actual = model.predict(X.iloc[28:])
    assert model.model.n_estimators == reference.n_estimators == 4
    assert model.model.clip_margin == reference.clip_margin == 0.1
    assert model.model.estimator.recipe.half_shuffle is True
    assert actual.shape == (4,) and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(model.predict(X.iloc[28:29]), actual[:1], atol=1e-6)
    restored = adapter.load(model.save())
    assert restored.model.estimator._encoded_bytes == 0
    np.testing.assert_array_equal(restored.predict(X.iloc[28:]), actual)
    assert model.model.estimator._encoded_bytes <= budget


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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
@pytest.mark.parametrize("classes", [2, 3, 12])
@pytest.mark.parametrize("budget", [0, 4096, 2**30])
def test_bounded_cache_matches_released_classifier(
    tmp_path, local_weights, classes, budget, device
):
    import pickle

    from neuralk_oss import SeldonClassifier

    from tabbench_bio.models.seldon_cache import BoundedSeldonICL

    torch.set_num_threads(1)
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(40, 9)))
    X["category"] = pd.Series(["a", "b"] * 18 + ["new"] * 4, dtype="category")
    X.iloc[2, 3] = np.nan
    y = np.arange(36) % classes
    options = dict(
        revision=CHECKPOINT_REVISIONS["seldon-mini"],
        device=device,
        estimator_batch_size=1,
        query_chunk_size=16,
        recipe=_FullFeatureRecipe(half_shuffle=True),
    )
    reference = SeldonClassifier.from_pretrained("seldon-mini", **options)
    bounded = SeldonClassifier.from_pretrained("seldon-mini", **options)
    old = bounded.estimator
    bounded.estimator = BoundedSeldonICL(old.model, old.recipe, query_chunk_size=16).to(device)
    bounded.estimator.cache_budget = budget
    reference.fit(X.iloc[:36], y)
    bounded.fit(X.iloc[:36], y)
    expected = reference.predict_proba(X.iloc[36:])
    actual = bounded.predict_proba(X.iloc[36:])
    np.testing.assert_array_equal(actual, expected)
    # Also compare with upstream automatic member batching, whose kernels can
    # differ in floating-point rounding from sequential member execution.
    native = SeldonClassifier.from_pretrained(
        "seldon-mini",
        revision=CHECKPOINT_REVISIONS["seldon-mini"],
        device=device,
        recipe=_FullFeatureRecipe(half_shuffle=True),
    )
    native.fit(X.iloc[:36], y)
    native_proba = native.predict_proba(X.iloc[36:])
    np.testing.assert_allclose(actual, native_proba, atol=5e-4 if device == "cuda" else 1e-6)
    np.testing.assert_array_equal(actual.argmax(axis=1), native_proba.argmax(axis=1))
    assert bounded.estimator._encoded_bytes <= budget
    np.testing.assert_array_equal(bounded.predict_proba(X.iloc[36:]), expected)
    restored = pickle.loads(pickle.dumps(bounded))
    assert restored.estimator._encoded_bytes == 0
    np.testing.assert_array_equal(restored.predict_proba(X.iloc[36:]), expected)
    np.testing.assert_allclose(bounded.predict_proba(X.iloc[36:37]), expected[:1], atol=1e-6)


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_headroom_uses_tightest_ancestor(monkeypatch, version):
    from pathlib import Path

    from tabbench_bio.models.seldon_cache import _headroom

    if version == 1:
        records = "10:memory:/job/step\n0::/system.slice/slurmd.service"
        root = "/sys/fs/cgroup/memory"
        limit, usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
        unlimited = str(2**63 - 4096)
    else:
        records = "0::/job/step"
        root = "/sys/fs/cgroup"
        limit, usage = "memory.max", "memory.current"
        unlimited = "max"
    files = {
        "/proc/self/cgroup": records,
        f"{root}/job/step/{limit}": "200",
        f"{root}/job/step/{usage}": "50",
        f"{root}/job/{limit}": "160",
        f"{root}/job/{usage}": "140",
        f"{root}/{limit}": unlimited,
    }
    monkeypatch.setattr(Path, "read_text", lambda p: files[str(p).replace("\\", "/")])
    monkeypatch.setattr(Path, "is_file", lambda p: str(p).replace("\\", "/") in files)
    assert _headroom() == 20
