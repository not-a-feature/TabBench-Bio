"""Released FTFM backend with small local weights; no network or GPU required."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("ftfm")

import torch
from autogluon.tabular import TabularPredictor
from ftfm import FTFM, FTFMConfig
from safetensors.torch import save_file

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models import ftfm
from tabbench_bio.models.ftfm import CHECKPOINT_REVISION, FTFMModel


@pytest.fixture
def local_weights(tmp_path, monkeypatch):
    config = FTFMConfig(
        dim=16,
        num_heads=2,
        num_layers=2,
        label_add=True,
        block_ffn=2,
        latent_ffn=2,
        equivariant_head=True,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = FTFM(config).eval()
    save_file(
        {k: v.clone().contiguous() for k, v in model.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(config),
                "checkpoint_had_ema": True,
                "weight_selection": "weight_ema",
            }
        )
    )

    def download(repo_id, revision, allow_patterns):
        assert repo_id == "josifgrabocka/ftfm"
        assert revision == CHECKPOINT_REVISION
        assert allow_patterns == ["config.json", "model.safetensors"]
        return str(tmp_path)

    monkeypatch.setattr(ftfm, "snapshot_download", download)


@pytest.mark.parametrize("classes", [2, 3, 12])
def test_predict_restore_and_query_independence(tmp_path, local_weights, classes):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(28, 5))).add_prefix("f")
    X["category"] = pd.Series(["a", "b"] * 12 + ["new"] * 4, dtype="category")
    X.loc[2, "f0"] = np.nan
    model = FTFMModel(
        path=str(tmp_path), name="FTFM", problem_type="binary" if classes == 2 else "multiclass"
    )
    model.fit(X=X.iloc[:24], y=pd.Series(np.arange(24) % classes), num_cpus=1, num_gpus=0)
    assert model.model.n_estimators == 8
    assert model.model.view_batch_size == 1
    assert model.model.random_state == 42
    actual = model.predict_proba(X.iloc[24:])
    assert actual.shape == ((4,) if classes == 2 else (4, classes))
    assert np.isfinite(actual).all()
    restored = FTFMModel.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[24:]), actual, atol=1e-6)
    np.testing.assert_allclose(model.predict_proba(X.iloc[24:25]), actual[:1], atol=1e-6)


def test_no_adapter_feature_cap(tmp_path, local_weights):
    X = pd.DataFrame(np.random.default_rng(5).normal(size=(14, 513))).add_prefix("f")
    model = FTFMModel(path=str(tmp_path), name="WideFTFM", problem_type="binary")
    model.fit(X=X.iloc[:12], y=pd.Series([0, 1] * 6), num_cpus=1, num_gpus=0)
    assert model.model.n_features_in_ == 513
    assert np.isfinite(model.predict_proba(X.iloc[12:])).all()


def test_explicit_gpu_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        FTFMModel(problem_type="binary")._fit(None, None, num_gpus=1)


def test_registration_and_full_grid():
    from tabbench_bio.model_registry import MODEL_REGISTRY
    from tabbench_bio.predictions import CLASSIFICATION_ONLY_MODELS
    from tabbench_bio.resources import resource_root

    assert _resolve_hyperparameters(["FTFM"], 1) == {FTFMModel: [{"ag.num_gpus": 1}]}
    assert "FTFM" in CLASSIFICATION_ONLY_MODELS
    spec = MODEL_REGISTRY["FTFM"]
    assert spec.training_data_overlap and spec.max_features is None and spec.max_classes is None
    grid = json.loads((resource_root() / "configs/grid_sweep_all.json").read_text())
    roster = json.loads((resource_root() / "configs/models/all.json").read_text())
    assert any(row["key"] == "FTFM" for row in roster)
    assert "FTFM" not in grid["model_cells"]


def test_autogluon_refits_all_rows(tmp_path, local_weights):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(30, 4))).add_prefix("f")
    train = X.iloc[:24].copy()
    train["target"] = np.arange(24) % 2
    predictor = TabularPredictor(label="target", path=str(tmp_path / "predictor"))
    predictor.fit(
        train_data=train,
        hyperparameters={FTFMModel: {}},
        num_cpus=1,
        num_gpus=0,
        num_bag_folds=0,
        fit_weighted_ensemble=False,
        refit_full=True,
        set_best_to_refit_full=True,
    )
    assert predictor.model_best.endswith("_FULL")
    model = predictor._trainer.load_model(predictor.model_best)
    assert model.model.n_features_in_ == 4
    assert np.isfinite(predictor.predict_proba(X.iloc[24:])).all().all()
