"""TabOmics adapter integration with small local weights."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")
pytest.importorskip("tabomics")

import torch
from tabomics import Classifier
from tabomics.model import TabOmicsConfig
from tabomics.model import TabOmicsModel as Backend

from tabbench_bio.models import tabomics
from tabbench_bio.models.tabomics import TabOmicsModel


@pytest.mark.parametrize("classes,features", [(2, 6), (3, 513)])
def test_predictions_restore_and_query_independence(tmp_path, monkeypatch, classes, features):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        backend = Backend(
            TabOmicsConfig(
                embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2
            )
        )

    def classifier(model, device, n_estimators):
        assert model == "v2.5"
        return Classifier(model=backend, device=device, n_estimators=n_estimators)

    monkeypatch.setattr(tabomics, "Classifier", classifier)
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(28, features))).add_prefix("f")
    X["category"] = pd.Series(["a", "b"] * 12 + ["new"] * 4, dtype="category")
    X.loc[2, "f0"] = np.nan
    model = TabOmicsModel(
        path=str(tmp_path), name="TabOmics", problem_type="binary" if classes == 2 else "multiclass"
    )
    model.fit(X=X.iloc[:24], y=pd.Series(np.arange(24) % classes), num_cpus=1, num_gpus=0)
    prediction = model.predict_proba(X.iloc[24:])
    assert prediction.shape == ((4,) if classes == 2 else (4, classes))
    assert model.model.n_features_in_ == features + 1
    assert np.isfinite(prediction).all()
    restored = TabOmicsModel.load(model.save())
    np.testing.assert_allclose(restored.predict_proba(X.iloc[24:]), prediction, atol=1e-6)
    np.testing.assert_allclose(model.predict_proba(X.iloc[24:25]), prediction[:1], atol=1e-6)
    assert model.model.preprocessor_.transform(X.iloc[24:]).shape[1] == features + 1


def test_registration_and_limits():
    from tabbench_bio.model import _resolve_hyperparameters
    from tabbench_bio.model_registry import MODEL_REGISTRY

    spec = MODEL_REGISTRY["TABOMICS"]
    assert spec.supported_tasks == ("classification",)
    assert spec.max_classes == 10
    assert spec.max_features is None
    assert _resolve_hyperparameters(["TABOMICS"], 1) == {TabOmicsModel: [{"ag.num_gpus": 1}]}
    assert MODEL_REGISTRY["TABOMICS-8E"].max_classes == 10
    assert _resolve_hyperparameters(["TABOMICS-8E"], 1) == {
        TabOmicsModel: [{"ag.num_gpus": 1, "n_estimators": 8}]
    }
