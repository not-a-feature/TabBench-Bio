import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models import tabpfn_v3_5
from tabbench_bio.models.tabpfn_v3 import TabPFNV3Model
from tabbench_bio.models.tabpfn_v3_5 import TabPFNV35Model, _estimator


@pytest.mark.parametrize("task,n_classes", [("binary", 2), ("multiclass", 12), ("regression", 0)])
def test_autogluon_fit_and_predict_contract(tmp_path, monkeypatch, task, n_classes):
    backend = DummyClassifier(strategy="prior") if n_classes else DummyRegressor()
    monkeypatch.setattr(tabpfn_v3_5, "_estimator", lambda *args: backend)
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(48, 4)))
    X.columns = ["a", "b", "c", "d"]
    y = pd.Series(np.arange(48) % n_classes if n_classes else np.arange(48, dtype=float))
    model = TabPFNV35Model(path=str(tmp_path), name="V35", problem_type=task)
    model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
    assert model.predict(X.iloc[:4]).shape == (4,)
    assert np.isfinite(model.predict_proba(X.iloc[:4])).all()


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_v35_factory_selects_explicit_version_and_settings(monkeypatch, problem_type):
    calls = []

    class Classifier:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls.append((cls, version, kwargs))
            return cls()

    class Regressor(Classifier):
        pass

    class Versions:
        V3_5 = "v3.5"

    tabpfn = ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = Classifier
    tabpfn.TabPFNRegressor = Regressor
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = Versions
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)

    _estimator(problem_type, "cuda", [2])

    assert calls == [
        (
            Regressor if problem_type == "regression" else Classifier,
            "v3.5",
            {
                "device": "cuda",
                "n_estimators": 8,
                "categorical_features_indices": [2],
                "ignore_pretraining_limits": True,
                "random_state": 0,
            },
        )
    ]


@pytest.mark.parametrize("num_gpus", [0, 1])
def test_v3_and_v35_have_distinct_registry_entries(num_gpus):
    resolved = _resolve_hyperparameters(["TABPFN-V3", "TABPFN-V3.5"], num_gpus)
    assert set(resolved) == {TabPFNV3Model, TabPFNV35Model}
    assert resolved[TabPFNV35Model] == [{"ag.num_gpus": 1} if num_gpus else {}]
    assert TabPFNV3Model()._get_default_auxiliary_params()["max_features"] == 10_000
    assert TabPFNV35Model()._get_default_auxiliary_params()["max_features"] == 20_000


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_v35_checkpoint_survives_sklearn_cloning(problem_type):
    pytest.importorskip("tabpfn")
    from tabpfn.constants import ModelVersion

    if "V3_5" not in ModelVersion.__members__:
        pytest.skip("Requires the tabpfn35 extra (TabPFN 9.0.0)")
    model = clone(_estimator(problem_type, "cpu", None))
    assert Path(model.model_path).name == "tabpfn-v3.5-20260909.safetensors"
    assert model.n_estimators == 8
    assert model.random_state == 0


def test_gpu_request_cannot_fall_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        TabPFNV35Model()._fit(None, None, num_gpus=1)


@pytest.mark.parametrize("many_class", [False, True])
def test_device_changes_reach_the_backend(many_class):
    class Backend:
        def to(self, device):
            self.device = device

        def set_params(self, *, device):
            self.device = device

    backend = Backend()
    model = TabPFNV35Model(problem_type="multiclass")
    model.num_classes = 11 if many_class else 3
    if many_class:
        from tabpfn_extensions.many_class import ManyClassClassifier

        model.model = ManyClassClassifier(estimator=backend, alphabet_size=10)
    else:
        model.model = backend
    model._fit_device = "cuda"
    model._set_device("cpu")
    assert model.get_device() == "cpu"
    assert backend.device == "cpu"


def test_v35_roster_and_reference_config_agree():
    root = Path(__file__).resolve().parents[1]
    roster = json.loads((root / "configs/models/all.json").read_text())
    model = next(row for row in roster if row["key"] == "TABPFN-V3.5")
    reference = json.loads((root / "configs/grid_tabpfn_v3_5.json").read_text())
    assert model == {"key": "TABPFN-V3.5", "device": "gpu", "environment": "tabpfn35"}
    assert (
        next(row for row in roster if row["key"] == "TABPFN-WIDE")["environment"]
        != model["environment"]
    )
    assert reference["models"] == [model]
    assert reference["caps"] == [10_000]
    assert reference["samples"] == [100]
