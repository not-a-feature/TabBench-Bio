import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")

from tabbench_bio.models.tabpfn_v3 import (
    TabPFNV3Model,
    _classifier,
    _CloneableTabPFNV3,
    _regressor,
)


class _ModelVersion:
    V3 = "v3"
    V3_5 = "v3.5"


class _FakeEstimator:
    calls = []

    @classmethod
    def create_default_for_version(cls, version="v3.5", **kwargs):
        cls.calls.append((version, kwargs))
        return cls()


def test_factories_pin_tabpfn_v3_defaults(monkeypatch):
    tabpfn = ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = _FakeEstimator
    tabpfn.TabPFNRegressor = _FakeEstimator
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = _ModelVersion
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    _FakeEstimator.calls.clear()

    _classifier("cuda", [1, 3])
    _regressor("cpu", None)

    assert [version for version, _ in _FakeEstimator.calls] == ["v3", "v3"]
    classifier_kwargs = _FakeEstimator.calls[0][1]
    regressor_kwargs = _FakeEstimator.calls[1][1]
    assert classifier_kwargs == {
        "device": "cuda",
        "n_estimators": 8,
        "categorical_features_indices": [1, 3],
        "ignore_pretraining_limits": True,
        "random_state": 0,
    }
    assert regressor_kwargs["device"] == "cpu"
    assert regressor_kwargs["categorical_features_indices"] is None
    assert regressor_kwargs["n_estimators"] == 8
    assert regressor_kwargs["random_state"] == 0
    assert regressor_kwargs["ignore_pretraining_limits"] is True


@pytest.mark.parametrize(
    ("factory", "model_source"),
    [(_classifier, "get_classifier_v3"), (_regressor, "get_regressor_v3")],
)
def test_installed_tabpfn_resolves_v3_checkpoints(factory, model_source):
    pytest.importorskip("tabpfn")
    from tabpfn.model_loading import ModelSource

    sources = {
        "get_classifier_v3": ModelSource.get_classifier_v3,
        "get_regressor_v3": ModelSource.get_regressor_v3,
    }
    model = factory("cpu", None)

    assert Path(model.model_path).name == sources[model_source]().default_filename
    assert model.n_estimators == 8
    assert model.random_state == 0
    assert model.ignore_pretraining_limits is True


def test_cloneable_wrapper_delegates_autogluon_device_hooks():
    class FittedEstimator:
        devices_ = [type("Device", (), {"type": "cuda"})()]

        def to(self, device):
            self.target_device = device

    wrapper = _CloneableTabPFNV3(device="cuda")
    wrapper.model_ = FittedEstimator()

    assert wrapper.devices_[0].type == "cuda"
    assert wrapper.to("cpu") is wrapper
    assert wrapper.model_.target_device == "cpu"
    assert wrapper.device == "cpu"


def test_model_accepts_reference_cell_feature_budget():
    assert TabPFNV3Model()._get_default_auxiliary_params()["max_features"] == 10_000
