import sys
from types import ModuleType

import pytest

pytest.importorskip("autogluon", reason="TabPFN v3 integration requires the model extra")

from tabbench_bio.models.tabpfn_v3 import (
    TabPFNV3Model,
    _classifier,
    _CloneableTabPFNV3,
    _regressor,
)


class _ModelVersion:
    V3 = "v3"


class _FakeEstimator:
    calls = []

    @classmethod
    def create_default_for_version(cls, version, **kwargs):
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
