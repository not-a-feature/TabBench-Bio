import pytest

pytest.importorskip("torch")
pytest.importorskip("autogluon.tabular")

from tabbench_bio.model import _resolve_hyperparameters
from tabbench_bio.models.tabfm import TabFMModel, _CloneableTabFM
from tabbench_bio.models.tabpfn_wide import TabPFNWideModel, _CloneableTabPFNWide
from tabbench_bio.predictions import CLASSIFICATION_ONLY_MODELS


class _TorchBackend:
    def to(self, device):
        self.device = str(device)


class _TabFMEstimator:
    def __init__(self):
        self.model = _TorchBackend()


class _TabPFNWideEstimator:
    def to(self, device):
        self.device = str(device)


def test_tabfm_reports_the_device_selected_for_fit():
    model = TabFMModel()
    model._fit_device = "cuda"

    assert model.get_device() == "cuda"


def test_tabpfn_wide_reports_the_device_selected_for_fit():
    model = TabPFNWideModel()
    model._fit_device = "cpu"

    assert model.get_device() == "cpu"


def test_tabfm_cloneable_moves_its_backend():
    model = _CloneableTabFM(device="cuda")
    model.model_ = _TabFMEstimator()

    assert model.to("cpu") is model
    assert model.device == "cpu"
    assert model.model_.model.device == "cpu"


def test_tabpfn_wide_cloneable_moves_its_estimator():
    model = _CloneableTabPFNWide(device="cuda")
    model.model_ = _TabPFNWideEstimator()

    assert model.to("cpu") is model
    assert model.device == "cpu"
    assert model.model_.device == "cpu"


def test_tabpfn_wide_5k_ne3_resolves_the_regular_5k_checkpoint():
    hyperparameters = _resolve_hyperparameters(["TABPFN-WIDE-5K-NE3"], num_gpus=1)

    assert hyperparameters == {
        TabPFNWideModel: [
            {
                "ag.num_gpus": 1,
                "model_name": "wide-v2-5k",
                "n_estimators": 3,
            }
        ]
    }
    assert "TABPFN-WIDE-5K-NE3" in CLASSIFICATION_ONLY_MODELS
