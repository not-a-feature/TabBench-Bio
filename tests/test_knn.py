import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.tabular")

from autogluon.tabular import TabularPredictor

from tabbench_bio.dataset import TaskType
from tabbench_bio.model import AutoGluonModel


def test_knn_fits_binary_fingerprints_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.integers(0, 2, size=(80, 16)).astype(float))
    frame.columns = [f"bit_{i}" for i in frame.columns]
    frame["target"] = np.tile(["inactive", "active"], 40)
    model = AutoGluonModel(
        ["KNN"],
        ensemble=False,
        optimize=False,
        task_type=TaskType.Classification,
        autogluon_time_limit=30,
        autogluon_path=str(tmp_path / "models"),
    )
    model.fit(frame.iloc[:60])
    test = frame.iloc[60:]
    probabilities = model.predict_proba(test)
    assert list(probabilities.columns) == model.predictor.class_labels
    assert probabilities.shape == (20, 2)
    assert np.isfinite(probabilities).all().all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    restored = TabularPredictor.load(model.predictor.path)
    pd.testing.assert_frame_equal(probabilities, restored.predict_proba(test))
