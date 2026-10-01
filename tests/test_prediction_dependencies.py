import pytest

from tabbench_bio import predictions


def test_missing_autogluon_stops_before_results_or_dataset_loading(monkeypatch):
    monkeypatch.setattr(predictions, "AutoGluonModel", None)
    cause = ImportError("missing backend")
    monkeypatch.setattr(predictions, "_MODEL_IMPORT_ERROR", cause)
    monkeypatch.setattr(predictions, "ResultRepository", lambda *a: pytest.fail("Wrote results"))
    with pytest.raises(ImportError, match="Prediction runs require AutoGluon") as error:
        predictions.compute_predictions({})
    assert error.value.__cause__ is cause
    assert callable(predictions.prepare_splits)


def test_legacy_model_subsampling_is_rejected_before_writing(monkeypatch):
    monkeypatch.setattr(predictions, "AutoGluonModel", object)
    monkeypatch.setattr(predictions, "ResultRepository", lambda *a: pytest.fail("Wrote results"))
    with pytest.raises(AssertionError, match="Per-model subsampling"):
        predictions.compute_predictions({"subsample": {"combinations": {}}})
