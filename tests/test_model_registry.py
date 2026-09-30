import json
from pathlib import Path

from tabbench_bio import dashboard, site
from tabbench_bio.config import parse_models
from tabbench_bio.model_registry import (
    MODEL_REGISTRY,
    REGULAR_MAX_FEATURES,
    canonical_model_key,
    model_entry,
)
from tabbench_bio.predictions import CLASSIFICATION_ONLY_MODELS


def test_shared_metadata_and_capabilities():
    assert site.MODEL_CATEGORY == dashboard.MODEL_CATEGORY
    assert site.MODEL_DISPLAY == dashboard.MODEL_DISPLAY
    assert site.MODEL_CATEGORY["TABM"] == "Deep Learning"
    for key, spec in MODEL_REGISTRY.items():
        assert spec.display and spec.category
        assert spec.device in {"cpu", "gpu"}
        assert spec.supported_tasks
        assert (key in CLASSIFICATION_ONLY_MODELS) == ("regression" not in spec.supported_tasks)
        assert (key in REGULAR_MAX_FEATURES) == (spec.max_features is not None)
        assert spec.max_classes is None or spec.max_classes >= 2


def test_roster_inherits_execution_defaults():
    root = Path(__file__).resolve().parents[1]
    roster = json.loads((root / "configs/models/all.json").read_text())
    assert all("device" not in row and "environment" not in row for row in roster)
    assert parse_models(roster) == [
        (row["key"], MODEL_REGISTRY[row["key"]].device) for row in roster
    ]
    assert model_entry({"key": "RF-TUNED", "base_model": "RF"})["environment"] == "standard"
    assert model_entry({"key": "TABPFNV35"})["environment"] == "tabpfn35"
    assert model_entry({"key": "RF", "device": "gpu"})["device"] == "gpu"


def test_aliases_and_checkpoint_overrides():
    assert canonical_model_key("tabpfnwide") == "TABPFN-WIDE"
    assert canonical_model_key("TABPFNV3") == "TABPFN-V3"
    spec = MODEL_REGISTRY["TABPFN-WIDE-5K-NE3"]
    assert dict(spec.hyperparameters) == {"model_name": "wide-v2-5k", "n_estimators": 3}
    assert spec.adapter == MODEL_REGISTRY["TABPFN-WIDE"].adapter


def test_class_limit_is_a_design_skip(tmp_path, monkeypatch, debug_config):
    from unittest.mock import Mock

    import pandas as pd

    from tabbench_bio import predictions
    from tabbench_bio.dataset import TaskType
    from tabbench_bio.result_store import ResultRepository

    frame = pd.DataFrame({"x": range(12), "target": range(12)})

    class Benchmark:
        _key_list = ["toy_0"]
        _task_type_list = [TaskType.Classification]

        def __len__(self):
            return 1

        def __iter__(self):
            yield frame, frame, "toy_0", TaskType.Classification

    debug_config.update(output_dir=str(tmp_path / "run"), models=["LIMIX-2"], train_subsample=None)
    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "get_seeds", lambda config: [0])
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    model = Mock(side_effect=AssertionError("Must skip before fitting"))
    monkeypatch.setattr(predictions, "AutoGluonModel", model)
    predictions.compute_predictions(debug_config)
    (attempt,) = ResultRepository(debug_config["output_dir"], debug_config).current_attempts()
    assert (attempt.status, attempt.reason) == ("skip", "class_limit")
    model.assert_not_called()
