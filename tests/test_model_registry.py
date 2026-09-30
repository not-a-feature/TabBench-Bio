import json
from pathlib import Path

from tabbench_bio import dashboard, site
from tabbench_bio.config import parse_models
from tabbench_bio.model_constraints import REGULAR_MAX_FEATURES
from tabbench_bio.model_registry import MODEL_REGISTRY, canonical_model_key, model_entry
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
