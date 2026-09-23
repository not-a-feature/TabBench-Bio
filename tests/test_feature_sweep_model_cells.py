"""Tests for sparse, explicitly cell-scoped reference models."""

from scripts.feature_sweep import _models_for_cell

MODELS = ["RF", {"key": "AUTOGLUON", "device": "gpu", "solo": True}]
MODEL_CELLS = {"AUTOGLUON": ["cap_10000_n100", "cap_full"]}


def _keys(models):
    return [entry if isinstance(entry, str) else entry["key"] for entry in models]


def test_autogluon_runs_only_in_the_two_prespecified_cells():
    assert _keys(_models_for_cell(MODELS, MODEL_CELLS, 10_000, 100)) == ["RF", "AUTOGLUON"]
    assert _keys(_models_for_cell(MODELS, MODEL_CELLS, None, None)) == ["RF", "AUTOGLUON"]
    assert _keys(_models_for_cell(MODELS, MODEL_CELLS, 2_000, 100)) == ["RF"]
    assert _keys(_models_for_cell(MODELS, MODEL_CELLS, 10_000, 50)) == ["RF"]


def test_unrestricted_models_still_run_across_the_grid():
    assert _keys(_models_for_cell(MODELS, {}, 2_000, 20)) == ["RF", "AUTOGLUON"]
