"""Tuning specifications, split isolation and result compatibility."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabbench_bio import cli
from tabbench_bio.config import load_config, resolve_nan_policy
from tabbench_bio.result_store import _merge_cell_config
from tabbench_bio.sample_fallback import _assert_compatible_configs
from tabbench_bio.tuning import candidates, selection_split, tuning_fingerprint, tuning_specs

ROSTER = json.loads((Path(__file__).parents[1] / "configs/models/rf_tuned.json").read_text())
SPEC = tuning_specs(ROSTER)["RF-TUNED"]


@pytest.mark.parametrize(
    "policies,expected",
    [
        (None, "native"),
        ({"default": "zero"}, "zero"),
        ({"RF": "median", "default": "zero"}, "median"),
        ({"RF-TUNED": "mean", "RF": "median", "default": "zero"}, "mean"),
    ],
)
def test_variant_nan_policy_precedence(policies, expected):
    config = {"nan_policy": policies, "model_tuning": {"RF-TUNED": SPEC}}
    assert resolve_nan_policy(config, "RF-TUNED") == expected


def test_fallback_rejects_different_inherited_nan_policies():
    config = {"models": ["RF-TUNED"], "model_tuning": {"RF-TUNED": SPEC}}
    target = {**config, "nan_policy": {"RF": "median"}}
    source = {**config, "nan_policy": {"RF": "mean"}}
    with pytest.raises(AssertionError, match="non-sample config differs"):
        _assert_compatible_configs(target, source, "RF-TUNED")
    source["nan_policy"]["RF-TUNED"] = "median"
    _assert_compatible_configs(target, source, "RF-TUNED")


@pytest.mark.parametrize("parent,count", [("RF", 8), ("XT", 13), ("XGB", 20), ("LR", 6)])
def test_shipped_grids_load_for_both_tasks(tmp_path, debug_config, parent, count):
    roster = Path(__file__).parents[1] / f"configs/models/{parent.lower()}_tuned.json"
    key = f"{parent}-TUNED"
    spec = tuning_specs(json.loads(roster.read_text()))[key]
    config = {k: v for k, v in debug_config.items() if not k.startswith("dataset_names")}
    config.update(datasets_classification=[], datasets_regression=[], models=str(roster))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    loaded = load_config(path)
    assert loaded["models"] == [key]
    assert loaded["model_tuning"] == {key: spec}
    for task in ("classification", "regression"):
        grid = candidates(spec, task)
        assert {} not in grid and len(grid) == count


def test_grid_omits_default_and_has_stable_candidates():
    grid = candidates(SPEC, "classification")
    assert {} not in grid and len(grid) == 8
    assert grid == candidates(SPEC, "regression")
    task_spec = {
        **SPEC,
        "grid": {
            "classification": [{}, {"max_depth": [2]}],
            "regression": [{}, {"max_depth": [4]}],
        },
    }
    assert candidates(task_spec, "regression")[1] == {"max_depth": 4}
    assert tuning_fingerprint(SPEC) != tuning_fingerprint(task_spec)


def test_mlp_grid_uses_registered_parent_and_supported_architecture_parameters():
    roster = json.loads((Path(__file__).parents[1] / "configs/models/mlp_tuned.json").read_text())
    spec = tuning_specs(roster)["NN_TORCH-TUNED"]
    assert spec["base_model"] == "NN_TORCH" and roster[0]["device"] == "gpu"
    for task in ("classification", "regression"):
        grid = candidates(spec, task)
        assert {} not in grid and len(grid) == 16
        assert {c["num_layers"] for c in grid} == {2, 4}
        assert {c["hidden_size"] for c in grid} == {64, 128}
        assert {c["learning_rate"] for c in grid} == {0.0003, 0.001}
        assert {c["dropout_prob"] for c in grid} == {0.0, 0.2}


@pytest.mark.parametrize("grid", [[], [{}, {}], [{"max_depth": []}], [{"ag_args": [{}]}]])
def test_invalid_grids_are_rejected(grid):
    entry = copy.deepcopy(ROSTER[0])
    entry["tuning"]["grid"] = grid
    with pytest.raises((AssertionError, ValueError)):
        tuning_specs([entry])


def test_grouped_holdout_is_deterministic_and_disjoint():
    frame = pd.DataFrame({"x": np.arange(80), "target": [0, 1] * 40}, index=range(100, 180))
    groups = pd.Series(np.repeat(np.arange(20), 4), index=frame.index)
    train, valid = selection_split(frame, groups, SPEC, True)
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[valid])
    assert set(train) | set(valid) == set(range(80))
    again = selection_split(frame, groups, SPEC, True)
    np.testing.assert_array_equal(valid, again[1])


def test_missing_class_in_group_holdout_fails():
    frame = pd.DataFrame({"target": [0] * 10 + [1] * 10})
    groups = frame["target"].copy()
    with pytest.raises(AssertionError, match="every class"):
        selection_split(frame, groups, SPEC, True)


def test_loading_roster_keeps_resolved_tuning(tmp_path, debug_config):
    config = {k: v for k, v in debug_config.items() if not k.startswith("dataset_names")}
    config.update(datasets_classification=[], datasets_regression=[], models="models.json")
    (tmp_path / "models.json").write_text(json.dumps(ROSTER))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    loaded = load_config(path)
    assert loaded["models"] == ["RF-TUNED"]
    assert loaded["model_tuning"] == {"RF-TUNED": SPEC}


def test_merge_tuned_with_default_but_reject_changed_protocol():
    base = {"models": ["RF"], "model_limits": {}, "model_overrides": {}}
    tuned = {**base, "models": ["RF-TUNED"], "model_tuning": {"RF-TUNED": SPEC}}
    merged = _merge_cell_config(base, tuned, "reference")
    assert merged["models"] == ["RF", "RF-TUNED"]
    assert merged["model_tuning"] == tuned["model_tuning"]
    changed = copy.deepcopy(tuned)
    changed["model_tuning"]["RF-TUNED"]["seed"] += 1
    with pytest.raises(AssertionError, match="model_tuning"):
        _merge_cell_config(tuned, changed, "reference")


def test_cli_accepts_tuning_roster(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "run_model", captured.append)
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "RF-TUNED", "--model-config", "grid.json"])
    cli.main()
    assert captured[0].model_config == "grid.json"
