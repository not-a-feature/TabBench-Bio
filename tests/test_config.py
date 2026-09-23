"""Tests for config loading, validation, and list resolution."""

import json

import pytest

from tabbench_bio.config import REQUIRED_KEYS, load_config


def _complete_cfg(**overrides):
    """A config declaring every required key (the loader rejects partial configs)."""
    cfg = {k: None for k in REQUIRED_KEYS}
    cfg.update(
        {
            "datasets_classification": [],
            "datasets_regression": [],
            "models": ["RF"],
            "test_size": 0.2,
            "random_state": 42,
            "n_repetitions": 1,
            "min_samples_per_class": 5,
            "group_regression_splits": False,
            "bio_max_features": 1000,
            "cache_dir": ".cache",
            "output_dir": "results/test",
            "autogluon_time_limit": 60,
            "autogluon_presets": "medium_quality",
            "optimize": False,
            "ensemble": False,
            "num_hpo_trials": 0,
            "exclude_keys": [],
            "exclude_datasets": [],
            "exclude_targets": [],
        }
    )
    cfg.update(overrides)
    return cfg


def _write(tmp_path, cfg):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_load_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.json"))


def test_load_config_complete(tmp_path):
    loaded = load_config(_write(tmp_path, _complete_cfg()))
    assert loaded["models"] == ["RF"]
    assert loaded["dataset_names_classification"] == []
    assert loaded["dataset_names_regression"] == []


def test_load_config_rejects_missing_key(tmp_path):
    cfg = _complete_cfg()
    del cfg["bio_max_features"]
    with pytest.raises(AssertionError, match="missing required config key"):
        load_config(_write(tmp_path, cfg))


def test_load_config_rejects_unknown_key(tmp_path):
    # A typo'd key would otherwise silently default; the loader rejects it.
    cfg = _complete_cfg(group_regresion_splits=True)
    with pytest.raises(AssertionError, match="unknown config key"):
        load_config(_write(tmp_path, cfg))


def test_load_config_allows_comment_keys(tmp_path):
    loaded = load_config(_write(tmp_path, _complete_cfg(_doc="a comment")))
    assert loaded["models"] == ["RF"]


def test_load_config_datasets_null_pass_through(tmp_path):
    # null is the explicit "load every registered dataset" sentinel.
    cfg = _complete_cfg(datasets_classification=None, datasets_regression=None)
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded["dataset_names_classification"] is None
    assert loaded["dataset_names_regression"] is None


def test_load_config_resolves_list_file(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "clf.json").write_text(json.dumps(["OpenML-1138"]))
    cfg = _complete_cfg(datasets_classification="datasets/clf.json")
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded["dataset_names_classification"] == ["OpenML-1138"]
