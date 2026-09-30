"""Versioned, finite grids and reproducible training-only selection splits."""

import hashlib
import json
import re

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid, train_test_split

TUNABLE_MODELS = frozenset({"RF", "XT", "LR", "KNN", "GBM", "CAT", "XGB", "NN_TORCH"})


def tuning_specs(models):
    """Extract explicit tuned variants from a model roster."""
    specs = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        assert ("base_model" in entry) == ("tuning" in entry), (
            "A tuned variant requires both base_model and tuning"
        )
        if "tuning" in entry:
            assert entry["key"] not in specs, "Duplicate tuned model key"
            assert "base_model" not in entry["tuning"], "Declare base_model beside tuning"
            spec = {"base_model": entry["base_model"], **entry["tuning"]}
            validate_tuning(entry["key"], spec)
            specs[entry["key"]] = spec
    return specs


def validate_tuning(key, spec):
    required = {
        "base_model",
        "protocol",
        "selection",
        "validation_fraction",
        "seed",
        "search_fraction",
        "grid",
    }
    assert set(spec) == required, f"Tuning fields must be {sorted(required)}"
    assert spec["base_model"] in TUNABLE_MODELS, "Unsupported tuning parent"
    assert re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]*", key) and key != spec["base_model"]
    assert isinstance(spec["protocol"], str) and spec["protocol"].strip(), (
        "Missing protocol version"
    )
    assert spec["selection"] == "inner_holdout", "Only inner_holdout is supported"
    assert 0 < spec["validation_fraction"] < 1
    assert 0 < spec["search_fraction"] < 1, "Reserve part of the total budget for refitting"
    assert type(spec["seed"]) is int and spec["seed"] >= 0
    for task in ("classification", "regression"):
        candidates(spec, task)


def candidates(spec, task):
    """Expand a shared grid, or explicit classification/regression grids, in fixed order."""
    grid = spec["grid"]
    if isinstance(grid, dict) and set(grid) == {"classification", "regression"}:
        grid = grid[task]
    expanded = list(ParameterGrid(grid))
    assert expanded and {} in expanded, "Include {} as a library-default candidate"
    encodings = [json.dumps(p, sort_keys=True, allow_nan=False) for p in expanded]
    assert len(set(encodings)) == len(encodings), "Duplicate grid candidates"
    assert all(not k.startswith("ag") for p in expanded for k in p), (
        "Grid entries may only change estimator parameters, not AutoGluon controls"
    )
    return expanded


def tuning_fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selection_split(frame, groups, spec, classification):
    """Select one reproducible holdout; fail when class/group separation is impossible."""
    assert frame.index.is_unique, "Training row identifiers must be unique"
    y = frame["target"]
    positions = np.arange(len(frame))
    if groups is not None:
        assert groups.index.equals(frame.index) and not groups.isna().any()
    if groups is not None and groups.nunique() < len(groups):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=spec["validation_fraction"], random_state=spec["seed"]
        )
        train, valid = next(splitter.split(frame, y, groups))
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[valid])
    else:
        train, valid = train_test_split(
            positions,
            test_size=spec["validation_fraction"],
            random_state=spec["seed"],
            stratify=y if classification else None,
        )
    if classification:
        assert set(y.iloc[train]) == set(y) == set(y.iloc[valid]), (
            "Inner split must contain every class in both partitions; use a larger sample budget"
        )
    assert len(train) >= 2 and len(valid) >= 2, "Inner split is too small"
    return train, valid
