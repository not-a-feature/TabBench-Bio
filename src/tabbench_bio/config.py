"""Configuration loading and validation for the benchmark pipeline."""

import json
import os

#: Every key a benchmark config must declare. The config file is the complete, explicit
#: record of a run: nothing defaults implicitly, so a missing or misspelled key is an
#: error rather than a silent fallback to a code default. Optional features are listed
#: here too and must be set explicitly (``null`` / ``[]`` / ``{}``).
REQUIRED_KEYS = frozenset(
    {
        # data + models
        "datasets_classification",
        "datasets_regression",
        "models",
        # split + repetitions
        "test_size",
        "random_state",
        "n_repetitions",
        "cv_folds",
        "min_samples_per_class",
        "group_regression_splits",
        "bio_max_features",
        "max_classes",
        "train_subsample",
        "model_limits",
        "model_overrides",
        # io
        "cache_dir",
        "output_dir",
        # training
        "autogluon_time_limit",
        "autogluon_presets",
        "optimize",
        "ensemble",
        "num_hpo_trials",
        "subsample",
        "nan_policy",
        # evaluation
        "exclude_keys",
        "exclude_datasets",
        "exclude_targets",
    }
)


def resolve_list(value, base_dir):
    """Resolve a dataset/model list spec to a Python list.

    A list is returned as-is; a string is a path to a JSON array file, resolved
    relative to *base_dir* unless absolute. ``None`` passes through (the caller's
    "use the default set" sentinel).
    """
    if value is None or isinstance(value, list):
        return value
    path = value if os.path.isabs(value) else os.path.join(base_dir, value)
    with open(path) as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"List config file {path} must contain a JSON array.")
    return items


def parse_models(models):
    """Return ``(key, device)`` pairs; each GPU worker has exclusive use of its device."""
    pairs = []
    for m in models:
        key, device = m["key"], m["device"]
        assert device in ("gpu", "cpu"), (
            f"model {key!r}: device must be 'gpu'/'cpu', got {device!r}"
        )
        pairs.append((key, device))
    return pairs


def model_keys(models):
    """The plain model-key list the pipeline consumes, from either the object form
    (``configs/models/*.json``) or the bare-key lists the grid writes into per-cell configs."""
    return [m["key"] if isinstance(m, dict) else m for m in models]


def model_limits(models):
    """Map model key -> calibrated memory-prior metadata.

    ``max_cells`` is an empirical ``post-cap features x training rows`` boundary.
    ``enforce_memory_prior`` must be explicit: false keeps uncertain or stale observations
    available for reporting without skipping a fit. ``memory_prior_version`` identifies the
    calibration, so changing a boundary cannot silently reuse older skip records. Models
    without a prior are attempted normally.
    """
    limits = {}
    for model in models:
        if not isinstance(model, dict) or "max_cells" not in model:
            continue
        assert isinstance(model["max_cells"], int) and model["max_cells"] > 0, (
            f"model {model['key']!r}: max_cells must be a positive integer"
        )
        assert (
            "memory_prior_version" in model
            and isinstance(model["memory_prior_version"], int)
            and model["memory_prior_version"] > 0
        ), f"model {model['key']!r}: max_cells requires a positive memory_prior_version"
        assert "enforce_memory_prior" in model and isinstance(
            model["enforce_memory_prior"], bool
        ), f"model {model['key']!r}: max_cells requires explicit enforce_memory_prior"
        limits[model["key"]] = {
            "max_cells": model["max_cells"],
            "memory_prior_version": model["memory_prior_version"],
            "enforce_memory_prior": model["enforce_memory_prior"],
        }
    return limits


#: Fitting-regime keys a roster entry may override for one model. Anything else in an entry
#: is scheduling metadata or memory-prior metadata.
_OVERRIDE_KEYS = ("presets", "ensemble", "optimize", "num_hpo_trials")


def model_overrides(models):
    """Map model key -> the fitting-regime keys it overrides, from the object form.

    The run-wide regime in the config applies to every benchmarked model, so the leaderboard
    compares models under one budget rather than comparing tuning budgets. An entry that
    declares any of :data:`_OVERRIDE_KEYS` opts out of that regime and must be reported as a
    reference rather than a ranked peer — this is how ``AUTOGLUON`` is given bagging,
    stacking and portfolio ensembling while the single models stay at one default
    configuration. Entries declaring none are omitted."""
    overrides = {}
    for m in models:
        if isinstance(m, dict):
            declared = {k: m[k] for k in _OVERRIDE_KEYS if k in m}
            if declared:
                overrides[m["key"]] = declared
    return overrides


def load_config(config_path):
    """Load, validate, and normalise a benchmark JSON config file.

    The config must declare *every* key in :data:`REQUIRED_KEYS` and nothing else (keys
    prefixed with ``_`` are treated as comments and ignored), so a run is fully described
    by its config — no key silently falls back to a code default, and a typo'd key is
    rejected rather than quietly ignored.

    ``datasets_classification`` / ``datasets_regression`` / ``models`` each accept an
    inline list or a path to a JSON array file (relative to the config's directory); set
    a datasets key to ``null`` to load every registered dataset of that task.
    """
    with open(config_path) as f:
        config = json.load(f)

    keys = {k for k in config if not k.startswith("_")}
    missing = REQUIRED_KEYS - keys
    unknown = keys - REQUIRED_KEYS
    assert not missing, f"{config_path}: missing required config key(s): {sorted(missing)}"
    assert not unknown, f"{config_path}: unknown config key(s): {sorted(unknown)}"

    base_dir = os.path.dirname(os.path.abspath(config_path))
    config["dataset_names_classification"] = resolve_list(
        config["datasets_classification"], base_dir
    )
    config["dataset_names_regression"] = resolve_list(config["datasets_regression"], base_dir)
    # Device tags (if any) are scheduling metadata only; the pipeline consumes plain keys.
    config["models"] = model_keys(resolve_list(config["models"], base_dir))

    return config


def cell_name(cap: int | None, n_train: int | None) -> str:
    """Name a feature/sample cell; full-sample cells omit the sample suffix."""
    name = f"cap_{'full' if cap is None else cap}"
    return name if n_train is None else f"{name}_n{n_train}"


def config_for_cell(
    cap,
    n_train,
    *,
    datasets,
    datasets_regression,
    models,
    limits,
    overrides,
    n_rep,
    cv_folds,
    time_limit,
    out_dir,
    cache_dir,
    test_size,
    random_state,
    min_samples_per_class,
):
    # cv_folds set => stratified k-fold per cell x dataset (n_repetitions pinned to 1);
    # None => legacy holdout with n_rep repetitions.
    return {
        "datasets_classification": datasets,
        "datasets_regression": datasets_regression,
        "test_size": test_size,
        "n_repetitions": 1 if cv_folds is not None else n_rep,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "cache_dir": cache_dir,
        "output_dir": out_dir,
        "models": models,
        "model_limits": limits,
        "model_overrides": overrides,
        "autogluon_time_limit": time_limit,
        # Run-wide fitting regime: one library-default configuration per model, no HPO, no
        # bagging. Roster entries may override it (see config.model_overrides); AUTOGLUON does,
        # which is why it is reported as a best-case AutoML reference and not a ranked peer.
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "num_hpo_trials": 0,
        "min_samples_per_class": min_samples_per_class,
        "group_regression_splits": False,
        "bio_max_features": cap,
        "max_classes": None,
        # Sample axis: cap TRAINING rows (stratified, train-only). null = all rows.
        "train_subsample": n_train,
        "subsample": None,
        # KNN's AutoGluon preprocessor drops every column carrying a NaN, which on the sparse
        # metagenomic abundance matrices leaves it nothing to fit ("No valid features to train
        # KNeighbors"); an explicit train-fitted median makes the unit measure the model.
        "nan_policy": {"default": "native", "KNN": "median"},
        "exclude_keys": [],
        "exclude_datasets": [],
        "exclude_targets": [],
    }
