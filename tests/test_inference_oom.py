"""Tests for bounded inference OOM retries and MITRA's initial sample limits."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tabbench_bio.models.mitra_limits import _train_with_actual_sample_limits
from tabbench_bio.predictions import _memory_prior_skip_details, _predict_with_oom_batching


def test_prediction_uses_complete_test_set_when_it_fits():
    data = pd.DataFrame({"x": range(6)}, index=[10, 11, 12, 13, 14, 15])
    calls = []

    def predict(batch):
        calls.append(len(batch))
        return batch["x"].rename("target")

    predictions, batch_size, retries = _predict_with_oom_batching(predict, data)

    assert calls == [6]
    assert predictions.equals(data["x"].rename("target"))
    assert batch_size == 6
    assert retries == 0


def test_prediction_halves_only_until_a_batch_fits():
    data = pd.DataFrame({"x": range(10)}, index=range(20, 30))
    calls = []

    def predict(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory while allocating a tensor")
        return batch["x"].rename("target")

    predictions, batch_size, retries = _predict_with_oom_batching(predict, data)

    assert calls == [10, 5, 2, 2, 2, 2, 2]
    assert predictions.equals(data["x"].rename("target"))
    assert batch_size == 2
    assert retries == 2


def test_prediction_halves_until_batch_size_one_then_raises():
    data = pd.DataFrame({"x": range(16)})
    calls = []

    def predict(batch):
        calls.append(len(batch))
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _predict_with_oom_batching(predict, data)

    assert calls == [16, 8, 4, 2, 1]


def test_prediction_does_not_retry_non_oom_errors():
    data = pd.DataFrame({"x": range(8)})
    calls = []

    def predict(batch):
        calls.append(len(batch))
        raise ValueError("invalid feature schema")

    with pytest.raises(ValueError, match="invalid feature schema"):
        _predict_with_oom_batching(predict, data)

    assert calls == [8]


def test_only_an_enforced_memory_prior_skips_a_unit():
    legacy = {"MODEL": {"max_cells": 1_000}}
    advisory = {
        "MODEL": {
            "max_cells": 1_000,
            "memory_prior_version": 2,
            "enforce_memory_prior": False,
        }
    }
    enforced = {
        "MODEL": {
            "max_cells": 1_000,
            "memory_prior_version": 2,
            "enforce_memory_prior": True,
        }
    }

    assert _memory_prior_skip_details(legacy, "MODEL", 20, 100) is None
    assert _memory_prior_skip_details(advisory, "MODEL", 20, 100) is None
    assert _memory_prior_skip_details(enforced, "MODEL", 10, 100) is None
    details = _memory_prior_skip_details(enforced, "MODEL", 20, 100)
    assert details == {
        "n_features": 100,
        "required_cells": 2_000,
        "memory_prior_max_cells": 1_000,
        "memory_prior_version": 2,
        "recommended_max_n_train": 10,
    }


def test_mitra_starts_from_actual_support_and_query_sizes():
    estimator = SimpleNamespace()
    captured = {}

    def create_config(task, dim_output, time_limit=None):
        return (
            SimpleNamespace(hyperparams={"max_samples_support": 8192, "max_samples_query": 1024}),
            object,
        )

    estimator._create_config = create_config

    def original_train(
        estimator,
        X_train,
        y_train,
        X_valid,
        y_valid,
        task,
        dim_output,
        n_classes=0,
        time_limit=None,
    ):
        cfg, _ = estimator._create_config(task, dim_output, time_limit)
        captured.update(cfg.hyperparams)
        return "trained"

    result = _train_with_actual_sample_limits(
        original_train,
        estimator,
        X_train=range(500),
        y_train=range(500),
        X_valid=range(125),
        y_valid=range(125),
        task="classification",
        dim_output=10,
    )

    assert result == "trained"
    assert captured == {"max_samples_support": 500, "max_samples_query": 125}
    assert estimator._create_config is create_config


def test_mitra_does_not_raise_smaller_upstream_limits():
    estimator = SimpleNamespace()
    captured = {}

    def create_config(task, dim_output, time_limit=None):
        return (
            SimpleNamespace(hyperparams={"max_samples_support": 256, "max_samples_query": 64}),
            object,
        )

    estimator._create_config = create_config

    def original_train(estimator, *args, **kwargs):
        cfg, _ = estimator._create_config("classification", 10)
        captured.update(cfg.hyperparams)

    _train_with_actual_sample_limits(
        original_train,
        estimator,
        X_train=range(500),
        y_train=range(500),
        X_valid=range(125),
        y_valid=range(125),
        task="classification",
        dim_output=10,
    )

    assert captured == {"max_samples_support": 256, "max_samples_query": 64}
