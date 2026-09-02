import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.feature_sweep import _write_cell_config
from tabbench_bio.dataset import TaskType
from tabbench_bio.evaluation import _filter_current_metric_rows
from tabbench_bio.io_utils import atomic_to_csv, atomic_write_json
from tabbench_bio.logging_utils import run_file_logger
from tabbench_bio.predictions import (
    MEMORY_RETRY_VERSION,
    _assert_ground_truth_compatible,
    _is_current_memory_failure_record,
    _is_stage_oom,
    _saved_predictions_are_valid,
    _write_skip_record,
)
from tabbench_bio.result_fork import fork_grid_results


def test_atomic_json_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"status": "pass"}\n', encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(TypeError):
        atomic_write_json(path, {"not_serializable": object()})

    assert path.read_bytes() == original


def test_atomic_csv_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("target\n1\n", encoding="utf-8")
    original = path.read_bytes()

    class BrokenFrame:
        def to_csv(self, temporary, *, index):
            Path(temporary).write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted writer")

    with pytest.raises(RuntimeError):
        atomic_to_csv(BrokenFrame(), path, index=False)

    assert path.read_bytes() == original


def test_skip_record_cannot_replace_a_pass(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    path = stats / "toy_MODEL.json"
    atomic_write_json(
        path,
        {
            "dataset": "toy",
            "model": "MODEL",
            "status": "pass",
            "n_train_samples": 100,
            "n_test_samples": 20,
        },
    )
    original = path.read_bytes()

    with pytest.raises(AssertionError):
        _write_skip_record(
            stats,
            "toy",
            "MODEL",
            100,
            20,
            "duplicate_cell",
            "duplicate",
        )

    assert path.read_bytes() == original


def test_retry_log_preserves_previous_attempt(tmp_path):
    path = tmp_path / "unit.log"
    path.write_text("first attempt\n", encoding="utf-8")

    with run_file_logger(str(path)):
        logging.getLogger("tabbench_bio.recovery_test").warning("second attempt")

    history = list((tmp_path / "history").glob("unit.*.log"))
    assert len(history) == 1
    assert history[0].read_text(encoding="utf-8") == "first attempt\n"
    assert "second attempt" in path.read_text(encoding="utf-8")


def test_memory_retry_version_targets_only_old_records():
    current = {
        "reason": "fit_oom",
        "memory_retry_version": MEMORY_RETRY_VERSION,
    }
    old = {"reason": "fit_oom", "memory_retry_version": MEMORY_RETRY_VERSION - 1}
    legacy = {"reason": "fit_error"}
    assert _is_current_memory_failure_record(current)
    assert not _is_current_memory_failure_record(old)
    assert not _is_current_memory_failure_record(legacy)
    advisory_limit = {
        "max_cells": 8_500_000,
        "memory_prior_version": 1,
        "enforce_memory_prior": False,
    }
    prior_skip = {
        "reason": "model_limit",
        "memory_prior_max_cells": 8_500_000,
        "memory_prior_version": 1,
    }
    assert not _is_current_memory_failure_record(prior_skip, advisory_limit)
    assert not _is_current_memory_failure_record(prior_skip, {"max_cells": 8_500_000})


def test_wrapped_fit_oom_reads_the_current_attempt_log(tmp_path):
    previous = tmp_path / "previous.log"
    current = tmp_path / "current.tmp"
    previous.write_text("ordinary prior failure\n", encoding="utf-8")
    current.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8")
    error = RuntimeError("No models were trained successfully during fit()")

    assert _is_stage_oom(error, "fit", str(current))
    assert not _is_stage_oom(error, "fit", str(previous))


def test_ground_truth_mismatch_does_not_modify_saved_truth(tmp_path):
    path = tmp_path / "truth.csv"
    pd.DataFrame({"target": [0, 1]}, index=[10, 11]).to_csv(path)
    original = path.read_bytes()
    changed = pd.DataFrame({"feature": [1, 2], "target": [1, 1]}, index=[10, 11])

    with pytest.raises(AssertionError):
        _assert_ground_truth_compatible(str(path), changed)

    assert path.read_bytes() == original


def test_ground_truth_numeric_string_labels_survive_csv_inference(tmp_path):
    path = tmp_path / "truth.csv"
    data_test = pd.DataFrame({"feature": [1, 2], "target": ["2", "3"]}, index=[10, 11])
    data_test[["target"]].to_csv(path)

    _assert_ground_truth_compatible(str(path), data_test)


def test_pass_resume_rejects_corrupt_prediction_artifacts(tmp_path):
    data_test = pd.DataFrame({"feature": [1, 2], "target": [0, 1]}, index=[10, 11])
    prediction = tmp_path / "prediction.csv"
    probability = tmp_path / "probability.csv"
    pd.DataFrame({"target": [0, 1]}, index=[10, 11]).to_csv(prediction)
    pd.DataFrame({"0": [0.9, 0.1], "1": [0.1, 0.9]}, index=[10, 99]).to_csv(probability)

    assert not _saved_predictions_are_valid(
        str(prediction), str(probability), data_test, TaskType.Classification
    )


def test_stale_metric_for_failed_unit_is_removed_without_touching_prediction(tmp_path):
    output = tmp_path / "results"
    predictions = output / "seed_0" / "predictions"
    stats = output / "seed_0" / "stats"
    predictions.mkdir(parents=True)
    stats.mkdir(parents=True)
    key = "toy__target"
    model = "MODEL"
    prediction = predictions / f"{key}_{model}_predictions.csv"
    truth = predictions / f"{key}_ground_truth.csv"
    record = stats / f"{key}_{model}.json"
    pd.DataFrame({"target": [0]}, index=[1]).to_csv(prediction)
    pd.DataFrame({"target": [0]}, index=[1]).to_csv(truth)
    atomic_write_json(record, {"status": "fail"})
    metric_path = output / "metrics.csv"
    frame = pd.DataFrame([{"seed": 0, "key": key, "model": model, "f1_macro": 1.0}])
    frame.to_csv(metric_path, index=False)
    os.utime(metric_path, (metric_path.stat().st_atime, metric_path.stat().st_mtime + 1))
    original_prediction = prediction.read_bytes()

    filtered, done, dirty = _filter_current_metric_rows(
        frame,
        str(metric_path),
        str(output),
        {(0, key, model): "fail"},
        requires_proba=False,
    )

    assert filtered.empty
    assert not done
    assert dirty
    assert prediction.read_bytes() == original_prediction


def test_existing_grid_config_is_immutable(tmp_path):
    kwargs = {
        "datasets": ["toy"],
        "datasets_regression": [],
        "models": [{"key": "DUMMY", "device": "cpu", "solo": False}],
        "n_rep": None,
        "cv_folds": 2,
        "time_limit": 60,
        "out_root": str(tmp_path),
        "cache_dir": ".cache",
        "test_size": 0.2,
        "random_state": 42,
        "min_samples_per_class": 2,
    }
    first = _write_cell_config(2000, 100, **kwargs)
    config_path = Path(first["full_cfg"])
    original = config_path.read_bytes()

    _write_cell_config(2000, 100, **kwargs)
    assert config_path.read_bytes() == original

    resumed = _write_cell_config(2000, 100, **{**kwargs, "time_limit": 120})
    assert resumed["n_splits"] == 2
    assert config_path.read_bytes() == original


def test_expanded_result_fork_copies_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source_cell = source / "cap_2000_n100"
    destination_cell = destination / "cap_2000_n100"
    source_cell.mkdir(parents=True)
    destination_cell.mkdir(parents=True)
    shared = {
        "models": ["MODEL"],
        "bio_max_features": 2000,
        "train_subsample": 100,
        "cv_folds": 2,
    }
    atomic_write_json(
        source_cell / "config.json",
        {
            **shared,
            "output_dir": str(source_cell),
            "datasets_classification": ["old"],
            "datasets_regression": [],
            "model_limits": {"MODEL": {"max_cells": 100}},
        },
    )
    atomic_write_json(
        destination_cell / "config.json",
        {
            **shared,
            "models": ["MODEL", "NEW_MODEL"],
            "output_dir": str(destination_cell),
            "datasets_classification": ["old", "new"],
            "datasets_regression": ["new_regression"],
            "model_limits": {
                "MODEL": {
                    "max_cells": 90,
                    "memory_prior_version": 2,
                    "enforce_memory_prior": True,
                }
            },
        },
    )
    source_prediction = source_cell / "seed_0" / "predictions" / "old_MODEL_predictions.csv"
    source_prediction.parent.mkdir(parents=True)
    source_prediction.write_text("target\n1\n", encoding="utf-8")

    marker = fork_grid_results(source, destination)
    destination_prediction = (
        destination_cell / "seed_0" / "predictions" / "old_MODEL_predictions.csv"
    )
    assert marker["files"]["copied"] == 1
    assert destination_prediction.read_bytes() == source_prediction.read_bytes()
    assert not os.path.samefile(source_prediction, destination_prediction)

    destination_prediction.write_text("target\n0\n", encoding="utf-8")
    assert source_prediction.read_text(encoding="utf-8") == "target\n1\n"
    assert fork_grid_results(source, destination) == marker
