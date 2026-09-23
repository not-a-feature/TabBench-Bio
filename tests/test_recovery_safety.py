import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.feature_sweep import _write_cell_config
from tabbench_bio.io_utils import atomic_to_csv, atomic_write_json
from tabbench_bio.logging_utils import run_file_logger
from tabbench_bio.predictions import (
    MEMORY_RETRY_VERSION,
    _assert_ground_truth_frame_compatible,
    _is_current_memory_failure_record,
    _is_stage_oom,
)


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
        _assert_ground_truth_frame_compatible(pd.read_csv(path, index_col=0), changed)

    assert path.read_bytes() == original


def test_ground_truth_numeric_string_labels_survive_csv_inference(tmp_path):
    path = tmp_path / "truth.csv"
    data_test = pd.DataFrame({"feature": [1, 2], "target": ["2", "3"]}, index=[10, 11])
    data_test[["target"]].to_csv(path)

    _assert_ground_truth_frame_compatible(pd.read_csv(path, index_col=0), data_test)


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
