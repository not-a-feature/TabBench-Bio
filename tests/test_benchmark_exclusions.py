import sqlite3

import pandas as pd
import pytest

from tabbench_bio.dataset import TaskType
from tabbench_bio.exclusions import excluded_dataset_keys, materialize_exclusions
from tabbench_bio.leaderboard import _metrics_from_sqlite
from tabbench_bio.result_store import ResultRepository
from tabbench_bio.sample_fallback import _load_status_records


@pytest.mark.parametrize(
    "cell", ["cap_25000_n100", "cap_10000_n100", "cap_full_n500", "cap_full_n1000"]
)
def test_other_grid_cells_remain_in_scope(cell):
    assert excluded_dataset_keys(cell) == set()


@pytest.mark.parametrize(
    "cell", ["cap_full_n20", "cap_full_n50", "cap_full_n100", "cap_full_n200", "cap_full"]
)
def test_binding_full_width_cells_are_excluded(cell):
    assert excluded_dataset_keys(cell) == {"gp-maize-FT_0", "gp-switchgrass-HT_0"}


def test_exclusions_override_old_passes_and_failures_without_deleting_history(tmp_path):
    cell = tmp_path / "experiment" / "cap_full_n100"
    repository = ResultRepository(cell, {"models": ["DUMMY", "RF"], "train_subsample": 100})
    keys = ["gp-maize-FT_0", "gp-switchgrass-HT_0", "other_0"]
    truth = pd.DataFrame({"target": [1.0, 2.0]}, index=[10, 11])
    repository.write(
        {"dataset": keys[0], "model": "RF", "status": "pass", "reason": ""},
        seed=0,
        prediction=truth,
        ground_truth=truth,
    )
    repository.write(
        {"dataset": keys[1], "model": "DUMMY", "status": "fail", "reason": "time_limit"},
        seed=0,
    )
    excluded = materialize_exclusions(repository, [0, 1], ["DUMMY", "RF"], keys)
    assert excluded == set(keys[:2])
    assert len(repository.attempts()) == 10
    assert len(repository.current_attempts()) == 8
    assert all(
        a.status == "skip" and a.reason == "benchmark_exclusion"
        for a in repository.current_attempts()
    )
    assert repository.current(0, "other_0", "RF") is None
    materialize_exclusions(repository, [0, 1], ["DUMMY", "RF"], keys)
    assert len(repository.attempts()) == 10
    reader = ResultRepository.from_root(cell.parent, cell=cell.name)
    assert all(a.status == "skip" for a in reader.current_attempts())
    with sqlite3.connect(repository.writer_path) as connection:
        reg, clf, status = _metrics_from_sqlite(connection, cell.name)
    assert reg.empty and clf.empty
    assert len(status) == 8
    assert set(status["status"]) == {"skip"}
    assert sum(a.status == "pass" for a in repository.attempts()) == 1
    assert _load_status_records(reader, [cell.name]).empty


def test_prediction_runner_excludes_before_iteration_and_fit(tmp_path, monkeypatch, debug_config):
    import tabbench_bio.predictions as predictions

    cell = tmp_path / "experiment" / "cap_full_n100"
    debug_config["output_dir"] = str(cell)
    debug_config["models"] = ["DUMMY", "RF"]

    class Benchmark:
        _key_list = ["gp-maize-FT_0", "gp-switchgrass-HT_0"]
        _task_type_list = [TaskType.Regression, TaskType.Regression]

        def __len__(self):
            return len(self._key_list)

        def __iter__(self):
            assert self._key_list == []
            assert self._task_type_list == []
            return iter(())

    def forbidden_model(**kwargs):
        pytest.fail("Excluded cells must not instantiate a model")

    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "AutoGluonModel", forbidden_model)
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    predictions.compute_predictions(debug_config)
    repository = ResultRepository.from_root(cell.parent, cell=cell.name)
    assert len(repository.current_attempts()) == 4
    assert all(a.reason == "benchmark_exclusion" for a in repository.current_attempts())
