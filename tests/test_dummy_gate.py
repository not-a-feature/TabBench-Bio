import sqlite3
from contextlib import nullcontext

import pandas as pd
import pytest

from scripts.apply_dummy_gate import apply_dummy_gate
from tabbench_bio import predictions
from tabbench_bio.dataset import TaskType
from tabbench_bio.doctor import check_results
from tabbench_bio.exclusions import materialize_dummy_skips
from tabbench_bio.leaderboard import _metrics_from_sqlite
from tabbench_bio.result_store import ResultRepository


def test_gate_overrides_successes_preserves_history_and_is_idempotent(tmp_path):
    config = {
        "models": ["RF", "DUMMY"],
        "datasets_classification": [],
        "datasets_regression": ["toy"],
        "cv_folds": 3,
        "n_repetitions": 1,
        "random_state": 42,
    }
    repository = ResultRepository(tmp_path / "experiment/cap_10000", config)
    truth = pd.DataFrame({"target": [0.0, 1.0]})
    for seed, status in enumerate(["fail", "skip", "pass"]):
        repository.write(
            {
                "dataset": "toy_0",
                "model": "DUMMY",
                "status": status,
                "reason": "fit_error"
                if status == "fail"
                else "empty_split"
                if status == "skip"
                else "",
            },
            seed=seed,
            prediction=truth if status == "pass" else None,
            ground_truth=truth if status == "pass" else None,
        )
        repository.write(
            {
                "dataset": "toy_0",
                "model": "RF",
                "status": "pass" if seed == 0 else "fail",
                "reason": "" if seed == 0 else "fit_error",
            },
            seed=seed,
            prediction=truth if seed == 0 else None,
            ground_truth=truth if seed == 0 else None,
        )
    before = repository.writer_path.read_bytes()
    migrated = tmp_path / "migrated.sqlite"
    report = apply_dummy_gate(repository.writer_path, migrated)
    assert report["added_skips"] == 3
    assert repository.writer_path.read_bytes() == before
    assert any("DUMMY eligibility" in problem for problem in check_results(repository.writer_path))
    assert not any("DUMMY eligibility" in problem for problem in check_results(migrated))
    assert apply_dummy_gate(migrated, tmp_path / "again.sqlite")["added_skips"] == 0
    materialize_dummy_skips(repository, [0, 1, 2], ["DUMMY", "RF"], ["toy_0"])
    for seed in [0, 1]:
        assert all(
            repository.current(seed, "toy_0", model).status == "skip" for model in config["models"]
        )
    assert repository.current(2, "toy_0", "RF").status == "fail"
    assert len(repository.attempts()) == 9
    materialize_dummy_skips(repository, [0, 1, 2], config["models"], ["toy_0"])
    assert len(repository.attempts()) == 9
    with sqlite3.connect(repository.writer_path) as connection:
        reg, _, status = _metrics_from_sqlite(connection, repository.cell)
    assert set(reg["seed"]) == {2}
    assert set(status[status.seed.isin([0, 1])].status) == {"skip"}


@pytest.mark.parametrize("failure", [None, "fit_error", "time_limit"])
@pytest.mark.parametrize("shards", [1, 2])
def test_runner_executes_dummy_first_and_gates_all_peers(
    tmp_path, monkeypatch, debug_config, failure, shards
):
    cell = tmp_path / "experiment/cap_2000_n20"
    debug_config.update(
        output_dir=str(cell),
        models=["RF"],
        dataset_names_classification=[],
        dataset_names_regression=["toy"],
    )
    frame = pd.DataFrame({"feature": range(6), "target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
    calls = []

    class Benchmark:
        _key_list = [f"toy_{index}" for index in range(shards)]
        _task_type_list = [TaskType.Regression] * shards

        def __len__(self):
            return len(self._key_list)

        def __iter__(self):
            for key in self._key_list:
                yield frame.iloc[:4], frame.iloc[4:], key, TaskType.Regression

        def training_groups(self, key, data):
            return None

    class Model:
        def __init__(self, models, **kwargs):
            self.name = models[0]
            self.predictor = None
            self.autogluon_path = ""

        def fit(self, data, *, groups):
            calls.append(self.name)
            if failure == "time_limit":
                raise predictions.UnitTimeoutError("unit budget exceeded")
            if failure == "fit_error":
                raise AssertionError(
                    "Internal validation requires each class in at least two biological groups"
                )

        def predict(self, data):
            return data[["target"]]

        def get_fit_stats(self):
            return {}

    monkeypatch.setattr(predictions, "AutoGluonModel", Model)
    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "_time_budget", lambda seconds: nullcontext())
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    monkeypatch.setattr(predictions, "_cleanup_model", lambda model: None)
    for shard in range(shards):
        predictions.compute_predictions(debug_config, num_shards=shards, shard_index=shard)
    predictions.compute_predictions(debug_config)
    repository = ResultRepository.from_root(cell.parent, cell=cell.name)
    if failure is None:
        assert calls == ["DUMMY", "RF"] * shards
        assert all(a.status == "pass" for a in repository.current_attempts())
    else:
        assert calls == ["DUMMY"] * shards
        assert all(a.status == "skip" for a in repository.current_attempts())
        assert all(a.reason == "dummy_unavailable" for a in repository.current_attempts())
        assert len(repository.attempts()) == 3 * shards
        assert repository.attempts()[0].status == "fail"
