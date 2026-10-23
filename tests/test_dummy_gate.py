import sqlite3
from contextlib import nullcontext

import pandas as pd
import pytest

from tabbench_bio import predictions
from tabbench_bio.dataset import TaskType
from tabbench_bio.doctor import check_results
from tabbench_bio.exclusions import materialize_dummy_skips
from tabbench_bio.result_store import ResultRepository, consolidate_results


@pytest.mark.parametrize("failure", [None, "fit_error", "time_limit"])
@pytest.mark.parametrize("shards", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("worker_order", ["forward", "backward"])
def test_runner_executes_dummy_first_and_gates_all_peers(
    tmp_path, monkeypatch, debug_config, failure, shards, reverse, worker_order
):
    cell = tmp_path / "experiment/cap_2000_n20"
    debug_config.update(
        output_dir=str(cell),
        models=["RF"],
        dataset_names_classification=[],
        dataset_names_regression=["toy"],
        datasets_classification=[],
        datasets_regression=["toy"],
        cv_folds=2,
    )
    frame = pd.DataFrame({"feature": range(6), "target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
    calls = []

    class Benchmark:
        _key_list = [f"toy_{index}" for index in range(2 * shards)]
        _task_type_list = [TaskType.Classification] * shards + [TaskType.Regression] * shards

        def __len__(self):
            return len(self._key_list)

        def __iter__(self):
            for key, task in zip(self._key_list, self._task_type_list, strict=True):
                yield frame.iloc[:4], frame.iloc[4:], key, task

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

        def predict_proba(self, data):
            return pd.DataFrame({0: 1 - data.target, 1: data.target}, index=data.index)

        def get_fit_stats(self):
            return {}

    monkeypatch.setattr(predictions, "AutoGluonModel", Model)
    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "_time_budget", lambda seconds: nullcontext())
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    monkeypatch.setattr(predictions, "_cleanup_model", lambda model: None)
    workers = range(shards) if worker_order == "forward" else reversed(range(shards))
    for shard in workers:
        with monkeypatch.context() as worker:
            worker.setenv("TABBENCH_WRITER_ID", f"worker-{shard}")
            predictions.compute_predictions(
                debug_config, num_shards=shards, shard_index=shard, reverse=reverse
            )
    repository = ResultRepository.from_root(cell.parent, cell=cell.name)
    if failure is None:
        assert calls == ["DUMMY", "RF"] * (4 * shards)
        assert all(a.status == "pass" for a in repository.current_attempts())
        for attempt in repository.current_attempts():
            expected = (
                "classification" if int(attempt.dataset.split("_")[1]) < shards else "regression"
            )
            assert attempt.record["task_type"] == expected
    else:
        assert calls == ["DUMMY"] * (4 * shards)
        assert all(
            a.status == ("fail" if a.model == "DUMMY" else "skip")
            for a in repository.current_attempts()
        )
        assert all(
            a.reason == "dummy_unavailable"
            for a in repository.current_attempts()
            if a.model != "DUMMY"
        )
    assert len(repository.attempts()) == 8 * shards
    for bundle in repository.bundle_paths():
        shard = int(bundle.stem.removeprefix("worker-"))
        with sqlite3.connect(bundle) as connection:
            assert set(connection.execute("SELECT DISTINCT dataset FROM attempts")) == {
                (f"toy_{shard}",),
                (f"toy_{shard + shards}",),
            }
    assert check_results(consolidate_results(cell.parent)) == []
    before = [a.attempt_id for a in repository.attempts()]
    for shard in reversed(range(shards)):
        with monkeypatch.context() as worker:
            worker.setenv("TABBENCH_WRITER_ID", f"worker-{shard}")
            predictions.compute_predictions(
                debug_config, num_shards=shards, shard_index=shard, reverse=not reverse
            )
    repository.reload()
    assert [a.attempt_id for a in repository.attempts()] == before


@pytest.mark.parametrize("status", ["pass", "fail"])
def test_dummy_gate_rejects_preexisting_peer_attempts(tmp_path, status):
    repository = ResultRepository(tmp_path, {"models": ["DUMMY", "RF"]})
    repository.write(
        {"dataset": "toy_0", "model": "DUMMY", "status": "fail", "reason": "fit_error"}, seed=0
    )
    truth = pd.DataFrame({"target": [0.0, 1.0]})
    repository.write(
        {"dataset": "toy_0", "model": "RF", "status": status, "reason": "fit_error"},
        seed=0,
        prediction=truth if status == "pass" else None,
        ground_truth=truth if status == "pass" else None,
    )
    before = repository.writer_path.read_bytes()
    with pytest.raises(AssertionError, match="Model attempted without a passing DUMMY"):
        materialize_dummy_skips(repository, [0], ["DUMMY", "RF"], ["toy_0"])
    assert repository.writer_path.read_bytes() == before
