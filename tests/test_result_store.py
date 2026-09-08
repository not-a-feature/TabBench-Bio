import json
import os
import shutil
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext

import pandas as pd
import pytest

from tabbench_bio.dataset import TaskType
from tabbench_bio.evaluation import _compute_metrics_from_store
from tabbench_bio.io_utils import atomic_write_json
from tabbench_bio.result_store import (
    ResultRepository,
    _connect,
    _frame_bytes,
    _sha256,
    consolidate_results,
    import_legacy_results,
    install_snapshots,
    snapshot_database,
    snapshot_writers,
)
from tabbench_bio.split_manifest import split_versions, unit_id


def test_reader_uses_sqlite_read_only_mode(tmp_path):
    path = tmp_path / "results # shared.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE marker (value INTEGER)")
    with closing(_connect(path, read_only=True)) as connection:
        connection.execute("PRAGMA query_only=OFF")
        assert connection.execute("SELECT count(*) FROM marker").fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO marker VALUES (1)")


@pytest.mark.skipif(os.name != "posix", reason="POSIX group permissions")
def test_published_databases_remain_group_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    root = tmp_path / "experiment"
    cell = root / "cap_2000_n20"
    repository = ResultRepository(cell, _config(cell))
    repository.writer_path.chmod(0o600)
    canonical = consolidate_results(root)
    manifest = snapshot_writers(root)
    for path in (canonical, manifest, manifest.parent / repository.writer_path.name):
        assert path.stat().st_mode & stat.S_IRGRP, path


def _config(output_dir):
    return {
        "output_dir": str(output_dir),
        "cache_dir": ".cache",
        "random_state": 42,
        "models": ["LR"],
    }


def _pass_record(dataset="dataset", model="LR"):
    return {
        "dataset": dataset,
        "model": model,
        "status": "pass",
        "reason": "",
        "error": "",
        "timestamp": "2026-08-31T12:00:00Z",
        "train_time_s": 1.25,
    }


def test_pass_roundtrip_and_blob_deduplication(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    cell = tmp_path / "experiment" / "cap_2000_n20"
    config = _config(cell)
    repository = ResultRepository(cell, config)
    prediction = pd.DataFrame({"target": [0, 1]}, index=[10, 11])
    probability = pd.DataFrame({"0": [0.8, 0.1], "1": [0.2, 0.9]}, index=[10, 11])
    truth = pd.DataFrame({"target": [0, 1]}, index=[10, 11])

    first = repository.write(
        _pass_record(),
        seed=42,
        prediction=prediction,
        probability=probability,
        ground_truth=truth,
        log_text="fit complete\n",
    )
    second = repository.write(
        _pass_record(dataset="second"),
        seed=42,
        prediction=prediction,
        probability=probability,
        ground_truth=truth,
    )

    pd.testing.assert_frame_equal(repository.dataframe(first, "prediction"), prediction)
    pd.testing.assert_frame_equal(repository.dataframe(first, "probability"), probability)
    pd.testing.assert_frame_equal(repository.dataframe(first, "ground_truth"), truth)
    assert repository.log_text(first) == "fit complete\n"
    assert first.ground_truth_sha256 == second.ground_truth_sha256
    with sqlite3.connect(repository.writer_path) as connection:
        assert connection.execute("SELECT count(*) FROM attempts").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM blobs").fetchone()[0] == 3


def test_named_prediction_series_is_stored_as_target_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    cell = tmp_path / "experiment" / "cap_2000_n20"
    repository = ResultRepository(cell, _config(cell))
    prediction = pd.Series([0, 1], index=[10, 11], name="target")
    truth = prediction.to_frame()

    attempt = repository.write(
        _pass_record(),
        seed=42,
        prediction=prediction,
        probability=pd.DataFrame({"0": [0.8, 0.1], "1": [0.2, 0.9]}, index=[10, 11]),
        ground_truth=truth,
    )

    pd.testing.assert_frame_equal(repository.dataframe(attempt, "prediction"), truth)


def test_failed_attempt_is_retained_when_retry_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    cell = tmp_path / "experiment" / "cap_2000_n20"
    config = _config(cell)
    repository = ResultRepository(cell, config)
    failure = {
        "dataset": "dataset",
        "model": "LR",
        "status": "fail",
        "reason": "fit_error",
        "error": "failed",
        "timestamp": "2026-08-31T12:00:00Z",
    }
    repository.write(failure, seed=42)
    repository.write(failure, seed=42)
    prediction = pd.DataFrame({"target": [1]}, index=[10])
    truth = pd.DataFrame({"target": [1]}, index=[10])
    repository.write(
        {**_pass_record(), "timestamp": "2026-08-31T12:01:00Z"},
        seed=42,
        prediction=prediction,
        ground_truth=truth,
    )

    current = repository.current(42, "dataset", "LR")
    assert current.status == "pass"
    assert [attempt.status for attempt in repository.attempts()] == ["fail", "pass"]


def test_divergent_passing_results_are_rejected(tmp_path, monkeypatch):
    cell = tmp_path / "experiment" / "cap_2000_n20"
    config = _config(cell)
    truth = pd.DataFrame({"target": [0]}, index=[10])
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    first = ResultRepository(cell, config)
    first.write(
        _pass_record(),
        seed=42,
        prediction=pd.DataFrame({"target": [0]}, index=[10]),
        ground_truth=truth,
    )

    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-b")
    second = ResultRepository(cell, config)
    with pytest.raises(AssertionError, match="Divergent passing results"):
        second.write(
            _pass_record(),
            seed=42,
            prediction=pd.DataFrame({"target": [1]}, index=[10]),
            ground_truth=truth,
        )


def test_snapshot_and_consolidation_are_integral_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    root = tmp_path / "experiment"
    cell = root / "cap_2000_n20"
    config = _config(cell)
    repository = ResultRepository(cell, config)
    frame = pd.DataFrame({"target": [0]}, index=[10])
    repository.write(_pass_record(), seed=42, prediction=frame, ground_truth=frame)

    snapshot = snapshot_database(repository.writer_path, tmp_path / "snapshot.sqlite")
    with closing(sqlite3.connect(snapshot)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    canonical = consolidate_results(root)
    with closing(sqlite3.connect(canonical)) as connection:
        before = connection.execute("SELECT count(*) FROM attempts").fetchone()[0]
    consolidate_results(root)
    with closing(sqlite3.connect(canonical)) as connection:
        assert connection.execute("SELECT count(*) FROM attempts").fetchone()[0] == before
    consolidated = ResultRepository.from_root(root)
    assert len(consolidated.current_attempts()) == 1


def test_consolidation_allows_identical_content_in_different_artifact_roles(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    cell = root / "cap_2000_n20"
    config = _config(cell)
    shared = pd.DataFrame({"target": [0]}, index=[10])

    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    first = ResultRepository(cell, config)
    first.write(
        {**_pass_record(), "dataset": "dataset-a"},
        seed=42,
        prediction=pd.DataFrame({"target": [1]}, index=[10]),
        ground_truth=shared,
    )

    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-b")
    second = ResultRepository(cell, config)
    second.write(
        {**_pass_record(), "dataset": "dataset-b"},
        seed=42,
        prediction=shared,
        ground_truth=pd.DataFrame({"target": [2]}, index=[10]),
    )

    canonical = consolidate_results(root)
    with closing(sqlite3.connect(canonical)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_legacy_import_is_read_only_and_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    cell = root / "cap_2000_n20"
    seed = cell / "seed_42"
    stats = seed / "stats"
    predictions = seed / "predictions"
    logs = seed / "logs"
    stats.mkdir(parents=True)
    predictions.mkdir()
    logs.mkdir()
    config = _config(cell)
    (cell / "config.json").write_text(json.dumps(config), encoding="utf-8")
    record_path = stats / "dataset_LR.json"
    record_path.write_text(json.dumps(_pass_record()), encoding="utf-8")
    frame = pd.DataFrame({"target": [0, 1]}, index=[10, 11])
    frame.to_csv(predictions / "dataset_LR_predictions.csv")
    frame.to_csv(predictions / "dataset_ground_truth.csv")
    (logs / "dataset_LR.log").write_text("legacy log\n", encoding="utf-8")
    original = {
        path: path.read_bytes() for path in (record_path, *predictions.iterdir(), *logs.iterdir())
    }

    import_legacy_results(root)
    import_legacy_results(root)
    monkeypatch.setenv("TABBENCH_WRITER_ID", "new-node")
    repository = ResultRepository.from_root(root)

    assert len(repository.attempts()) == 1
    attempt = repository.current(42, "dataset", "LR", cell="cap_2000_n20")
    pd.testing.assert_frame_equal(repository.dataframe(attempt, "prediction"), frame)
    assert repository.log_text(attempt) == original[logs / "dataset_LR.log"].decode("utf-8")
    assert all(path.read_bytes() == content for path, content in original.items())


def test_writer_refuses_unimported_legacy_tree(tmp_path):
    cell = tmp_path / "experiment" / "cap_2000_n20"
    stats = cell / "seed_42" / "stats"
    stats.mkdir(parents=True)
    (stats / "dataset_LR.json").write_text(
        json.dumps({"dataset": "dataset", "model": "LR", "status": "fail"}),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="import-legacy"):
        ResultRepository(cell, _config(cell))


def test_metrics_are_rebuilt_from_bundled_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    cell = tmp_path / "experiment" / "cap_2000_n20"
    config = {
        **_config(cell),
        "exclude_keys": [],
        "exclude_datasets": [],
        "exclude_targets": [],
    }
    repository = ResultRepository(cell, config)
    truth = pd.DataFrame({"target": [0, 1]}, index=[10, 11])
    probability = pd.DataFrame({"0": [0.9, 0.1], "1": [0.1, 0.9]}, index=[10, 11])
    repository.write(
        _pass_record(dataset="toy_0"),
        seed=42,
        prediction=truth,
        probability=probability,
        ground_truth=truth,
    )

    class Benchmark:
        dataset_names_regression = []

        @staticmethod
        def split_key(key):
            return key.rsplit("_", 1)[0], int(key.rsplit("_", 1)[1])

    monkeypatch.setattr(
        "tabbench_bio.evaluation.configure_benchmark", lambda *_args, **_kwargs: Benchmark()
    )
    _compute_metrics_from_store(config, repository)

    metrics = pd.read_csv(cell / "metrics" / "classification_metrics.csv")
    assert metrics.loc[0, "key"] == "toy_0"
    assert metrics.loc[0, "model"] == "LR"
    assert metrics.loc[0, "f1_macro"] == pytest.approx(1.0)


def test_writer_snapshots_transfer_as_a_small_validated_bundle_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    source = tmp_path / "source" / "experiment"
    cell = source / "cap_2000_n20"
    config = _config(cell)
    repository = ResultRepository(cell, config)
    frame = pd.DataFrame({"target": [0]}, index=[10])
    repository.write(_pass_record(), seed=42, prediction=frame, ground_truth=frame)
    manifest = snapshot_writers(source)
    stale_snapshot = tmp_path / "stale-snapshot"
    shutil.copytree(manifest.parent, stale_snapshot)

    destination = tmp_path / "destination" / "experiment"
    installed = install_snapshots(destination, manifest.parent)
    imported = ResultRepository.from_root(destination)

    assert [path.name for path in installed] == ["node-a.sqlite"]
    assert len(imported.current_attempts()) == 1

    repository.write(
        {"dataset": "second", "model": "LR", "status": "fail", "reason": "fit_error"},
        seed=42,
    )
    snapshot_writers(source)
    install_snapshots(destination, manifest.parent)
    assert len(ResultRepository.from_root(destination).current_attempts()) == 2
    with pytest.raises(AssertionError, match="would discard 1 attempt"):
        install_snapshots(destination, stale_snapshot)


def test_writer_snapshot_can_exclude_frozen_legacy_bundle(tmp_path, monkeypatch):
    source = tmp_path / "source" / "experiment"
    cell = source / "cap_2000_n20"
    frame = pd.DataFrame({"target": [0]}, index=[10])
    for writer in ("legacy", "node-a"):
        monkeypatch.setenv("TABBENCH_WRITER_ID", writer)
        repository = ResultRepository(cell, _config(cell))
        repository.write(
            _pass_record(dataset=writer), seed=42, prediction=frame, ground_truth=frame
        )

    first_manifest = snapshot_writers(source)
    assert {path.name for path in first_manifest.parent.glob("*.sqlite")} == {
        "legacy.sqlite",
        "node-a.sqlite",
    }

    manifest = snapshot_writers(source, exclude_writers=("legacy",))
    assert {path.name for path in manifest.parent.glob("*.sqlite")} == {"node-a.sqlite"}


@pytest.mark.parametrize("frozen", [False, True])
def test_prediction_pipeline_commits_and_resumes_from_bundle(tmp_path, monkeypatch, frozen):
    import tabbench_bio.predictions as predictions

    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    cell = tmp_path / "experiment" / "cap_2000_n20"
    config = {
        "output_dir": str(cell),
        "cache_dir": str(tmp_path / "cache"),
        "cv_folds": 1,
        "n_repetitions": 1,
        "random_state": 42,
        "models": ["LR"],
        "autogluon_time_limit": 60,
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "num_hpo_trials": 0,
        "subsample": None,
        "model_limits": {},
        "model_overrides": {},
        "train_subsample": None,
        "nan_policy": {},
    }
    train = pd.DataFrame({"x": [0.0, 1.0], "target": [0, 1]}, index=[1, 2])
    test = pd.DataFrame({"x": [0.2, 0.8], "target": [0, 1]}, index=[10, 11])

    class Benchmark:
        _key_list = ["toy_0"]

        def __len__(self):
            return 1

        def __iter__(self):
            yield train, test, "toy_0", TaskType.Classification

    class Model:
        fits = 0

        def __init__(self, **_kwargs):
            self.predictor = None
            self.autogluon_path = ""

        def fit(self, _data):
            type(self).fits += 1

        def predict(self, data):
            return pd.DataFrame({"target": [0, 1]}, index=data.index)

        def predict_proba(self, data):
            return pd.DataFrame({"0": [0.9, 0.1], "1": [0.1, 0.9]}, index=data.index)

        @staticmethod
        def get_fit_stats():
            return {}

    monkeypatch.setattr(predictions, "configure_benchmark", lambda _config: Benchmark())
    monkeypatch.setattr(predictions, "AutoGluonModel", Model)
    monkeypatch.setattr(predictions, "_time_budget", lambda _seconds: nullcontext())
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda _seed: None)
    monkeypatch.setattr(predictions, "_cleanup_model", lambda _model: None)

    if frozen:
        atomic_write_json(
            cell.parent / "split_manifest.json",
            {
                "schema_version": 1,
                "experiment_id": "experiment",
                "versions": split_versions(),
                "units": {
                    unit_id(0, "toy_0"): {
                        "test_indices": [10, 11],
                        "train_indices": [1, 2],
                        "ground_truth_sha256": _sha256(_frame_bytes(test[["target"]])),
                    }
                },
            },
        )
    predictions.compute_predictions(config)
    predictions.compute_predictions(config)

    repository = ResultRepository.from_root(cell.parent)
    attempt = repository.current(0, "toy_0", "LR", cell=cell.name)
    assert attempt.status == "pass"
    assert Model.fits == 1
    assert not (cell / "seed_0").exists()


def test_same_host_writers_serialize_transactions(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "shared-node")
    cell = tmp_path / "experiment" / "cap_2000_n20"
    config = _config(cell)

    def write(index):
        repository = ResultRepository(cell, config)
        repository.write(
            {
                "dataset": f"dataset-{index}",
                "model": "LR",
                "status": "fail",
                "reason": "fit_error",
                "error": "test",
                "timestamp": f"2026-08-31T12:00:{index:02d}Z",
            },
            seed=42,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(16)))

    repository = ResultRepository.from_root(cell.parent)
    assert len(repository.current_attempts()) == 16


def test_tier_workers_register_the_frozen_full_cell_config(tmp_path, monkeypatch):
    cell = tmp_path / "experiment" / "cap_2000_n20"
    cell.mkdir(parents=True)
    full_config = {**_config(cell), "models": ["LR", "RF"]}
    (cell / "config.json").write_text(json.dumps(full_config), encoding="utf-8")

    monkeypatch.setenv("TABBENCH_WRITER_ID", "gpu-node")
    gpu_repository = ResultRepository(cell, {**full_config, "models": ["LR"]})
    gpu_repository.write(
        {"dataset": "toy_0", "model": "LR", "status": "skip", "reason": "model_limit"},
        seed=42,
    )

    monkeypatch.setenv("TABBENCH_WRITER_ID", "cpu-node")
    cpu_repository = ResultRepository(cell, {**full_config, "models": ["RF"]})
    cpu_repository.write(
        {"dataset": "toy_0", "model": "RF", "status": "skip", "reason": "model_limit"},
        seed=42,
    )

    canonical = consolidate_results(cell.parent)
    with closing(sqlite3.connect(canonical)) as connection:
        stored = json.loads(
            connection.execute(
                "SELECT config_json FROM cells WHERE cell = ?", (cell.name,)
            ).fetchone()[0]
        )
    assert stored["models"] == ["LR", "RF"]
