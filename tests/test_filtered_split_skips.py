import pandas as pd
import pytest

from tabbench_bio import predictions
from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.dataset import TaskType
from tabbench_bio.result_store import ResultRepository


@pytest.mark.parametrize("missing", ["train", "test", "both"])
def test_missing_filtered_split_is_terminal_for_each_model_and_seed(
    tmp_path, monkeypatch, debug_config, missing
):
    cell = tmp_path / "experiment" / "cap_2000_n20"
    debug_config.update(output_dir=str(cell), models=["DUMMY", "RF"])
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

    class Benchmark:
        _key_list = ["OpenML-1106_0"]
        _task_type_list = [TaskType.Classification]

        def __len__(self):
            return 1

        def __iter__(self):
            yield (
                None if missing in {"train", "both"} else frame,
                None if missing in {"test", "both"} else frame,
                self._key_list[0],
                self._task_type_list[0],
            )

    def forbidden_model(**kwargs):
        pytest.fail("A filtered split must never instantiate a model")

    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "get_seeds", lambda config: [0, 1])
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    monkeypatch.setattr(predictions, "AutoGluonModel", forbidden_model)
    # Each shard must write only its own units; restarting preserves terminal coverage.
    predictions.compute_predictions(debug_config, num_shards=2, shard_index=0)
    reader = ResultRepository.from_root(cell.parent, cell=cell.name)
    assert {(a.seed, a.model) for a in reader.current_attempts()} == {(0, "DUMMY"), (1, "DUMMY")}
    predictions.compute_predictions(debug_config, num_shards=2, shard_index=1)
    predictions.compute_predictions(debug_config)
    reader = ResultRepository.from_root(cell.parent, cell=cell.name)
    assert len(reader.current_attempts()) == 4
    for attempt in reader.current_attempts():
        assert (attempt.status, attempt.reason) == ("skip", "empty_split_after_filtering")
        assert attempt.record["n_train_samples"] == (None if missing in {"train", "both"} else 2)
        assert attempt.record["n_test_samples"] == (None if missing in {"test", "both"} else 2)


@pytest.mark.parametrize("minimum", [10, 2])
def test_cached_split_preserves_the_previously_defined_class_space(tmp_path, minimum):
    benchmark = TabBenchBio.__new__(TabBenchBio)
    benchmark.cache_dir_processed = str(tmp_path)
    benchmark.dataset_names_classification = ["OpenML-1106"]
    benchmark.min_samples_per_class = minimum
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    for path in benchmark._get_cache_paths("OpenML-1106_0"):
        frame.to_pickle(path)
    train, test = benchmark._load_dataset_from_cache("OpenML-1106_0")
    pd.testing.assert_frame_equal(train, frame)
    pd.testing.assert_frame_equal(test, frame)


def test_budgeted_cache_keeps_eligible_minority_class_and_test_rows(tmp_path):
    benchmark = TabBenchBio.__new__(TabBenchBio)
    benchmark.cache_dir_processed = str(tmp_path)
    benchmark.dataset_names_classification = ["toy"]
    benchmark.min_samples_per_class = 10
    full = pd.DataFrame({"target": ["A"] * 80 + ["B"] * 60 + ["C"] * 10})
    assert benchmark._filter_rare_classes(full, "toy_0")[1] == []
    train = pd.DataFrame({"target": ["A"] * 11 + ["B"] * 7 + ["C"] * 2})
    test = pd.DataFrame({"target": ["A"] * 16 + ["B"] * 12 + ["C"] * 2}, index=range(120, 150))
    benchmark._save_dataset("toy_0", train, test)
    actual_train, actual_test = benchmark._load_dataset_from_cache("toy_0")
    pd.testing.assert_frame_equal(actual_train, train)
    pd.testing.assert_frame_equal(actual_test, test)


def test_loader_errors_are_not_converted_to_skips(tmp_path, monkeypatch, debug_config):
    cell = tmp_path / "experiment" / "cap_2000_n20"
    debug_config["output_dir"] = str(cell)

    def broken_loader(config):
        raise OSError("dataset unavailable")

    monkeypatch.setattr(predictions, "configure_benchmark", broken_loader)
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    with pytest.raises(OSError, match="dataset unavailable"):
        predictions.compute_predictions(debug_config)
    assert ResultRepository.from_root(cell.parent, cell=cell.name).current_attempts() == []
