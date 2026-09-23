from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.dataset import TaskType


def test_gene_expression_datasets_precede_the_remaining_schedule(tmp_path, monkeypatch):
    benchmark = TabBenchBio(
        dataset_names_classification=["OpenML-1458", "OpenML-1138"],
        dataset_names_regression=["OpenML-430", "OpenML-46983"],
        cache_dir=str(tmp_path),
    )
    benchmark._index = {
        "OpenML-1458": 1,
        "OpenML-1138": 1,
        "OpenML-430": 1,
        "OpenML-46983": 1,
    }
    monkeypatch.setattr(benchmark, "_load_datasets", lambda _names: None)
    monkeypatch.setattr(benchmark, "_save_index", lambda: None)

    benchmark.init_datasets()

    assert benchmark._key_list == [
        "OpenML-1138_0",
        "OpenML-46983_0",
        "OpenML-1458_0",
        "OpenML-430_0",
    ]
    assert benchmark._task_type_list == [
        TaskType.Classification,
        TaskType.Regression,
        TaskType.Classification,
        TaskType.Regression,
    ]
