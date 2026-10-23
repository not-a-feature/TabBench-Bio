import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from tabbench_bio import benchmark, config, leaderboard
from tabbench_bio.dataset import TaskType
from tabbench_bio.evaluate import main
from tabbench_bio.result_store import ResultRepository, consolidate_results


def test_overall_evaluation_uses_the_estimator_for_each_task(monkeypatch):
    settings = {
        "cv_folds": 5,
        "n_repetitions": 1,
        "random_state": 42,
        "dataset_names_classification": ["clf"],
        "dataset_names_regression": ["reg"],
    }
    clf = pd.DataFrame({"x": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    reg = pd.DataFrame({"x": [0, 1, 2, 3], "target": [0.1, 0.4, 0.8, 1.2]})
    monkeypatch.setattr(config, "load_config", lambda _: settings.copy())
    monkeypatch.setattr(
        benchmark,
        "configure_benchmark",
        lambda _: [
            (clf, clf, "clf_0", TaskType.Classification),
            (reg, reg, "reg_0", TaskType.Regression),
        ],
    )
    lb = leaderboard.Leaderboard(pd.DataFrame(), pd.DataFrame())
    result = lb.evaluate_and_add(
        "SVM",
        {
            "classification": DummyClassifier(),
            "regression": DummyRegressor(),
        },
        "unused.json",
    )
    assert len(result) == 10
    assert set(lb._clf_metrics.model) == {"SVM"}
    assert set(lb._reg_metrics.model) == {"SVM"}
    assert np.isfinite(result.loc[result.key == "clf_0", "f1_macro"]).all()
    assert np.isfinite(result.loc[result.key == "reg_0", "rmse"]).all()


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("datasets", [["OpenML-1083"], ["OpenML-1083", "OpenML-1088"], []])
def test_evaluation_writes_reports_and_matches_published_folds(
    tmp_path, monkeypatch, published, datasets
):
    frames = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    cell = "cap_10000_n100"
    settings = config.config_for_cell(
        10000,
        100,
        datasets=["OpenML-1083", "OpenML-1088", "OpenML-1138"],
        datasets_regression=[],
        models=["RF"],
        limits={},
        overrides={},
        n_rep=1,
        cv_folds=2,
        time_limit=60,
        out_dir=str(tmp_path / "reference" / cell),
        cache_dir=str(tmp_path / "cache"),
        test_size=0.2,
        random_state=42,
        min_samples_per_class=2,
    )
    reference = ResultRepository(tmp_path / "reference" / cell, settings)
    for dataset in settings["datasets_classification"]:
        for seed in (0, 1):
            truth = frames.iloc[seed * 4 : seed * 4 + 4][["target"]]
            reference.write(
                {"dataset": dataset + "_0", "model": "RF", "status": "pass", "reason": ""},
                seed=seed,
                ground_truth=truth,
                prediction=truth,
                probability=pd.DataFrame({0: 1 - truth.target, 1: truth.target}, index=truth.index),
            )
    database = consolidate_results(tmp_path / "reference")
    checksum = hashlib.sha256(database.read_bytes()).hexdigest()

    def splits(settings):
        seed = settings["random_state"] % 2
        test = frames.iloc[seed * 4 : seed * 4 + 4]
        for dataset in settings["dataset_names_classification"]:
            yield frames.drop(test.index), test, dataset + "_0", TaskType.Classification

    monkeypatch.setattr(benchmark, "configure_benchmark", splits)
    model = tmp_path / "custom.py"
    model.write_text(
        "from sklearn.dummy import DummyClassifier\n"
        "def create_model(task):\n"
        "    return DummyClassifier(strategy='most_frequent')\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"
    main(
        [
            "--model",
            str(model),
            "--name",
            "Custom <classifier>",
            "--baseline",
            str(database) if published else "none",
            *[value for dataset in datasets for value in ("--dataset", dataset)],
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--threads",
            "1",
        ]
    )
    metrics = pd.read_csv(output / "fold_metrics.csv")
    custom = metrics[metrics.model == "Custom <classifier>"]
    selected = datasets or (
        settings["datasets_classification"] if published else ["OpenML-1083", "OpenML-1088"]
    )
    assert len(custom) == (2 if published else 5) * len(selected)
    expected_keys = {dataset + "_0" for dataset in selected}
    assert set(custom.key) == expected_keys
    exported = pd.read_csv(output / "metrics" / "classification_metrics.csv")
    assert set(exported.key) == expected_keys
    assert set(exported[exported.model == "RF"].key) == expected_keys
    ranking = pd.read_csv(output / "report" / "leaderboard.csv")
    assert set(ranking["# Targets"]) == {len(selected)}
    assert np.allclose(custom.f1_macro, 1 / 3)
    assert "Custom &lt;classifier&gt;" in (output / "report" / "report.html").read_text(
        encoding="utf-8"
    )
    for name in [
        "leaderboard.csv",
        "leaderboard.png",
        "leaderboard.svg",
        "metrics/classification_metrics.csv",
    ]:
        assert (output / (name if name.startswith("metrics/") else "report/" + name)).is_file()
    if published:
        assert (output / "split_manifest.json").is_file()
    assert hashlib.sha256(database.read_bytes()).hexdigest() == checksum
