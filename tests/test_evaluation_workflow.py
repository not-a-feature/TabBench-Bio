import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from tabbench_bio import benchmark, config, evaluate, leaderboard
from tabbench_bio.cli import main as cli_main
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


@pytest.mark.parametrize("datasets", [["OpenML-1083"], ["OpenML-1083", "OpenML-1088"], []])
def test_evaluation_writes_reports_and_matches_published_folds(tmp_path, monkeypatch, datasets):
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
            for name in ("DUMMY", "RF"):
                reference.write(
                    {"dataset": dataset + "_0", "model": name, "status": "pass", "reason": ""},
                    seed=seed,
                    ground_truth=truth,
                    prediction=truth,
                    probability=pd.DataFrame(
                        {0: 1 - truth.target, 1: truth.target}, index=truth.index
                    ),
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
            str(database),
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
    selected = datasets or settings["datasets_classification"]
    assert len(custom) == 2 * len(selected)
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
    assert (output / "split_manifest.json").is_file()
    assert json.loads((output / "baseline.json").read_text()) == {
        "path": str(database.resolve()),
        "sha256": checksum,
    }
    assert hashlib.sha256(database.read_bytes()).hexdigest() == checksum


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("outcome", ["fail", "skip", "missing"])
@pytest.mark.parametrize("peer_outcome", ["pass", "skip"])
@pytest.mark.parametrize("baseline_mode", ["path", "latest", "default"])
def test_published_dummy_outcome_gates_custom_fits(
    tmp_path, monkeypatch, task, outcome, peer_outcome, baseline_mode
):
    name = "OpenML-1083" if task == "classification" else "CHEMBL206"
    cell = "cap_10000_n100"
    settings = config.config_for_cell(
        10000,
        100,
        datasets=[name] if task == "classification" else [],
        datasets_regression=[name] if task == "regression" else [],
        models=["DUMMY", "RF"],
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
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    reference = ResultRepository(tmp_path / "reference" / cell, settings)
    for seed in (0, 1):
        truth = frame.iloc[seed * 4 : seed * 4 + 4][["target"]]
        for model in ("DUMMY", "RF"):
            if model == "DUMMY" and seed == 1 and outcome == "missing":
                continue
            state = outcome if model == "DUMMY" and seed == 1 else "pass"
            if model == "RF" and seed == 1:
                state = peer_outcome
            passed = state == "pass"
            reference.write(
                {
                    "dataset": name + "_0",
                    "model": model,
                    "status": state,
                    "reason": "" if passed else "fit_error" if state == "fail" else "empty_split",
                },
                seed=seed,
                prediction=truth if passed else None,
                ground_truth=truth if passed else None,
                probability=(
                    pd.DataFrame({0: 1 - truth.target, 1: truth.target}, index=truth.index)
                    if passed and task == "classification"
                    else None
                ),
            )
    # An ineligible fold has no prediction artifacts in the selected cell. Other
    # cells retain the shared frozen identities needed to reconstruct its universe.
    other_cell = ResultRepository(tmp_path / "reference/cap_2000_n100", settings)
    for seed in (0, 1):
        truth = frame.iloc[seed * 4 : seed * 4 + 4][["target"]]
        other_cell.write(
            {"dataset": name + "_0", "model": "RF", "status": "pass", "reason": ""},
            seed=seed,
            prediction=truth,
            ground_truth=truth,
            probability=(
                pd.DataFrame({0: 1 - truth.target, 1: truth.target}, index=truth.index)
                if task == "classification"
                else None
            ),
        )
    database = consolidate_results(tmp_path / "reference")
    before = database.read_bytes()
    fits = []
    parent = DummyClassifier if task == "classification" else DummyRegressor

    class Model(parent):
        def fit(self, X, y):
            fits.append(X[:, 0].tolist())
            return super().fit(X, y)

    def splits(settings):
        seed = settings["random_state"]
        test = frame.iloc[seed * 4 : seed * 4 + 4]
        yield (
            frame.drop(test.index),
            test,
            name + "_0",
            (TaskType.Classification if task == "classification" else TaskType.Regression),
        )

    monkeypatch.setattr(benchmark, "configure_benchmark", splits)
    monkeypatch.setattr(evaluate, "load_model_factory", lambda _: lambda task: Model())
    checks = []

    def latest(cache_dir):
        checks.append(cache_dir)
        return database, {
            "tag": "v-test",
            "path": str(database),
            "sha256": hashlib.sha256(before).hexdigest(),
        }

    monkeypatch.setattr(evaluate, "latest_reference", latest)
    reports = []
    monkeypatch.setattr(
        evaluate, "generate_report", lambda args, lb, cfg, status: reports.append(lb)
    )
    output = tmp_path / "evaluation"
    arguments = [
        "--task",
        task,
        "--dataset",
        name,
        "--output",
        str(output),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if baseline_mode != "default":
        arguments += ["--baseline", str(database) if baseline_mode == "path" else "latest"]
    if outcome == "missing":
        with pytest.raises(AssertionError, match="Reference is missing DUMMY outcomes"):
            main(arguments)
        assert fits == []
        assert checks == ([] if baseline_mode == "path" else [(tmp_path / "cache").resolve()])
        assert database.read_bytes() == before
        return
    main(arguments)
    assert checks == ([] if baseline_mode == "path" else [(tmp_path / "cache").resolve()])
    assert fits == [[4, 5, 6, 7]]
    metrics = pd.read_csv(output / "fold_metrics.csv")
    assert metrics.seed.tolist() == [0]
    status = pd.read_csv(output / "attempts.csv", keep_default_na=False)
    custom = status.loc[status.model == "My model"].set_index("seed")
    assert custom.loc[0, "status"] == "pass"
    assert custom.loc[1, ["status", "reason"]].tolist() == ["skip", "dummy_unavailable"]
    scores = reports[0]._clf_metrics if task == "classification" else reports[0]._reg_metrics
    assert set(scores.model) == {"DUMMY", "RF", "My model"}
    assert set(scores.seed) == {0}
    assert database.read_bytes() == before


@pytest.mark.parametrize("baseline", ["none", "missing.sqlite"])
@pytest.mark.parametrize("entrypoint", ["helper", "cli"])
def test_evaluation_requires_reference_file_before_loading_model(
    tmp_path, monkeypatch, baseline, entrypoint
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        evaluate, "load_model_factory", lambda _: pytest.fail("No model import without a reference")
    )
    output = tmp_path / "evaluation"
    arguments = ["--output", str(output)]
    arguments += ["--baseline", baseline]
    with pytest.raises(AssertionError, match="Reference SQLite file not found"):
        if entrypoint == "helper":
            main(arguments)
        else:
            monkeypatch.setattr("sys.argv", ["tabbench-bio", "evaluate", *arguments])
            cli_main()
    assert not output.exists()
