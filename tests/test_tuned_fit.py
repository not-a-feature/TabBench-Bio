"""Real AutoGluon fits for fixed-grid variants (requires the model environment)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.tabular")

from tabbench_bio import predictions
from tabbench_bio.dataset import TaskType
from tabbench_bio.models import tuned
from tabbench_bio.models.tuned import TunedModel
from tabbench_bio.result_store import ResultRepository
from tabbench_bio.tuning import selection_split, tuning_specs

SPEC = tuning_specs(
    json.loads((Path(__file__).parents[1] / "configs/models/rf_tuned.json").read_text())
)["RF-TUNED"]


def make_model(tmp_path, task, spec):
    return TunedModel(
        models=[f"{spec['base_model']}-TUNED"],
        tuning=spec,
        nan_policy="median",
        task_type=task,
        ensemble=False,
        optimize=False,
        autogluon_time_limit=180,
        autogluon_presets="medium_quality",
        autogluon_path=str(tmp_path),
    )


@pytest.mark.parametrize("classes", [2, 3, None])
@pytest.mark.parametrize("parent,count", [("RF", 9), ("XT", 14), ("XGB", 21), ("LR", 7)])
def test_real_selection_refit_and_held_out_predictions(
    tmp_path, monkeypatch, classes, parent, count
):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "2")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    frame["target"] = np.arange(120) % classes if classes else 20 + 5 * frame["a"]
    task = TaskType.Classification if classes else TaskType.Regression
    roster = Path(__file__).parents[1] / f"configs/models/{parent.lower()}_tuned.json"
    spec = tuning_specs(json.loads(roster.read_text()))[f"{parent}-TUNED"]
    model = make_model(tmp_path, task, spec)
    groups = pd.Series(np.repeat(np.arange(30), 4), index=frame.index)
    model.fit(frame.iloc[:100], groups=groups.iloc[:100])
    assert model.time_limit == 180
    audit = model.get_fit_stats()["tuning"]
    assert audit["status"] == "completed" and audit["candidates_completed"] == count
    assert not audit["search_truncated"]
    scores = [c["validation_score"] for c in audit["candidates"]]
    assert audit["selected_candidate"] == scores.index(max(scores))
    assert set(audit["split"]["train_row_ids"]).isdisjoint(audit["split"]["validation_row_ids"])
    assert set(audit["split"]["train_row_ids"]) | set(audit["split"]["validation_row_ids"]) == set(
        map(str, range(100))
    )
    assert model.predictor.model_best in model.predictor.model_refit_map().values()
    prediction = model.predict(frame.iloc[100:].drop(columns="target"))
    assert np.isfinite(prediction).all()
    if classes:
        probabilities = model.predict_proba(frame.iloc[100:].drop(columns="target"))
        assert probabilities.shape == (20, classes)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    else:
        assert prediction.mean() > 10, "Regression predictions lost their target scale"


def test_candidate_error_remains_a_failed_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "2")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    spec = {**SPEC, "grid": [{"n_estimators": [-1]}, {}]}
    model = make_model(tmp_path, TaskType.Classification, spec)
    frame = pd.DataFrame({"a": range(60), "target": [0, 1] * 30})
    with pytest.raises(Exception):
        model.fit(frame)
    assert model.tuning_record["status"] == "failed"
    assert model.time_limit == 180
    assert model.tuning_record["candidates"][0]["status"] == "failed"
    assert model.tuning_record["selected_candidate"] is None
    assert not list(Path(model.autogluon_path).glob("candidate-*"))


@pytest.mark.parametrize("classes", [2, 3, None])
def test_mlp_selection_refit_on_cpu(tmp_path, monkeypatch, classes):
    """Exercise MLP selection and fresh refitting without a GPU dependency."""
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "2")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    roster = Path(__file__).parents[1] / "configs/models/mlp_tuned.json"
    spec = tuning_specs(json.loads(roster.read_text()))["NN_TORCH-TUNED"]
    spec = {
        **spec,
        "grid": [
            {},
            {
                "num_layers": [2],
                "hidden_size": [64],
                "learning_rate": [0.001],
                "dropout_prob": [0.2],
                "num_epochs": [5],
            },
        ],
    }
    rng = np.random.default_rng(71)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    frame["target"] = np.arange(120) % classes if classes else 20 + 5 * frame["a"]
    task = TaskType.Classification if classes else TaskType.Regression
    model = make_model(tmp_path, task, spec)
    model.fit(frame.iloc[:100])
    audit = model.tuning_record
    assert audit["status"] == "completed" and audit["candidates_completed"] == 2
    assert model.time_limit == 180
    assert model.predictor.model_best in model.predictor.model_refit_map().values()
    fitted = model.predictor._trainer.load_model(model.predictor.model_best)
    assert fitted.device.type == "cpu"
    assert next(fitted.model.parameters()).device.type == "cpu"
    prediction = model.predict(frame.iloc[100:].drop(columns="target"))
    assert np.isfinite(prediction).all()
    if classes:
        probabilities = model.predict_proba(frame.iloc[100:].drop(columns="target"))
        assert probabilities.shape == (20, classes)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    else:
        assert prediction.mean() > 10


def test_budget_truncation_and_inner_only_imputation(tmp_path, monkeypatch):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "2")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    clock = [0.0]
    monkeypatch.setattr(tuned, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_fit = tuned.TabularPredictor.fit
    captured = []

    def fit(predictor, train, **kwargs):
        if "tuning_data" in kwargs:
            captured.append((train.copy(), kwargs["tuning_data"].copy()))
        result = original_fit(predictor, train, **kwargs)
        clock[0] = 145.0  # Search allocation was 144 s; 35 s remain for the refit.
        return result

    monkeypatch.setattr(tuned.TabularPredictor, "fit", fit)
    frame = pd.DataFrame(
        {"a": np.arange(100, dtype=float), "b": np.arange(100), "target": [0, 1] * 50}
    )
    train, valid = selection_split(frame, None, SPEC, True)
    frame.loc[valid, "a"] = 1e9
    frame.loc[train[:4], "a"] = np.nan
    expected_fill = frame.iloc[train]["a"].median()
    model = make_model(tmp_path, TaskType.Classification, SPEC)
    model.fit(frame)
    assert model.time_limit == 180
    assert captured[0][0].loc[train[:4], "a"].eq(expected_fill).all()
    assert captured[0][1]["a"].eq(1e9).all()
    audit = model.tuning_record
    assert audit["candidates_completed"] == 1 and audit["search_truncated"]
    assert audit["selected_candidate"] == 0 and audit["status"] == "completed"


@pytest.mark.parametrize("invalid", [False, True])
def test_pipeline_persists_audit_and_keeps_variant_separate(
    tmp_path, monkeypatch, debug_config, invalid
):
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "2")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    frame = pd.DataFrame({"a": np.arange(120), "b": np.sin(np.arange(120)), "target": [0, 1] * 60})

    class Benchmark:
        _key_list = ["toy_0"]
        _task_type_list = [TaskType.Classification]

        def __len__(self):
            return 1

        def __iter__(self):
            yield frame.iloc[:100], frame.iloc[100:], "toy_0", TaskType.Classification

        def training_groups(self, key, training):
            return None

    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    spec = {**SPEC, "grid": [{"n_estimators": [-1]}, {}] if invalid else [{}, {"max_depth": [2]}]}
    cell = tmp_path / "run" / "cap_10000_n100"
    debug_config.update(
        output_dir=str(cell),
        models=["RF", "RF-TUNED"],
        model_tuning={"RF-TUNED": spec},
        autogluon_time_limit=180,
    )
    predictions.compute_predictions(debug_config)
    repository = ResultRepository.from_root(cell.parent)
    attempts = {a.model: a for a in repository.current_attempts()}
    assert set(attempts) == {"RF", "RF-TUNED"}
    assert attempts["RF"].status == "pass"
    attempt = attempts["RF-TUNED"]
    assert attempt.status == ("fail" if invalid else "pass"), attempt.record["error"]
    audit = attempt.record["tuning"]
    assert audit["status"] == ("failed" if invalid else "completed")
    assert audit["specification"] == spec
    if invalid:
        assert audit["candidates"][0]["error"]
    else:
        assert audit["candidates_completed"] == 2
        assert attempt.record["n_models_trained"] >= 3
        predictions.compute_predictions(debug_config)
        assert len(ResultRepository.from_root(cell.parent).attempts()) == 2
