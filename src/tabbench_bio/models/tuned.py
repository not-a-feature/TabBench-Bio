"""Fixed-grid selection on an inner holdout followed by a fresh full-data refit."""

import math
import os
import shutil
import time
from importlib.metadata import version

import pandas as pd
import torch
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor

from tabbench_bio.dataset import TaskType
from tabbench_bio.model import AutoGluonModel, _resolve_hyperparameters
from tabbench_bio.tuning import candidates, selection_split, tuning_fingerprint, validate_tuning


class TunedModel(AutoGluonModel):
    """One benchmark entry, with no candidate predictions exposed as benchmark results.

    Candidate fit errors fail the whole unit. A search stopped by its time budget
    may select an already completed candidate, but is explicitly marked truncated.
    """

    def __init__(self, *, tuning, nan_policy, **kwargs):
        (key,) = kwargs["models"]
        validate_tuning(key, tuning)
        assert not kwargs["ensemble"] and not kwargs["optimize"], (
            "Tuned variants require ensemble=false and optimize=false"
        )
        assert kwargs["autogluon_presets"] == "medium_quality"
        kwargs["models"] = [tuning["base_model"]]
        super().__init__(**kwargs)
        self.tuning = tuning
        self.nan_policy = nan_policy
        assert nan_policy in ("native", "none", "median", "mean", "zero")
        self._fill = pd.Series(dtype=float)
        task = "classification" if self.task_type == TaskType.Classification else "regression"
        self.tuning_record = {
            "model_key": key,
            "specification": tuning,
            "fingerprint": tuning_fingerprint(tuning),
            "metric": self.metric,
            "score_direction": "higher_is_better (AutoGluon negates losses)",
            "tie_break": "first candidate in declared grid order",
            "candidate_error_policy": "fail_unit",
            "status": "not_started",
            "budget_seconds": self.time_limit,
            "search_fraction": tuning["search_fraction"],
            "versions": {p: version(p) for p in ("autogluon.tabular", "scikit-learn")},
            "candidates": [
                {"parameters": p, "status": "not_started"} for p in candidates(tuning, task)
            ],
            "selected_candidate": None,
            "search_truncated": False,
        }

    def _fit_fill(self, frame):
        features = frame.drop(columns="target")
        if self.nan_policy == "median":
            self._fill = features.median(numeric_only=True)
        elif self.nan_policy == "mean":
            self._fill = features.mean(numeric_only=True)
        elif self.nan_policy == "zero":
            self._fill = pd.Series(0.0, index=features.columns)
        return self._transform(frame)

    def _transform(self, frame):
        return frame.fillna(self._fill)

    def fit(self, data_train, raise_on_no_models_fitted=True, *, groups=None):
        started = time.monotonic()
        total_budget = self.time_limit
        try:
            return self._fit(data_train, raise_on_no_models_fitted, groups=groups, started=started)
        finally:
            self.time_limit = total_budget
            audit = self.tuning_record
            audit["total_fit_time_s"] = time.monotonic() - started
            audit["candidates_attempted"] = sum(
                c["status"] != "not_started" for c in audit["candidates"]
            )
            audit["candidates_completed"] = sum(
                c["status"] == "completed" for c in audit["candidates"]
            )
            audit["search_truncated"] = audit["candidates_completed"] < len(audit["candidates"])
            if audit["status"] != "completed":
                audit["status"] = "failed"

    def _fit(self, data_train, raise_on_no_models_fitted, *, groups, started):
        total_budget = self.time_limit
        deadline = started + total_budget * self.tuning["search_fraction"]
        audit = self.tuning_record
        audit["status"] = "searching"
        train_idx, valid_idx = selection_split(
            data_train, groups, self.tuning, self.task_type == TaskType.Classification
        )
        audit["split"] = {
            "train_row_ids": [str(x) for x in data_train.index[train_idx]],
            "validation_row_ids": [str(x) for x in data_train.index[valid_idx]],
            "grouped": groups is not None and groups.nunique() < len(groups),
        }
        train = self._fit_fill(data_train.iloc[train_idx])
        valid = self._transform(data_train.iloc[valid_idx])
        if self.problem_type == "regression":
            mean, std = float(train["target"].mean()), float(train["target"].std()) or 1.0
            train, valid = train.copy(), valid.copy()
            train["target"] = (train["target"] - mean) / std
            valid["target"] = (valid["target"] - mean) / std
            audit["inner_target_scaling"] = {"mean": mean, "std": std}
        # Pre-fit on inner training rows: AutoGluon must not learn features from validation.
        generator = AutoMLPipelineFeatureGenerator()
        generator.fit_transform(train.drop(columns="target"), y=train["target"])
        num_cpus = int(
            os.environ["TABBENCH_MODEL_CPUS"]
            if "TABBENCH_MODEL_CPUS" in os.environ
            else (
                os.environ["SLURM_CPUS_PER_TASK"]
                if "SLURM_CPUS_PER_TASK" in os.environ
                else os.cpu_count()
            )
        )
        num_gpus = min(1, torch.cuda.device_count())
        scores = []
        for index, candidate in enumerate(audit["candidates"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            parameters = _resolve_hyperparameters(self.models, num_gpus)
            for configurations in parameters.values():
                assert len(configurations) == 1
                configurations[0].update(candidate["parameters"])
            path = os.path.join(self.autogluon_path, f"candidate-{index}")
            self.predictor = TabularPredictor(
                label="target",
                problem_type=self.problem_type,
                eval_metric=self.metric,
                path=path,
                verbosity=0,
            )
            candidate["status"] = "running"
            candidate_started = time.monotonic()
            try:
                self.predictor.fit(
                    train,
                    tuning_data=valid,
                    hyperparameters=parameters,
                    presets="medium_quality",
                    time_limit=remaining,
                    feature_generator=generator,
                    num_bag_folds=0,
                    num_stack_levels=0,
                    fit_weighted_ensemble=False,
                    hyperparameter_tune_kwargs=None,
                    calibrate_decision_threshold=False,
                    refit_full=False,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    ag_args_fit={"ag.max_features": None, "ag.max_memory_usage_ratio": 100.0},
                    raise_on_no_models_fitted=True,
                )
                board = self.predictor.leaderboard(silent=True)
                assert len(board) == 1, "Each candidate must fit exactly one model"
                score = float(board.iloc[0]["score_val"])
                assert math.isfinite(score), "Non-finite validation score"
                candidate.update(
                    status="completed",
                    validation_score=score,
                    fit_time_s=float(board.iloc[0]["fit_time"]),
                )
                scores.append((score, -index))
            finally:
                candidate["wall_time_s"] = time.monotonic() - candidate_started
                if candidate["status"] == "running":
                    candidate["status"] = "failed"
                # Logs and audit survive in the result record; fitted candidates are disposable.
                self.predictor = None
                if os.path.isdir(path):
                    shutil.rmtree(path)
        audit["search_time_s"] = time.monotonic() - started
        assert scores, "No candidate completed within the search budget"
        winner = -max(scores)[1]
        audit["selected_candidate"] = winner
        self.parameter_overrides = dict(audit["candidates"][winner]["parameters"])
        audit["selected_parameters"] = self.parameter_overrides
        audit["status"] = "refitting"
        self.time_limit = total_budget - (time.monotonic() - started)
        assert self.time_limit > 0, "No time remains for full-data refitting"
        # A fresh generator and imputer are fitted on all permitted outer-training rows.
        full_train = self._fit_fill(data_train)
        super().fit(full_train, raise_on_no_models_fitted=raise_on_no_models_fitted, groups=groups)
        refits = self.predictor.model_refit_map()
        assert self.predictor.model_best in refits.values(), "Winner was not fully refitted"
        assert time.monotonic() - started <= total_budget, "Total search and refit budget exceeded"
        audit["status"] = "completed"
        return self.predictor

    def predict(self, data_test):
        return super().predict(self._transform(data_test))

    def predict_proba(self, data_test):
        return super().predict_proba(self._transform(data_test))

    def get_fit_stats(self):
        stats = super().get_fit_stats()
        completed = [c for c in self.tuning_record["candidates"] if c["status"] == "completed"]
        if stats:
            stats["n_models_trained"] += len(completed)
            stats["n_base_models"] += len(completed)
            stats["ag_total_fit_time_s"] += sum(c["fit_time_s"] for c in completed)
            stats["ag_time_per_model_s"] = stats["ag_total_fit_time_s"] / stats["n_base_models"]
        return {**stats, "tuning": self.tuning_record}
