"""Public ranking and plotting use the same fold battles as the website."""

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.elo import compute_elo, fold_scores
from tabbench_bio.leaderboard import Leaderboard


def metrics():
    return pd.DataFrame(
        [
            {"model": model, "key": key, "seed": seed, "f1_macro": score}
            for key in ["first_0", "second_0"]
            for seed, value in enumerate([0.61, 0.61, 0.61, 0.1, 0.1])
            for model, score in [("RF", 0.6), ("MY", value)]
        ]
    )


def test_rank_and_plot_use_fold_elo_not_mean_score():
    frame = metrics()
    lb = Leaderboard(pd.DataFrame(), frame)
    actual = lb.rank("classification").set_index("model_id")
    expected = compute_elo(fold_scores(frame, None)).set_index("model_id")
    columns = ["Elo", "Elo_lo", "Elo_hi"]
    pd.testing.assert_frame_equal(actual[columns], expected[columns])
    assert actual.index.tolist() == ["MY", "RF"]
    assert actual.loc["MY", "Score"] < actual.loc["RF", "Score"]
    assert actual.loc["MY", "Rank"] == 1
    figure = lb.plot("classification")
    np.testing.assert_array_equal(
        figure.axes[0].collections[-1].get_offsets()[:, 0], actual.Elo.iloc[::-1]
    )
    assert "Elo" in figure.axes[0].get_xlabel()
    assert "95% CI" in lb.summary("classification")


def test_missing_anchor_does_not_silently_rank_by_score():
    lb = Leaderboard(pd.DataFrame(), metrics().replace({"RF": "OTHER"}))
    result = lb.rank("classification")
    assert result["Elo"].isna().all() and result["Rank"].isna().all()
    with pytest.raises(ValueError, match="Include RF"):
        lb.plot("classification")


def test_stale_metadata_cannot_supply_elo_and_adding_model_invalidates_cache():
    frame = metrics()
    lb = Leaderboard(pd.DataFrame(), frame, pd.DataFrame([{"model_id": "MY", "Elo": -9999}]))
    assert lb.rank("classification").iloc[0].Elo > 1000
    lb.add_results("NEW", frame[frame.model == "MY"].assign(f1_macro=0.8))
    actual = lb.rank("classification")
    assert actual.iloc[0].model_id == "NEW"
    with pytest.raises(AssertionError, match="already present"):
        lb.add_results("NEW", frame[frame.model == "MY"])


def test_custom_evaluation_clones_every_configured_fold_and_propagates_errors(monkeypatch):
    from sklearn.base import BaseEstimator, ClassifierMixin

    from tabbench_bio import benchmark, config
    from tabbench_bio.dataset import TaskType

    settings = {
        "cv_folds": 5,
        "n_repetitions": 1,
        "random_state": 42,
        "dataset_names_classification": ["toy"],
        "dataset_names_regression": [],
    }
    seen = []
    fitted = []
    train = pd.DataFrame({"feature": [0, 1, 2, 3], "target": [0, 1, 0, 1]})

    def configure(settings):
        seen.append(settings["random_state"])
        return [(train, train, "toy_0", TaskType.Classification)]

    class Model(ClassifierMixin, BaseEstimator):
        def __init__(self, fail=False):
            self.fail = fail

        def fit(self, X, y):
            if self.fail:
                raise RuntimeError("fit failed")
            fitted.append(self)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    monkeypatch.setattr(config, "load_config", lambda path: settings.copy())
    monkeypatch.setattr(benchmark, "configure_benchmark", configure)
    lb = Leaderboard(pd.DataFrame(), pd.DataFrame())
    original = Model()
    result = lb.evaluate_and_add("CUSTOM", original, "unused", task="classification")
    assert seen == list(range(5)) and result.seed.tolist() == list(range(5))
    assert len({id(model) for model in fitted}) == 5
    assert all(model is not original for model in fitted)
    with pytest.raises(ValueError, match="all 5"):
        lb.evaluate_and_add("PARTIAL", original, "unused", seeds=3, task="classification")
    with pytest.raises(RuntimeError, match="fit failed"):
        lb.evaluate_and_add("FAILED", Model(fail=True), "unused", task="classification")
    assert "FAILED" not in set(lb._clf_metrics.model)

    monkeypatch.setattr(
        benchmark,
        "configure_benchmark",
        lambda settings: [
            (
                train if settings["random_state"] != 4 else None,
                train,
                "toy_0",
                TaskType.Classification,
            )
        ],
    )
    with pytest.raises(AssertionError, match="every configured fold"):
        lb.evaluate_and_add("INCOMPLETE", Model(), "unused", task="classification")
    assert "INCOMPLETE" not in set(lb._clf_metrics.model)
