import argparse
import json

import numpy as np
import pandas as pd
import pytest

from tabbench_bio import Leaderboard
from tabbench_bio.evaluate import add_arguments, generate_report, main
from tabbench_bio.evaluation_report import paired_comparisons


def test_pairs_use_dataset_means_and_holm_family():
    means = pd.DataFrame(
        {"new": [1, 2, 3, 4], "a": [0, 1, 2, 3], "b": [2, 3, 4, 5], "absent": [np.nan] * 4}
    )
    pairs = paired_comparisons(means, "new").set_index("model")
    assert pairs.loc["a", ["datasets", "wins", "ties", "losses"]].tolist() == [4, 4, 0, 0]
    assert pairs.loc["a", ["mean_delta", "ci_low", "ci_high"]].tolist() == [1, 1, 1]
    assert pairs.loc["b", "losses"] == 4
    assert pairs.loc["a", "p_holm"] == pytest.approx(min(3 * pairs.loc["a", "p_raw"], 1))
    assert pd.isna(pairs.loc["absent", "p_holm"])


def test_regression_wins_follow_lower_error_and_single_targets_have_no_test():
    pairs = paired_comparisons(
        pd.DataFrame({"new": [1, 2], "base": [2, np.nan]}), "new", lower_is_better=True
    ).iloc[0]
    assert pairs["wins"] == 1 and pairs["mean_delta"] == -1
    assert pd.isna(pairs["p_raw"]) and pd.isna(pairs["p_holm"])
    assert pd.isna(pairs["ci_low"]) and pd.isna(pairs["ci_high"])


def test_all_ties_are_a_valid_null_comparison():
    pair = paired_comparisons(pd.DataFrame({"new": [1, 1], "base": [1, 1]}), "new").iloc[0]
    assert pair["p_holm"] == 1 and pair["ties"] == 2


def test_tuned_models_are_included_by_default():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert parser.parse_args([]).exclude_model == []


def test_regeneration_reuses_ratings_only_for_unchanged_scores(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--report-only", "--output", str(tmp_path), "--name", "New"])
    frame = pd.DataFrame(
        [
            {"key": key, "seed": seed, "model": model, "f1_macro": score}
            for key in ("a", "b")
            for seed in (0, 1)
            for model, score in (("RF", 0.5), ("New", 0.7))
        ]
    )
    ratings = pd.DataFrame(
        {
            "model_id": ["RF", "New"],
            "Elo": [1000, 1100],
            "Elo_lo": [1000, 1050],
            "Elo_hi": [1000, 1150],
            "n_targets": [2, 2],
        }
    )
    monkeypatch.setattr("tabbench_bio.leaderboard.compute_elo", lambda *a, **kw: ratings)
    monkeypatch.setattr(
        "tabbench_bio.evaluate.write_report",
        lambda lb, ranking, output, **kw: ranking.to_csv(output / "leaderboard.csv", index=False),
    )
    status = pd.DataFrame(columns=["seed", "key", "model", "status", "reason"])
    generate_report(args, Leaderboard(pd.DataFrame(), frame), {}, status)

    def recompute(*args, **kwargs):
        raise RuntimeError("ratings need recomputation")

    monkeypatch.setattr("tabbench_bio.leaderboard.compute_elo", recompute)
    generate_report(args, Leaderboard(pd.DataFrame(), frame.iloc[::-1]), {}, status)
    frame.loc[0, "f1_macro"] = 0.6
    with pytest.raises(RuntimeError, match="need recomputation"):
        generate_report(args, Leaderboard(pd.DataFrame(), frame), {}, status)


@pytest.mark.parametrize("task", ["classification", "regression", "overall"])
def test_report_only_does_not_load_estimator_or_reference(tmp_path, monkeypatch, task):
    (tmp_path / "metrics").mkdir()
    frame = pd.DataFrame(
        [
            {
                "seed": seed,
                "key": key,
                "model": model,
                "f1_macro": value,
                "imputed": model == "BASE-TUNED" and key == "two",
            }
            for seed in (0, 1)
            for key in ("one", "two")
            for model, value in (("New <model>", 0.8), ("RF", 0.7), ("BASE-TUNED", 0.6))
        ]
    )
    selected_task = "regression" if task == "regression" else "classification"
    if task != "regression":
        frame.to_csv(tmp_path / "metrics/classification_metrics.csv", index=False)
    if task != "classification":
        regression = frame.rename(columns={"f1_macro": "rmse"}).assign(key="reg_" + frame["key"])
        regression["rmse"] = 1 - regression["rmse"]
        regression.to_csv(tmp_path / "metrics/regression_metrics.csv", index=False)
    saved_metrics = tmp_path / f"metrics/{selected_task}_metrics.csv"
    config = {"cv_folds": 2, "train_subsample": 100, "bio_max_features": 10000}
    (tmp_path / "config.json").write_text(json.dumps(config))

    def rank(self, task, **kwargs):
        models = sorted(pd.concat([self._clf_metrics, self._reg_metrics])["model"].unique())
        return pd.DataFrame(
            {
                "model_id": models,
                "Model": models,
                "Rank": range(1, len(models) + 1),
                "Elo": 1000,
                "Elo_lo": 900,
                "Elo_hi": 1100,
                "Score": 0.5,
                "Avg Rank": 2.0,
                "# Targets": 2,
            }
        )

    monkeypatch.setattr(Leaderboard, "rank", rank)
    monkeypatch.setattr(
        "tabbench_bio.evaluate.load_model_factory",
        lambda _: pytest.fail("No estimator import during report regeneration"),
    )
    monkeypatch.setattr(
        "tabbench_bio.evaluate.latest_reference",
        lambda _: pytest.fail("No reference download during report regeneration"),
    )
    arguments = [
        "--report-only",
        "--name",
        "New <model>",
        "--output",
        str(tmp_path),
        "--task",
        task,
    ]
    main(arguments)
    report = (tmp_path / "report" / "report.html").read_text(encoding="utf-8")
    assert "BASE-TUNED" in report and "New &lt;model&gt;" in report
    assert 'src="data:image/svg+xml;base64,' in report
    assert "<a " not in report and 'src="leaderboard.svg"' not in report
    successful = pd.read_csv(
        tmp_path / "report" / f"paired_successful_{selected_task}.csv"
    ).set_index("model")
    assert successful.loc["BASE-TUNED", "datasets"] == 1
    before = saved_metrics.read_bytes()
    main(arguments + ["--exclude-model", "BASE-TUNED"])
    report = (tmp_path / "report" / "report.html").read_text(encoding="utf-8")
    assert "TUNED" not in report and "excluded" not in report.lower()
    assert saved_metrics.read_bytes() == before
    if task == "overall":
        assert "Macro-F1" in report and "RMSE" in report
        assert pd.read_csv(tmp_path / "report" / "dataset_scores_classification.csv").shape[0] == 2
        assert pd.read_csv(tmp_path / "report" / "dataset_scores_regression.csv").shape[0] == 2
