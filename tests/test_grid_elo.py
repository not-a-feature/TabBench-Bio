"""The grid and single-cell leaderboards must compare the same folds."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tabbench_bio import site
from tabbench_bio.elo import compute_elo, fold_scores


def test_grid_elo_uses_fold_wins_while_surfaces_use_means(tmp_path, monkeypatch):
    rows = []
    for seed, score in enumerate([0.61, 0.61, 0.61, 0.1, 0.1]):
        for model, value in [("RF", 0.6), ("LR", score)]:
            rows.append(
                {
                    "dataset": "synthetic",
                    "key": "synthetic_0",
                    "seed": seed,
                    "model": model,
                    "max_features": "2000",
                    "n_train": "100",
                    "f1_macro": value,
                }
            )
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.assign(dataset="other", key="other_0")], ignore_index=True)
    (tmp_path / "data").mkdir()
    frame.to_csv(tmp_path / "data" / "feature_grid_metrics.csv", index=False)
    monkeypatch.setattr(site, "is_bio_dataset", lambda _: True)
    monkeypatch.setattr(
        site, "get_spec", lambda name: SimpleNamespace(data_type=name.title(), source="local")
    )
    monkeypatch.setattr(site, "_dataset_display", lambda value: value)

    grid = site._build_grid(str(tmp_path))
    expected = compute_elo(fold_scores(frame, None)).set_index("model_id")["Elo"].to_dict()
    for domain in ["all", "Synthetic", "Other"]:
        actual = {row["model_id"]: row["Elo"] for row in grid["elo"][f"f1_macro|2000|100|{domain}"]}
        assert actual == expected
        assert actual["LR"] > actual["RF"] == 1000
    surface = grid["surface"]["synthetic"]["f1_macro"]
    assert surface["LR"]["z"][0][0] == pytest.approx(0.406)
    assert surface["RF"]["z"][0][0] == pytest.approx(0.6)


def test_grid_rejects_mean_only_legacy_input(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "feature_grid_summary.csv").write_text("model,f1_macro__mean\nRF,0.6\n")
    with pytest.raises(AssertionError, match="real fold scores"):
        site._build_grid(str(tmp_path))


@pytest.mark.parametrize("duplicate", [False, True])
def test_grid_rejects_missing_or_duplicate_fold_identifiers(tmp_path, duplicate):
    row = {
        "dataset": "synthetic",
        "key": "synthetic_0",
        "model": "RF",
        "max_features": "2000",
        "n_train": "100",
        "f1_macro": 0.6,
    }
    if duplicate:
        row["seed"] = 0
    (tmp_path / "data").mkdir()
    pd.DataFrame([row, row]).to_csv(tmp_path / "data" / "feature_grid_metrics.csv", index=False)
    with pytest.raises(AssertionError, match="Duplicate|Missing"):
        site._build_grid(str(tmp_path))
