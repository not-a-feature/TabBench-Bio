import json

import pandas as pd
import pytest

from tabbench_bio.cli import main
from tabbench_bio.result_store import ResultRepository, consolidate_results


def test_saved_evaluation_reports_leaderboard_and_modalities_without_reference(
    tmp_path, monkeypatch
):
    saved = tmp_path / "saved"
    (saved / "metrics").mkdir(parents=True)
    (saved / "report").mkdir()
    (saved / "config.json").write_text(
        json.dumps(
            {
                "datasets_classification": ["OpenML-1083"],
                "datasets_regression": ["toy"],
                "output_dir": "/other/machine/cap_2000_n20",
                "cv_folds": 1,
                "train_subsample": 20,
                "bio_max_features": 2000,
            }
        ),
        encoding="utf-8",
    )
    (saved / "report" / "report_config.json").write_text(
        json.dumps({"name": "My <model>", "cell": "cap_10000_n100"}), encoding="utf-8"
    )
    for task, target, scores in (
        ("classification", "OpenML-1083_0", {"f1_macro": [0.9, 0.5], "roc_auc": [0.8, 0.6]}),
        ("regression", "toy_0", {"rmse": [0.4, 0.6]}),
    ):
        pd.DataFrame({"key": target, "seed": 0, "model": ["My <model>", "RF"], **scores}).to_csv(
            saved / "metrics" / f"{task}_metrics.csv", index=False
        )

    monkeypatch.setattr(
        "sys.argv",
        ["tabbench-bio", "report", "--evaluation", str(saved), "--bootstrap-rounds", "3"],
    )
    main()

    output = saved / "category-report"
    overview = (output / "report.html").read_text(encoding="utf-8")
    assert "My &lt;model&gt;" in overview
    assert overview.index("<h2>Ranking</h2>") < overview.index("<h2>Modalities</h2>")
    assert 'href="#modalities"' not in overview
    for figure in (
        "leaderboard",
        "dataset_scores_classification",
        "dataset_scores_regression",
        "paired_classification",
        "paired_regression",
    ):
        assert (output / f"{figure}.svg").is_file()
        assert f'alt="{figure.replace("_", " ")}"' in overview
    for name in ("gene-expression", "custom"):
        assert f"categories/{name}.html" in overview
        page = (output / "categories" / f"{name}.html").read_text(encoding="utf-8")
        assert "p=2,000, n=20" in page
        assert "Pooled (all cells)" not in page
    classification = pd.read_csv(output / "categories/gene-expression.csv")
    assert set(classification["metric"]) == {"f1_macro"}
    regression = pd.read_csv(output / "categories/custom.csv").set_index("model_id")
    assert regression.loc["RF", "Ahead"] == 1
    assert regression.loc["RF", "Median Δ RMSE"] == pytest.approx(-0.2)


def test_database_report_keeps_cell_pairs_and_excludes_imputed_fits(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for samples in (100, 200):
        repository = ResultRepository(
            root / f"cap_10000_n{samples}",
            {
                "models": ["DUMMY", "RF", "CUSTOM"],
                "cv_folds": 2,
                "n_repetitions": 1,
                "random_state": 0,
                "bio_max_features": 10000,
                "train_subsample": samples,
                "datasets_classification": ["OpenML-1083"],
                "datasets_regression": [],
            },
        )
        for seed in range(2):
            truth = pd.DataFrame({"target": [0, 1, 0, 1]}, index=range(seed * 4, seed * 4 + 4))
            probability = pd.DataFrame({"0": 0.5, "1": 0.5}, index=truth.index)
            for model in ("DUMMY", "RF", "CUSTOM"):
                record = {
                    "dataset": "OpenML-1083_0",
                    "model": model,
                    "status": "pass",
                    "reason": "",
                }
                if model == "CUSTOM" and samples == 200 and seed == 0:
                    repository.write({**record, "status": "fail", "reason": "fit_error"}, seed=seed)
                else:
                    repository.write(
                        record,
                        seed=seed,
                        prediction=truth.assign(target=0) if model == "DUMMY" else truth,
                        probability=probability,
                        ground_truth=truth,
                    )
    database = consolidate_results(root)
    original = database.read_bytes()
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tabbench-bio",
            "report",
            "--database",
            str(database),
            "--model",
            "CUSTOM",
            "--output",
            str(output),
            "--metric-cache",
            str(tmp_path / "metrics.sqlite"),
            "--bootstrap-rounds",
            "3",
        ],
    )
    main()

    assert database.read_bytes() == original
    overview = (output / "report.html").read_text(encoding="utf-8")
    assert overview.index("<h2>Ranking</h2>") < overview.index("<h2>Modalities</h2>")
    assert "cap_10000_n100" in overview
    page = (output / "categories/gene-expression.html").read_text(encoding="utf-8")
    assert '<option value="cap_10000_n100" selected>' in page
    assert "Pooled (all cells)" in page
    assert '<script src="gene-expression/data.js"></script>' in page
    assert '<input type="checkbox" id="all-models">' in page
    assert "data:image/svg+xml" not in page
    data_js = (output / "categories/gene-expression/data.js").read_text(encoding="utf-8")
    assert data_js.startswith("window.REPORT_DATA = ")
    panels = json.loads(data_js.removeprefix("window.REPORT_DATA = ").rstrip().removesuffix(";"))[
        "panels"
    ]
    assert {(panel["cell"], panel["metric"]) for panel in panels} == {
        (cell, metric)
        for cell in ("cap_10000_n100", "cap_10000_n200", "pooled")
        for metric in ("f1_macro", "matthews_corrcoef", "balanced_accuracy")
    }
    for panel in panels:
        for version in ("all", "default"):
            image = output / "categories" / panel["plot"][version]
            assert image.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    pooled = next(
        panel for panel in panels if (panel["cell"], panel["metric"]) == ("pooled", "f1_macro")
    )
    assert "<span>Targets</span><strong>1</strong>" in pooled["cards"]
    assert "<span>Scored folds</span><strong>4</strong>" in pooled["cards"]
    summary = pd.read_csv(output / "categories/gene-expression.csv")
    assert set(summary["cell"]) == {"cap_10000_n100", "cap_10000_n200", "pooled"}
    rf = summary.loc[summary["model_id"].eq("RF") & summary["metric"].eq("f1_macro")].set_index(
        "cell"
    )
    assert rf["Elo"].eq(1000).all()
    assert rf.loc["cap_10000_n100", "Ties"] == 1
    assert rf.loc["cap_10000_n200", "Successful Pairs"] == 0
    assert rf.loc["pooled", "Pairs"] == 2
    assert rf.loc["pooled", "Behind"] == 1
    assert rf.loc["pooled", "Successful Pairs"] == 1
