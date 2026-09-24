"""The packaged dashboard builds from SQLite, without repository or paper files."""

import hashlib
import json

import pandas as pd
import pytest

from tabbench_bio import dashboard
from tabbench_bio.cli import main
from tabbench_bio.config import config_for_cell
from tabbench_bio.elo import compute_elo, fold_scores
from tabbench_bio.leaderboard import Leaderboard
from tabbench_bio.result_store import ResultRepository, consolidate_results


@pytest.fixture
def database(tmp_path):
    root = tmp_path / "results"
    for n in (20, 50):
        cell = root / f"cap_100_n{n}"
        config = config_for_cell(
            100,
            n,
            datasets=["toy"],
            datasets_regression=["reg"],
            models=["RF", "DUMMY", "NEW", "PARTIAL"],
            limits={},
            overrides={},
            n_rep=1,
            cv_folds=2,
            time_limit=60,
            out_dir=str(cell),
            cache_dir=".cache",
            test_size=0.2,
            random_state=42,
            min_samples_per_class=2,
        )
        repository = ResultRepository(cell, config)
        for seed in (0, 1):
            for dataset, values in (("toy", [0, 0, 1, 1]), ("reg", [1.0, 2.0, 3.0, 4.0])):
                truth = pd.DataFrame({"target": values}, index=range(seed * 4, seed * 4 + 4))
                for model in config["models"]:
                    if model == "PARTIAL" and seed == 1:
                        continue
                    failed = n == 50 and model == "NEW" and dataset == "toy" and seed == 0
                    record = {
                        "dataset": dataset + "_0",
                        "model": model,
                        "status": "fail" if failed else "pass",
                        "reason": "fit_oom" if failed else "",
                        "n_train_samples": n,
                        "train_time_s": 1.0 if n == 20 else 10.0,
                        "inference_time_s": 0.5,
                    }
                    prediction = (
                        truth.assign(target=[0, 0, 1, 0])
                        if model == "RF"
                        else truth.assign(target=0)
                    )
                    proba = (
                        pd.DataFrame(
                            {0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]}, index=truth.index
                        )
                        if dataset == "toy"
                        else None
                    )
                    repository.write(
                        record,
                        seed=seed,
                        ground_truth=truth,
                        prediction=None if failed else prediction,
                        probability=None if failed else proba,
                    )
    return consolidate_results(root)


def test_dashboard_schema_ratings_fallbacks_and_read_only(database, tmp_path, monkeypatch):
    original = hashlib.sha256(database.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "site"
    payload, strict = dashboard.build_website(database, output, n_boot=8)
    parsed = json.loads((output / "data/dashboard.json").read_text())
    assert parsed["schema_version"] == 9
    assert set(parsed["analysis_views"]) == {"strict", "adaptive", "conditional"}
    assert parsed["models"]["NEW"]["category"] == "Custom"
    assert parsed["meta"]["reference_cell"] == "cap_100_n20"
    assert parsed["progress"]["recorded"] < parsed["progress"]["expected"]
    assert "PARTIAL" not in {row["model_id"] for row in payload["cell_elo"]}
    for cell in ("cap_100_n20", "cap_100_n50"):
        lb = Leaderboard.from_sqlite(database, cell=cell)
        expected = compute_elo(fold_scores(lb._clf_metrics, lb._reg_metrics), n_boot=8)
        actual = [row for row in payload["cell_elo"] if row["cell"] == cell]
        assert {row["model_id"]: row["Elo"] for row in actual} == dict(
            zip(expected.model_id, expected.Elo)
        )
    costs = parsed["analysis_views"]["adaptive"]["cost_grid"]
    new_cost = next(
        row
        for row in costs
        if row["cell"] == "cap_100_n50" and row["model_id"] == "NEW" and row["domain"] == "all"
    )
    assert new_cost["train_time_s"] == 5.5  # Median of the source and nominal successful fits.
    assert strict["classification"].query("model == 'RF'")["roc_auc"].eq(1).all()
    registry = json.loads((output / "data/datasets/index.json").read_text())
    assert len(registry["datasets"]) == 2
    for dataset in registry["datasets"]:
        scores = json.loads((output / "data/datasets" / dataset["scores_file"]).read_text())
        assert scores["dataset_id"] == dataset["dataset_id"]
        assert scores["scores"]
    assert not list(output.rglob("*.tex"))
    assert (output / "models.html").is_file()
    assert (output / "assets/plotly-cartesian.min.js").is_file()
    for asset in (dashboard.PACKAGE_ROOT / "web").rglob("*"):
        if asset.is_file() and asset.name not in {"llms.txt", "robots.txt", "sitemap.xml"}:
            relative = asset.relative_to(dashboard.PACKAGE_ROOT / "web")
            assert (output / relative).read_bytes() == asset.read_bytes(), relative
    guide = (output / "llms.txt").read_text(encoding="utf-8")
    for section in (
        "Current reference results",
        "Evaluated models",
        "Agent skill",
        "Citation",
        "Machine-readable data",
    ):
        assert f"## {section}" in guide
    assert "2 registered datasets: 1 classification, 1 regression." in guide
    assert parsed["meta"]["snapshot_utc"] in guide
    assert "downloadable LaTeX" not in guide
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original
    assert not list(database.parent.glob("results.sqlite-*"))
    assert str(database.parent) not in (output / "data/dashboard.json").read_text()
    monkeypatch.setattr(
        dashboard, "compute_elo", lambda *a, **k: pytest.fail("Unchanged Elo should be cached")
    )
    monkeypatch.setattr(
        "tabbench_bio.leaderboard._sqlite_frame",
        lambda *a: pytest.fail("Unchanged fold metrics should be cached"),
    )
    assert (tmp_path / "site.cache/fold_metrics.sqlite").is_file()
    assert not list(output.rglob("*.sqlite"))
    dashboard.build_website(database, output, n_boot=8)


def test_cli_writes_matching_reports_and_website(database, tmp_path, monkeypatch):
    output = tmp_path / "web"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tabbench-bio",
            "leaderboard",
            str(database),
            "--out",
            str(output),
            "--bootstrap-rounds",
            "8",
            "--workers",
            "2",
        ],
    )
    main()
    payload = json.loads((output / "data/leaderboard.json").read_text())
    for cell in ("cap_100_n20", "cap_100_n50"):
        report = output / "local" / cell / "overall"
        assert (report / "elo.png").is_file()
        assert not (report / "elo.svg").exists()
        assert not (report / "report.html").exists()
        table = pd.read_csv(report / "leaderboard.csv").dropna(subset=["Elo"])
        assert dict(zip(table.model_id, table.Elo)) == {
            row["model_id"]: row["Elo"] for row in payload["cell_elo"] if row["cell"] == cell
        }


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_incomplete_database_without_rf_builds_an_empty_ranking(tmp_path, task):
    root = tmp_path / "empty"
    config = config_for_cell(
        100,
        20,
        datasets=["toy"] if task == "classification" else [],
        datasets_regression=["toy"] if task == "regression" else [],
        models=["NEW"],
        limits={},
        overrides={},
        n_rep=1,
        cv_folds=2,
        time_limit=60,
        out_dir=str(root / "cap_100_n20"),
        cache_dir=".cache",
        test_size=0.2,
        random_state=42,
        min_samples_per_class=2,
    )
    repository = ResultRepository(root / "cap_100_n20", config)
    repository.write(
        {"dataset": "toy_0", "model": "NEW", "status": "fail", "reason": "fit_error"}, seed=0
    )
    database = consolidate_results(root)
    result, _ = dashboard.build_website(database, tmp_path / "site", n_boot=8)
    assert result["cell_elo"] == []
    assert result["reference"] == []
    assert result["cost_grid"] == []
    assert result["progress"]["fraction"] == 0.5
