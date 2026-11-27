"""The packaged dashboard builds from SQLite, without repository or paper files."""

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from tabbench_bio import dashboard, resources, web_metadata
from tabbench_bio.cli import main
from tabbench_bio.config import config_for_cell
from tabbench_bio.dashboard_data import read_inputs
from tabbench_bio.elo import compute_elo, fold_scores
from tabbench_bio.external_results import load_external_evaluations
from tabbench_bio.io_utils import pack_attempt_record, read_result_blob
from tabbench_bio.leaderboard import Leaderboard
from tabbench_bio.model_metadata import PUBLIC_METADATA_FIELDS
from tabbench_bio.result_store import ResultRepository, consolidate_results
from tabbench_bio.split_manifest import unit_id


@pytest.mark.parametrize("parent", ["RF", "XT", "LR", "XGB"])
def test_tuned_model_metadata_preserves_family(parent):
    registry = dashboard.load_model_registry()
    base = dashboard.model_meta(parent, registry)
    tuned = dashboard.model_meta(f"{parent}-TUNED", registry)
    assert tuned["tuned_from"] == parent
    assert tuned["category"] == base["category"]
    assert tuned["color"] == base["color"]
    assert tuned["display"] == f"{base['display']} (tuned)"
    assert tuned["source_url"] == base["source_url"]
    assert tuned["source_url"].startswith("https://github.com/")
    assert base["description"]
    assert tuned["description"]


def test_source_metadata_can_override_parent_or_be_unavailable():
    registry = dashboard.load_model_registry()
    registry["CUSTOM-SEARCH"] = {
        "base_model": "RF",
        "source_url": "https://github.com/example/custom",
    }
    assert (
        dashboard.model_meta("CUSTOM-SEARCH", registry)["source_url"]
        == "https://github.com/example/custom"
    )
    assert dashboard.model_meta("UNKNOWN", registry)["source_url"] is None
    assert (
        dashboard.model_meta("CUSTOM-SEARCH", registry)["description"]
        == registry["RF"]["description"]
    )
    assert dashboard.model_meta("UNKNOWN", registry)["description"] is None
    del registry["CUSTOM-SEARCH"]["source_url"]
    with pytest.raises(KeyError, match="source_url"):
        dashboard.model_meta("CUSTOM-SEARCH", registry)


def test_all_registered_models_have_public_descriptions():
    for identifier, entry in dashboard.load_model_registry().items():
        assert entry["description"] and len(entry["description"].split()) >= 20, identifier


def test_model_metadata_uses_registry_flags_and_custom_parent():
    registry = dashboard.load_model_registry()
    assert dashboard.model_meta("TABDPT", registry)["training_data_overlap"] is True
    registry["TABDPT"]["training_data_overlap"] = False
    registry["CUSTOM-SEARCH"] = {
        "base_model": "RF",
        "training_data_overlap": True,
        "source_url": "https://github.com/scikit-learn/scikit-learn",
    }
    assert dashboard.model_meta("TABDPT", registry)["training_data_overlap"] is False
    custom = dashboard.model_meta("CUSTOM-SEARCH", registry)
    assert custom["tuned_from"] == "RF"
    assert custom["training_data_overlap"] is True
    assert custom["category"] == "Tree-based"


def test_frozen_results_supply_parent_for_retired_rosters():
    configs = [{"model_tuning": {"HISTORICAL-RF": {"base_model": "RF"}}}]
    registry = dashboard.load_model_registry(configs)
    meta = dashboard.model_meta("HISTORICAL-RF", registry)
    assert meta["tuned_from"] == "RF"
    assert meta["display"] == "Random Forest (tuned)"
    assert meta["category"] == dashboard.model_meta("RF", registry)["category"]
    configs.append({"model_tuning": {"HISTORICAL-RF": {"base_model": "XT"}}})
    with pytest.raises(AssertionError, match="Conflicting tuning parent"):
        dashboard.load_model_registry(configs)


def test_installed_registry_is_loaded_without_a_source_checkout(tmp_path, monkeypatch):
    package = tmp_path / "lib/site-packages/tabbench_bio"
    directory = package / "_resources/configs/models"
    directory.mkdir(parents=True)
    original = dashboard.load_model_registry()["RF"]
    fields = {"key", *PUBLIC_METADATA_FIELDS}
    (directory.parent / "model_metadata.json").write_text(
        json.dumps(
            [
                {key: value for key, value in original.items() if key in fields},
                {
                    "key": "CUSTOM",
                    "base_model": "RF",
                    "display": "Custom RF",
                    "training_data_overlap": True,
                },
            ]
        )
    )
    (directory / "custom.json").write_text(
        json.dumps(
            [
                {
                    "key": "CUSTOM",
                    "base_model": "RF",
                }
            ]
        )
    )
    monkeypatch.setattr(resources, "__file__", str(package / "resources.py"))
    registry = dashboard.load_model_registry()
    assert dashboard.model_meta("CUSTOM", registry)["tuned_from"] == "RF"


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
                        "train_time_s": (1.0 if n == 20 else 10.0)
                        * (20 if dataset == "reg" else 1),
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


def test_external_database_is_ranked_on_shared_folds(database, tmp_path, monkeypatch):
    external = tmp_path / "external.sql"
    shutil.copyfile(database, external)
    cell = "cap_100_n20"
    with sqlite3.connect(external) as db:
        db.execute(
            "DELETE FROM attempts WHERE cell != ? OR dataset != 'toy_0' OR model != 'RF'",
            (cell,),
        )
        db.execute("DELETE FROM cells WHERE cell != ?", (cell,))
        config = json.loads(db.execute("SELECT config_json FROM cells").fetchone()[0])
        config.update(models=["Contributor"], datasets_regression=[])
        encoded = json.dumps(config)
        db.execute(
            "UPDATE cells SET config_json=?, config_hash=?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
        db.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                "external_provenance",
                json.dumps(
                    {
                        "task": "classification",
                        "model": "Contributor",
                        "cell": cell,
                        "hardware": {"gpu": "NVIDIA H100"},
                    }
                ),
            ),
        )
        summary, blob = pack_attempt_record(
            '{"task_type":"classification","reported_metrics":{"f1_macro":0.8}}'
        )
        db.execute(
            "UPDATE attempts SET model='Contributor', prediction_hash=NULL, probability_hash=NULL, "
            "record_json=?, record_zlib=?",
            (summary, blob),
        )

    native = tmp_path / "native.sqlite"
    shutil.copyfile(database, native)
    with sqlite3.connect(native) as db:
        db.execute(
            "DELETE FROM attempts WHERE cell != ? OR dataset != 'toy_0' OR model NOT IN ('RF', 'DUMMY')",
            (cell,),
        )
        db.execute("DELETE FROM cells WHERE cell != ?", (cell,))
        db.execute("UPDATE attempts SET model='Native' WHERE model='RF'")
        encoded = json.dumps({**config, "models": ["DUMMY", "Native"]})
        db.execute(
            "UPDATE cells SET config_json=?, config_hash=?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )

    quickstart = tmp_path / "quickstart"
    (quickstart / "report").mkdir(parents=True)
    (quickstart / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (quickstart / "report/report_config.json").write_text(
        json.dumps({"cell": cell, "name": "Quickstart", "task": "classification"}), encoding="utf-8"
    )
    with sqlite3.connect(database) as db:
        witnesses = db.execute(
            "SELECT seed, dataset, ground_truth_hash FROM attempts WHERE cell=? AND model='RF' AND dataset='toy_0'",
            (cell,),
        ).fetchall()
        manifest = {
            "units": {
                unit_id(seed, key): {
                    "ground_truth_sha256": hashlib.sha256(read_result_blob(db, digest)).hexdigest()
                }
                for seed, key, digest in witnesses
            }
        }
    (quickstart / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (quickstart / "fold_metrics.csv").write_text(
        "seed,key,model,f1_macro\n0,toy_0,Quickstart,0.9\n", encoding="utf-8"
    )
    (quickstart / "attempts.csv").write_text(
        "seed,key,model,status\n0,toy_0,Quickstart,pass\n1,toy_0,Quickstart,fail\n",
        encoding="utf-8",
    )

    output = tmp_path / "site"
    _, _ = dashboard.build_website(
        database,
        output,
        external_inputs=[external, native, quickstart],
        reference_cell=cell,
        n_boot=4,
    )
    page = output / "external-results.html"
    assert not page.exists()
    artifacts = output / "artifacts.html"
    artifacts_html = artifacts.read_text(encoding="utf-8")
    assert 'href="data/external-results.sqlite" download="external.sql"' in artifacts_html
    assert "Contributor SQLite" in artifacts_html
    assert "NVIDIA H100 · author-reported Macro-F1" in artifacts_html
    for source, name in (
        (external, "external-results.sqlite"),
        (native, "external-results-2.sqlite"),
    ):
        published = output / "data" / name
        with (
            closing(sqlite3.connect(source)) as original,
            closing(sqlite3.connect(published)) as public,
        ):
            source_metadata = dict(original.execute("SELECT key, value FROM metadata"))
            assert {"host", "platform"} <= source_metadata.keys()
            assert dict(public.execute("SELECT key, value FROM metadata")) == {
                key: value
                for key, value in source_metadata.items()
                if key not in {"host", "platform"}
            }
            for table in ("attempts", "blobs", "cells"):
                query = f"SELECT * FROM {table} ORDER BY 1"
                assert public.execute(query).fetchall() == original.execute(query).fetchall()
    with ZipFile(output / "data/external-results-3.zip") as archive:
        assert set(archive.namelist()) == {
            "fold_metrics.csv",
            "split_manifest.json",
            "attempts.csv",
            "config.json",
            "report/report_config.json",
        }
        portable = tmp_path / "portable"
        archive.extractall(portable)
    portable_results = load_external_evaluations(database, [portable], cell)
    assert portable_results[0].model == "Quickstart"
    main_dashboard = output / "data/dashboard.json"
    assert "Contributor" not in json.loads(main_dashboard.read_text())["models"]
    index = output / "index.html"
    home = index.read_text(encoding="utf-8")
    app_id = hashlib.sha256((output / "js/app.js").read_bytes()).hexdigest()[:12]
    assert f'js/app.js?v={app_id}"' in home
    advanced = re.search(r'<div class="analysis-view-options"[^>]*>(.*?)</div>', home, re.S)[1]
    assert 'id="external-elo-toggle"' in advanced
    assert 'aria-pressed="false" aria-controls="elo-chart"' in advanced
    assert "Hide external models" in advanced
    assert 'id="external-elo-scope"' not in home
    assert re.search(r'id="external-elo-warning"[^>]*\bhidden\b', home)
    assert "may be less reliable than reference results and may use different hardware" in home
    assert (
        home.index('id="elo-note"')
        < home.index('id="external-elo-warning"')
        < home.index('id="elo-overlap-warning"')
    )
    payload = json.loads(re.search(r'id="external-elo-data">([^<]+)</script>', home)[1])
    assert payload["cell"] == cell
    models = payload["models"]
    assert models["Contributor"]["provenance"] == "NVIDIA H100 · author-reported"
    assert models["Contributor"]["classification_only"] is False
    assert "1 shared classification dataset" in payload["scope"][0]
    assert models["Contributor"]["artifact_url"] == "artifacts.html#external-results-download"
    assert models["Native"]["provenance"] == "External run · scored from predictions"
    assert models["Quickstart"]["artifact_url"] == "artifacts.html#external-results-download-3"
    rows = {row["model_id"]: row for row in payload["rows"]}
    assert {"Contributor", "Native", "Quickstart", "RF"} <= rows.keys()
    assert "PARTIAL" not in rows
    assert rows["RF"]["Elo"] == 1000
    assert rows["Native"]["Elo"] == rows["RF"]["Elo"]
    assert all(row["n_targets"] == 1 for row in rows.values())

    original = main_dashboard.read_bytes()
    page.write_text("<!-- Generated by tabbench-bio external comparison. -->", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "tabbench-bio",
            "leaderboard",
            str(database),
            "--external-sqlite",
            str(external),
            "--external-sqlite",
            str(native),
            "--external-evaluation",
            str(quickstart),
            "--external-only",
            "--out",
            str(output),
            "--bootstrap-rounds",
            "4",
        ],
    )
    main()
    assert main_dashboard.read_bytes() == original
    assert index.read_text(encoding="utf-8") == home
    assert artifacts.read_text(encoding="utf-8") == artifacts_html
    assert not page.exists()
    assert "external-results.html" not in (output / "sitemap.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_external_evaluations(database, [quickstart, quickstart], cell)

    with sqlite3.connect(external) as db:
        db.execute("UPDATE attempts SET ground_truth_hash=NULL WHERE seed=0")
    with pytest.raises(ValueError, match="Held-out targets differ"):
        load_external_evaluations(database, [external], cell)


def test_dashboard_schema_ratings_fallbacks_and_read_only(database, tmp_path, monkeypatch):
    original = hashlib.sha256(database.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "site"
    payload, strict = dashboard.build_website(database, output, n_boot=8)
    parsed = json.loads((output / "data/dashboard.json").read_text())
    assert parsed["schema_version"] == 12
    assert set(parsed["analysis_views"]) == {"strict"}
    assert set(parsed["analysis_view_files"]) == {"adaptive", "conditional"}
    for name, relative in parsed["analysis_view_files"].items():
        parsed["analysis_views"][name] = json.loads((output / relative).read_text())
    assert set(parsed["analysis_views"]["strict"]["domain_elo"]) == {"columns", "rows"}
    parsed["analysis_views"] = {
        name: dashboard.unpack_tables(view) for name, view in parsed["analysis_views"].items()
    }
    home = json.loads((output / "data/home.json").read_text())
    assert set(home["analysis_views"]["strict"]) == {"reference"}
    charts = dashboard.unpack_tables(json.loads((output / home["strict_chart_file"]).read_text()))
    assert charts["domain_elo"] == parsed["analysis_views"]["strict"]["domain_elo"]
    assert charts["pooled_cost_grid"] == parsed["analysis_views"]["strict"]["pooled_cost_grid"]
    models = json.loads((output / "data/model-pages.json").read_text())
    model = dashboard.unpack_tables(json.loads((output / models["model_files"]["RF"]).read_text()))
    assert model["domain_elo"] == [
        row
        for row in charts["domain_elo"]
        if row["model_id"] == "RF" and row["metric"] == "f1_macro"
    ]
    assert model["failure_summary"] == [
        row for row in parsed["failure_summary"] if row["model_id"] == "RF"
    ]
    artifacts = (output / "artifacts.html").read_text(encoding="utf-8")
    assert 'id="external-elo-toggle"' not in (output / "index.html").read_text(encoding="utf-8")
    assert 'href="data/leaderboard.json" download' in artifacts
    assert (
        'href="https://github.com/not-a-feature/TabBench-Bio/releases/latest/download/results.sqlite"'
        in artifacts
    )
    assert not set(parsed).intersection(parsed["analysis_views"]["strict"])
    for name, view in parsed["analysis_views"].items():
        assert view == json.loads(
            json.dumps(
                dashboard.clean_json(
                    dashboard.compact_dashboard_values(
                        payload["analysis_views"][name], models=payload["models"]
                    )
                )
            )
        )
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
    strict_view = parsed["analysis_views"]["strict"]
    for grid, expected_time in (
        ("cost_grid", 10.0),
        ("regression_cost_grid", 200.0),
        ("pooled_cost_grid", 105.0),
    ):
        cost = next(
            row
            for row in strict_view[grid]
            if row["cell"] == "cap_100_n50" and row["model_id"] == "RF" and row["domain"] == "all"
        )
        assert cost["train_time_s"] == expected_time
        if grid == "regression_cost_grid":
            assert cost["r2"] == pytest.approx(
                strict["regression"].query("cell == 'cap_100_n50' and model == 'RF'")["r2"].mean(),
                rel=1e-5,
            )
            assert "f1_macro" not in cost
        if grid == "pooled_cost_grid":
            assert "r2" not in cost and "f1_macro" not in cost
    assert strict["classification"].query("model == 'RF'")["roc_auc"].eq(1).all()
    registry = json.loads((output / "data/datasets/index.json").read_text())
    assert len(registry["datasets"]) == 2
    for dataset in registry["datasets"]:
        scores = json.loads((output / "data/datasets" / dataset["scores_file"]).read_text())
        assert scores["dataset_id"] == dataset["dataset_id"]
        assert scores["scores"]
    assert not list(output.rglob("*.tex"))
    home = (output / "index.html").read_text(encoding="utf-8")
    assert f'<strong id="dataset-count">{len(payload["datasets"]):,}</strong>' in home
    assert (
        f'<strong id="model-count">{payload["meta"]["configured_model_count"]:,}</strong>' in home
    )
    assert "podium-place-1" in home
    assert '<div id="elo-reference-table" class="table-wrap">' in home
    assert "Strict Macro-F1 Elo across all dataset modalities" in home
    table_rows = re.search(r'<tbody id="elo-reference-rows">(.*?)</tbody>', home, re.S)[1]
    reference_rows = sorted(
        (
            row
            for row in charts["domain_elo"]
            if row["cell"] == parsed["meta"]["reference_cell"]
            and row["domain"] == "all"
            and row["metric"] == "f1_macro"
            and row["model_id"] not in parsed["meta"]["plot_excluded_models"]
        ),
        key=lambda row: (-row["Elo"], row["model_id"]),
    )
    assert reference_rows
    assert re.findall(r'<td><a href="/models/([^/]+)/">[^<]+</a></td>', table_rows) == [
        row["model_id"].lower() for row in reference_rows
    ]
    for row in reference_rows:
        assert (
            f"<td>{row['Elo']:,}</td><td>{row['Elo_lo']:,}–{row['Elo_hi']:,}</td>"
            f"<td>{row['n_targets']:,}</td>" in table_rows
        )
    metadata = json.loads(
        re.search(r'<script type="application/ld\+json">(.*?)</script>', home, re.S)[1]
    )
    entities = {item["@type"]: item for item in metadata["@graph"]}
    assert set(entities) == {"WebSite", "WebPage"}
    assert entities["WebPage"]["isPartOf"]["@id"] == entities["WebSite"]["@id"]
    assert "mainEntity" not in entities["WebPage"]
    sqlite_url = next(row["path"] for row in parsed["raw_exports"] if row["format"] == "sqlite3")
    assert (
        re.search(r'<link rel="alternate" type="application/vnd.sqlite3"\s+href="([^"]+)"', home)[1]
        == sqlite_url
    )
    registry_html = (output / "datasets.html").read_text(encoding="utf-8")
    assert f"{len(payload['datasets'])} of {len(payload['datasets'])} datasets" in registry_html
    for dataset in payload["datasets"]:
        assert dataset["display_name"] in registry_html
    web_metadata.write_dataset_registry_summary(output)
    assert (output / "datasets.html").read_text(encoding="utf-8") == registry_html
    web_metadata.write_home_summary(output, parsed)
    assert (output / "index.html").read_text(encoding="utf-8") == home
    assert (output / "models.html").is_file()
    assert (output / "assets/plotly-cartesian.min.js").is_file()
    skill = Path(__file__).resolve().parents[1] / "skills" / web_metadata.SKILL_NAME
    assert b"license: EUPL-1.2" in (output / "skill.md").read_bytes()
    assert (output / "skill.md").read_bytes() == (skill / "SKILL.md").read_bytes()
    assert not (output / "skills").exists()
    for asset in (dashboard.PACKAGE_ROOT / "web").rglob("*"):
        if asset.is_file() and asset.name not in {
            "CITATION.bib",
            "CITATION.cff",
            "llms.txt",
            "robots.txt",
            "sitemap.xml",
            "index.html",
            "artifacts.html",
            "datasets.html",
            "models.html",
            "og.png",
        }:
            relative = asset.relative_to(dashboard.PACKAGE_ROOT / "web")
            assert (output / relative).read_bytes() == asset.read_bytes(), relative
    social_image = (output / "assets/og.png").read_bytes()
    social_version = hashlib.sha256(social_image).hexdigest()[:16]
    assert social_image != (dashboard.PACKAGE_ROOT / "web/assets/og.png").read_bytes()
    assert (output / "index.html").read_text().count(f"assets/og.png?v={social_version}") == 2
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
        "tabbench_bio.fold_metrics._sqlite_frame",
        lambda *a: pytest.fail("Unchanged fold metrics should be cached"),
    )
    assert (tmp_path / "site.cache/fold_metrics.sqlite").is_file()
    assert not list(output.rglob("*.sqlite"))
    (output / "assets/og.png").write_bytes(b"stale social card")
    stale_analysis = output / "data/analysis-adaptive-0000000000000000.json"
    stale_dataset = output / "data/datasets/0000000000000000.json"
    unrelated = output / "data/notes.json"
    for path in (stale_analysis, stale_dataset, unrelated):
        path.write_text("{}")
    dashboard.build_website(database, output, n_boot=8)
    assert (output / "assets/og.png").read_bytes() == social_image
    assert not stale_analysis.exists()
    assert not stale_dataset.exists()
    assert unrelated.exists()

    # Installed wheels must generate the same skill without a source checkout.
    installed = tmp_path / "installed/tabbench_bio/_resources/skills" / skill.name
    shutil.copytree(skill, installed)
    monkeypatch.setattr(
        web_metadata, "resource_root", lambda: tmp_path / "installed/tabbench_bio/_resources"
    )
    (output / "skill.md").write_text("stale copy", encoding="utf-8")
    web_metadata.write_agent_metadata(output, payload)
    assert not (output / "skills").exists()
    assert (output / "skill.md").read_bytes() == (skill / "SKILL.md").read_bytes()


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
    package_version = resources.__version__
    assert f"version: {package_version}\n" in (output / "CITATION.cff").read_text()
    assert f"TabBench-Bio version {package_version}}}" in (output / "CITATION.bib").read_text()
    for cell in ("cap_100_n20", "cap_100_n50"):
        report = output / "local" / cell / "overall"
        assert (report / "elo.png").is_file()
        assert not (report / "elo.svg").exists()
        assert not (report / "report.html").exists()
        table = pd.read_csv(report / "leaderboard.csv").dropna(subset=["Elo"])
        assert dict(zip(table.model_id, table.Elo)) == {
            row["model_id"]: row["Elo"] for row in payload["cell_elo"] if row["cell"] == cell
        }


def test_tuned_failure_scores_match_sqlite_and_dashboard_without_crossing_cells(tmp_path):
    root = tmp_path / "results"
    for n in (20, 50):
        cell = root / f"cap_100_n{n}"
        config = config_for_cell(
            100,
            n,
            datasets=["toy"],
            datasets_regression=["reg"],
            models=["RF", "DUMMY", "SEARCH"],
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
            tuning={"SEARCH": {"base_model": "RF"}},
        )
        repository = ResultRepository(cell, config)
        for seed in (0, 1):
            truth = pd.DataFrame({"target": [0, 0, 1, 1]}, index=range(seed * 4, seed * 4 + 4))
            for dataset in ("toy", "reg"):
                for model in config["models"]:
                    failed = model == "SEARCH"
                    prediction = (
                        truth.copy() if model == "RF" and n == 20 else truth.assign(target=0)
                    )
                    proba = (
                        pd.DataFrame({0: 1 - prediction.target, 1: prediction.target})
                        if dataset == "toy"
                        else None
                    )
                    repository.write(
                        {
                            "dataset": dataset + "_0",
                            "model": model,
                            "status": "fail" if failed else "pass",
                            "reason": "fit_error" if failed else "",
                            "n_train_samples": n,
                        },
                        seed=seed,
                        ground_truth=None if failed else truth,
                        prediction=None if failed else prediction,
                        probability=None if failed else proba,
                    )
    database = consolidate_results(root)
    checksum = hashlib.sha256(database.read_bytes()).hexdigest()
    _, views, status, _ = read_inputs(database)
    assert len(status.query("model == 'SEARCH' and status == 'fail'")) == 8
    for n in (20, 50):
        cell = f"cap_100_n{n}"
        leaderboard = Leaderboard.from_sqlite(database, cell=cell)
        for task, metric, table in (
            ("classification", "f1_macro", leaderboard._clf_metrics),
            ("regression", "rmse", leaderboard._reg_metrics),
        ):
            expected = table.query("model == 'DUMMY'").sort_values(["seed", "key"])[metric].tolist()
            assert (
                table.query("model == 'SEARCH'").sort_values(["seed", "key"])[metric].tolist()
                == expected
            )
            for name in ("strict", "adaptive"):
                rows = views[name][task].query("cell == @cell and model == 'SEARCH'")
                assert rows.sort_values(["seed", "key"])[metric].tolist() == expected
                assert rows.imputed.all()
            assert views["conditional"][task].query("model == 'SEARCH'").empty
    assert hashlib.sha256(database.read_bytes()).hexdigest() == checksum


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
    assert result["regression_cost_grid"] == []
    assert result["pooled_cost_grid"] == []
    assert result["progress"]["fraction"] == 0.5


def test_dashboard_display_precision_preserves_protocol_and_small_timings():
    source = {
        "model_id": "RF",
        "Elo": 1000,
        "n_targets": 43,
        "f1_macro": 0.123456789,
        "inference_time_s": 0.000000123456789,
        "tuning": {"grid": [{"learning_rate": 0.000123456789}]},
        "models": {"RF": {"tuning": {"grid": [{"learning_rate": 0.000123456789}]}}},
        "domain_elo_cache": {"algorithm_sha256": "0123456789"},
    }
    compact = dashboard.compact_dashboard_values(source)
    assert compact["f1_macro"] == 0.123457
    assert compact["inference_time_s"] == 0.000000123457
    assert compact["Elo"] == source["Elo"]
    assert compact["n_targets"] == source["n_targets"]
    assert compact["models"] == source["models"]
    assert compact["domain_elo_cache"] == source["domain_elo_cache"]
    assert not compact.keys() & source["models"]["RF"].keys()
    assert "tuning" not in compact
    assert "tuning" in source


def test_root_citation_matches_package_version():
    root = Path(__file__).parents[1] / "CITATION.cff"
    assert f"\nversion: {resources.__version__}\n" in root.read_text(encoding="utf-8")


@pytest.mark.parametrize("checkout", [True, False])
def test_citation_version_stamps_site_and_repository(tmp_path, monkeypatch, checkout):
    package, output = tmp_path / "src/tabbench_bio", tmp_path / "site"
    output.mkdir(parents=True)
    root = Path(__file__).parents[1] / "CITATION.cff"
    shutil.copy(root, tmp_path)
    if checkout:
        (tmp_path / "pyproject.toml").touch()
    for name in ("CITATION.cff", "CITATION.bib"):
        shutil.copy(dashboard.PACKAGE_ROOT / "web" / name, output)
    monkeypatch.setattr(dashboard, "PACKAGE_ROOT", package)
    dashboard.write_citation_version(output, "9.9.9")
    assert "\nversion: 9.9.9\n" in (output / "CITATION.cff").read_text(encoding="utf-8")
    assert "{TabBench-Bio version 9.9.9}\n}" in (output / "CITATION.bib").read_text()
    stamped = "\nversion: 9.9.9\n" in (tmp_path / "CITATION.cff").read_text(encoding="utf-8")
    assert stamped == checkout
