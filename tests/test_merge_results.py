"""Merge independent model runs without changing published results."""

import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from tabbench_bio.cli import main
from tabbench_bio.config import config_for_cell
from tabbench_bio.leaderboard import Leaderboard
from tabbench_bio.result_store import (
    ResultRepository,
    _canonical_json,
    _sha256,
    consolidate_results,
    merge_results,
)


def bundle(root, model, *, truth_values=(0, 1, 0, 1), correct=True, **settings):
    config = {
        "models": [model],
        "model_limits": {},
        "model_overrides": {},
        "cv_folds": 2,
        "n_repetitions": 1,
        "random_state": 42,
        "bio_max_features": 10000,
        "train_subsample": 100,
        "datasets_classification": ["toy", "other"],
        "datasets_regression": [],
        **settings,
    }
    repository = ResultRepository(root / "cap_10000_n100", config)
    for seed in range(2):
        truth = pd.DataFrame({"target": truth_values}, index=range(seed * 4, seed * 4 + 4))
        prediction = truth.copy() if correct else truth.assign(target=0)
        probability = pd.DataFrame({0: [0.5] * 4, 1: [0.5] * 4}, index=truth.index)
        repository.write(
            {"dataset": "toy_0", "model": model, "status": "pass", "reason": ""},
            seed=seed,
            prediction=prediction,
            probability=probability,
            ground_truth=truth,
        )
    repository.write(
        {"dataset": "other_0", "model": model, "status": "fail", "reason": "fit_error"},
        seed=0,
    )
    return repository.writer_path


def test_cli_merge_preserves_inputs_deduplicates_and_ranks(tmp_path, monkeypatch):
    first = bundle(tmp_path / "release", "RF", correct=False)
    second = bundle(tmp_path / "new-model", "MY")
    canonical = consolidate_results(first.parent.parent)
    sources = [first, second, canonical]
    before = {path: path.read_bytes() for path in sources}
    output = tmp_path / "merged" / "results.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tabbench-bio",
            "results",
            "merge",
            "--source",
            str(first.parent.parent),
            "--source",
            str(second),
            "--source",
            str(second),
            "--output",
            str(output),
        ],
    )
    main()
    assert all(path.read_bytes() == before[path] for path in sources)
    repository = ResultRepository.from_root(output.parent)
    assert len(repository.attempts()) == 6
    assert sum(a.status == "fail" for a in repository.attempts()) == 2
    with closing(sqlite3.connect(output)) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM cells").fetchone()[0])
        assert config["models"] == ["RF", "MY"]
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        provenance = json.loads(metadata["merge_sources"])
        assert {p["metadata"]["experiment_id"] for p in provenance} == {"release", "new-model"}
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    leaderboard = Leaderboard.from_sqlite(output).rank("classification").set_index("model_id")
    assert leaderboard.loc["RF", "Elo"] == 1000
    assert leaderboard.loc["MY", "Elo"] > 1000
    repeated = merge_results([output, second], tmp_path / "again" / "results.sqlite")
    assert len(ResultRepository.from_root(repeated.parent).attempts()) == 6
    assert not list(output.parent.glob(".merge-results-*"))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"bio_max_features": 2000}, "bio_max_features"),
        ({"random_state": 7}, "random_state"),
        ({"train_subsample": 200}, "train_subsample"),
        ({"truth_values": (1, 0, 0, 1)}, "Conflicting held-out-target"),
    ],
)
def test_incompatible_runs_leave_no_output(tmp_path, change, message):
    first = bundle(tmp_path / "release", "RF")
    second = bundle(tmp_path / "new-model", "MY", **change)
    output = tmp_path / "merged.sqlite"
    with pytest.raises(AssertionError, match=message):
        merge_results([first, second], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".merge-results-*"))


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"correct": False}, "Divergent passing results"),
        ({"model_overrides": {"RF": {"ensemble": True}}}, "model_overrides"),
        ({"model_limits": {"RF": {"max_cells": 100}}}, "model_limits"),
    ],
)
def test_same_model_conflicts_are_rejected(tmp_path, settings, message):
    first = bundle(tmp_path / "release", "RF")
    second = bundle(tmp_path / "repeat", "RF", **settings)
    with pytest.raises(AssertionError, match=message):
        merge_results([first, second], tmp_path / "merged.sqlite")


@pytest.mark.parametrize("corruption", ["blob", "config", "foreign_key", "attempt_id"])
def test_corrupt_source_is_rejected(tmp_path, corruption):
    first = bundle(tmp_path / "release", "RF")
    second = bundle(tmp_path / "new-model", "MY")
    with closing(sqlite3.connect(first)) as connection:
        attempt_id = connection.execute("SELECT attempt_id FROM attempts LIMIT 1").fetchone()[0]
    with closing(sqlite3.connect(second)) as connection:
        if corruption == "blob":
            connection.execute("UPDATE blobs SET uncompressed_bytes = 1")
        elif corruption == "config":
            connection.execute("UPDATE cells SET config_sha256 = 'wrong'")
        elif corruption == "foreign_key":
            connection.execute("UPDATE attempts SET prediction_sha256 = 'missing'")
        else:
            connection.execute("UPDATE attempts SET attempt_id = ? WHERE rowid = 1", (attempt_id,))
        connection.commit()
    output = tmp_path / "merged.sqlite"
    with pytest.raises(AssertionError):
        merge_results([first, second], output)
    assert not output.exists()


def test_output_is_never_overwritten_and_empty_root_is_rejected(tmp_path):
    source = bundle(tmp_path / "run", "RF")
    before = source.read_bytes()
    with pytest.raises(AssertionError, match="Output already exists"):
        merge_results([source], source)
    assert source.read_bytes() == before
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AssertionError, match="No result bundles"):
        merge_results([empty], tmp_path / "new.sqlite")


def test_new_cell_and_model_settings_are_retained(tmp_path):
    first = bundle(tmp_path / "release", "RF")
    second = bundle(tmp_path / "new-model", "MY", model_overrides={"MY": {"ensemble": True}})
    with closing(sqlite3.connect(second)) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM cells").fetchone()[0])
        config["train_subsample"] = 200
        payload = _canonical_json(config)
        connection.execute(
            "INSERT INTO cells VALUES (?, ?, ?)",
            (
                "cap_10000_n200",
                payload,
                _sha256(payload.encode()),
            ),
        )
        connection.commit()
    output = merge_results([first, second], tmp_path / "merged.sqlite")
    assert Leaderboard.sqlite_cells(output) == ["cap_10000_n100", "cap_10000_n200"]
    with closing(sqlite3.connect(output)) as connection:
        config = json.loads(
            connection.execute(
                "SELECT config_json FROM cells WHERE cell = 'cap_10000_n100'"
            ).fetchone()[0]
        )
        assert config["model_overrides"] == {"MY": {"ensemble": True}}


def test_dataset_subsets_are_combined_but_task_changes_are_rejected(tmp_path):
    first = bundle(tmp_path / "release", "RF", datasets_classification=["toy", "retired"])
    second = bundle(tmp_path / "new-model", "MY")
    output = merge_results([first, second], tmp_path / "merged.sqlite")
    with closing(sqlite3.connect(output)) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM cells").fetchone()[0])
        assert config["datasets_classification"] == ["toy", "retired", "other"]
    different_task = bundle(
        tmp_path / "wrong-task", "MY", datasets_classification=[], datasets_regression=["toy"]
    )
    with pytest.raises(AssertionError, match="task types conflict"):
        merge_results([first, different_task], tmp_path / "wrong.sqlite")


def test_short_merge_and_default_leaderboard_generate_reports(tmp_path, monkeypatch):
    first = bundle(tmp_path / "release", "RF", correct=False)
    second = bundle(tmp_path / "new-model", "MY")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "merge", str(first), str(second)])
    main()
    website = tmp_path / "website"
    website.mkdir()
    (website / ".git").write_text("gitdir: ../.git/modules/website\n")
    (website / "CNAME").write_text("tabbench-bio.eu\n")
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "leaderboard", "--bootstrap-rounds", "8"])
    main()
    root = website / "local/cap_10000_n100/overall"
    for name in ("elo.png", "leaderboard.csv"):
        assert (root / name).stat().st_size > 0
    assert not list((website / "local").rglob("*.svg"))
    assert not list((website / "local").rglob("*.html"))
    assert "/local/" in (website / ".gitignore").read_text().splitlines()
    assert (website / ".git").read_text() == "gitdir: ../.git/modules/website\n"
    assert (website / "CNAME").read_text() == "tabbench-bio.eu\n"
    assert (tmp_path / "website.cache/fold_metrics.sqlite").is_file()
    assert not (tmp_path / "results/merged/leaderboard").exists()
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "merge", str(first), str(second)])
    with pytest.raises(AssertionError, match="Output already exists"):
        main()


def test_merged_pipeline_generates_metrics_and_plot(tmp_path, monkeypatch):
    sources = []
    for model in ("RF", "MY"):
        cell = tmp_path / model / "cap_10000_n100"
        config = config_for_cell(
            10000,
            100,
            datasets=["OpenML-1083"],
            datasets_regression=["OpenML-46983"],
            models=[model],
            limits={},
            overrides={},
            n_rep=1,
            cv_folds=2,
            time_limit=3600,
            out_dir=str(cell),
            cache_dir=str(tmp_path / "cache"),
            test_size=0.2,
            random_state=42,
            min_samples_per_class=10,
        )
        repository = ResultRepository(cell, config)
        for seed in range(2):
            for dataset, values in (
                ("OpenML-1083_0", [0, 1, 0, 1]),
                ("OpenML-46983_0", [0.1, 0.2, 0.3, 0.4]),
            ):
                truth = pd.DataFrame({"target": values}, index=range(seed * 4, seed * 4 + 4))
                prediction = truth.copy() if model == "MY" else truth.assign(target=0)
                probability = (
                    pd.DataFrame({0: [0.5] * 4, 1: [0.5] * 4}, index=truth.index)
                    if dataset == "OpenML-1083_0"
                    else None
                )
                repository.write(
                    {
                        "dataset": dataset,
                        "model": model,
                        "status": "pass",
                        "reason": "",
                        "train_time_s": 1.0,
                        "train_peak_memory_mb": 10.0,
                        "inference_time_per_sample_ms": 0.1,
                    },
                    seed=seed,
                    prediction=prediction,
                    ground_truth=truth,
                    probability=probability,
                )
        sources.append(repository.writer_path)
    output = merge_results(sources, tmp_path / "combined" / "results.sqlite")
    cell = output.parent / "cap_10000_n100"
    cell.mkdir()
    with closing(sqlite3.connect(output)) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM cells").fetchone()[0])
    config.update(output_dir=str(cell), cache_dir=str(tmp_path / "cache"))
    config_path = cell / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    commands = [
        ["run", "--config", str(config_path), "--step", "metrics"],
        [
            "leaderboard",
            "--sqlite",
            str(output),
            "--cell",
            cell.name,
            "--plot",
            "--plot-path",
            str(tmp_path / "elo.png"),
            "--csv",
            str(tmp_path / "leaderboard.csv"),
        ],
    ]
    for args in commands:
        monkeypatch.setattr("sys.argv", ["tabbench-bio", *args])
        main()
    for path in (tmp_path / "elo.png", tmp_path / "leaderboard.csv"):
        assert path.stat().st_size > 0
