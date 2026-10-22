import json
import sqlite3
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from tabbench_bio import doctor
from tabbench_bio.bio.cache import save_cached_raw
from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.fingerprint import spec_fingerprint
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.model_registry import MODEL_REGISTRY


def test_doctor_reports_stale_raw_and_processed_caches_without_writing(tmp_path, monkeypatch):
    spec = BioDatasetSpec("toy", "local", "x.csv", target="y", problem_type="binary")
    monkeypatch.setattr(doctor, "load_specs", lambda: {"toy": spec})
    fingerprint = spec_fingerprint(spec)
    raw = BioRawDataset(
        "toy",
        pd.DataFrame({"x": [1]}),
        pd.Series([0]),
        "binary",
        "",
        "",
        "",
        {"spec_sha256": fingerprint},
    )
    raw_path = save_cached_raw(tmp_path / "bio", raw)
    split = tmp_path / "datasets_processed/seed_0_hash/split_params.json"
    split.parent.mkdir(parents=True)
    split.write_text("{}")
    dataset_metadata = split.parent / fingerprint / "dataset_spec.json"
    dataset_metadata.parent.mkdir()
    dataset_metadata.write_text(json.dumps({"bio_id": "toy", "spec_sha256": fingerprint}))
    before = {p: p.read_bytes() for p in (raw_path, split, dataset_metadata)}
    assert doctor.check_cache(tmp_path) == []
    spec = replace(spec, target="changed")
    problems = doctor.check_cache(tmp_path)
    assert any("stale dataset cache" in issue for issue in problems)
    assert any("stale split specification" in issue for issue in problems)
    assert before == {p: p.read_bytes() for p in before}


def test_skill_copy_license_drift_is_reported(tmp_path):
    canonical = tmp_path / "skills" / doctor.SKILL_NAME / "SKILL.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("---\nlicense: EUPL-1.2\n---\n")
    site = tmp_path / "site"
    site.mkdir()
    (site / "skill.md").write_text("license: Apache-2.0\n")
    problems = doctor.check_skill(tmp_path, site)
    assert any("Stale skill copy" in issue for issue in problems)
    assert not any("Missing generated skill asset" in issue for issue in problems)
    assert "Apache" in (site / "skill.md").read_text()
    (site / "skill.md").write_bytes(canonical.read_bytes())
    assert doctor.check_skill(tmp_path, site) == []
    (site / "skill.md").unlink()
    assert any(
        "Missing generated skill asset" in issue for issue in doctor.check_skill(tmp_path, site)
    )


def test_missing_profile_and_adapter_are_reported(tmp_path, monkeypatch):
    spec = replace(MODEL_REGISTRY["RF"], adapter="tabbench_bio.models.absent:Model")
    monkeypatch.setattr(doctor, "MODEL_REGISTRY", {"RF": spec})
    problems = doctor.check_metadata(tmp_path)
    assert any("missing environment profile" in issue for issue in problems)
    assert any("missing adapter module" in issue for issue in problems)


def test_import_probe_uses_the_model_environment(monkeypatch):
    monkeypatch.setattr(doctor, "model_python", lambda profile: f"/{profile}/python")

    def probe(command, **kwargs):
        assert command[0] == "/tabpfn35/python"
        assert command[-1] == "TABPFN-V3.5"
        assert "_resolve_hyperparameters" in command[2]
        return SimpleNamespace(returncode=1, stderr="ImportError: missing dependency")

    monkeypatch.setattr(doctor.subprocess, "run", probe)
    assert "missing dependency" in doctor.check_adapter("TABPFNV35")[0]


@pytest.mark.parametrize("problems,code", [([], 0), (["stale cache"], 1)])
def test_command_status_reflects_failures(tmp_path, monkeypatch, capsys, problems, code):
    (tmp_path / "configs/models").mkdir(parents=True)
    monkeypatch.setattr(doctor, "CHECKOUT", tmp_path)
    monkeypatch.setattr(doctor, "check_metadata", lambda _: [])
    monkeypatch.setattr(doctor, "check_skill", lambda *a: [])
    monkeypatch.setattr(doctor, "check_cache", lambda _: problems)
    with pytest.raises(SystemExit) as error:
        doctor.cmd_doctor(Namespace(cache_dir=tmp_path, site_dir=None, model=[], results=None))
    assert error.value.code == code
    assert "Adapter imports not tested" in capsys.readouterr().out


def test_cli_help_does_not_import_doctor(monkeypatch):
    import sys

    from tabbench_bio.cli import main

    monkeypatch.setitem(sys.modules, "tabbench_bio.doctor", None)
    monkeypatch.setattr(sys, "argv", ["tabbench-bio", "--help"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0


@pytest.fixture
def result_database(tmp_path):
    path = tmp_path / "results.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cells (cell TEXT, config_json TEXT)")
        connection.execute(
            "CREATE TABLE attempts (cell TEXT, seed INTEGER, dataset TEXT, model TEXT, status TEXT, "
            "reason TEXT DEFAULT '', timestamp TEXT DEFAULT '2026-01-01', attempt_id INTEGER PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO cells VALUES (?, ?)",
            [
                (
                    "a",
                    json.dumps(
                        {
                            "models": ["DUMMY", "TUNED"],
                            "datasets_classification": ["toy"],
                            "datasets_regression": [],
                            "cv_folds": 2,
                            "n_repetitions": 1,
                            "random_state": 42,
                        }
                    ),
                ),
                (
                    "b",
                    json.dumps(
                        {
                            "models": ["DUMMY"],
                            "datasets_classification": [],
                            "datasets_regression": ["other"],
                            "cv_folds": None,
                            "n_repetitions": None,
                            "random_state": 2,
                        }
                    ),
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO attempts (cell,seed,dataset,model,status) VALUES (?, ?, ?, ?, ?)",
            [
                ("a", 0, "toy_0", "DUMMY", "pass"),
                ("a", 1, "toy_0", "DUMMY", "pass"),
                ("a", 0, "toy_0", "TUNED", "fail"),
                ("a", 1, "toy_0", "TUNED", "skip"),
                ("b", 2, "other_0", "DUMMY", "pass"),
            ],
        )
    return path


def test_result_presence_accepts_all_outcomes_and_keeps_cells_separate(result_database):
    before = result_database.read_bytes()
    assert doctor.check_results(result_database) == []
    assert result_database.read_bytes() == before


@pytest.mark.parametrize("delete_all, missing", [(False, 1), (True, 2)])
def test_results_find_missing_folds_and_entire_model(result_database, delete_all, missing):
    with sqlite3.connect(result_database) as connection:
        connection.execute(
            "DELETE FROM attempts WHERE model = 'TUNED'" + ("" if delete_all else " AND seed = 1")
        )
    problems = doctor.check_results(result_database)
    assert len(problems) == 1
    assert f"a: {missing} missing split coordinates" in problems[0]
    assert "(1, 'toy_0', 'TUNED')" in problems[0]


def test_results_report_retries_as_duplicates_without_modifying_database(result_database):
    with sqlite3.connect(result_database) as connection:
        connection.execute(
            "INSERT INTO attempts (cell,seed,dataset,model,status) VALUES ('a', 0, 'toy_0', 'TUNED', 'pass')"
        )
    before = result_database.read_bytes()
    problems = doctor.check_results(result_database)
    assert len(problems) == 1
    assert "1 duplicate split coordinates (1 extra attempts)" in problems[0]
    assert result_database.read_bytes() == before


def test_results_missing_file_is_not_created(tmp_path):
    path = tmp_path / "absent.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        doctor.check_results(path)
    assert not path.exists()


@pytest.mark.parametrize("delete_all", [False, True])
def test_results_find_splits_absent_for_every_model(result_database, delete_all):
    with sqlite3.connect(result_database) as connection:
        connection.execute(
            "DELETE FROM attempts WHERE cell = 'a'" + ("" if delete_all else " AND seed = 1")
        )
    problems = doctor.check_results(result_database)
    assert len(problems) == 1
    assert f"a: {4 if delete_all else 2} missing split coordinates" in problems[0]


def test_results_find_configured_dataset_with_no_outcomes(result_database):
    with sqlite3.connect(result_database) as connection:
        config = json.loads(
            connection.execute("SELECT config_json FROM cells WHERE cell = 'a'").fetchone()[0]
        )
        config["datasets_classification"].append("never_run")
        connection.execute(
            "UPDATE cells SET config_json = ? WHERE cell = 'a'", (json.dumps(config),)
        )
    problems = doctor.check_results(result_database)
    assert len(problems) == 1
    assert "a: 4 missing split coordinates" in problems[0]
    assert "never_run_0" in problems[0]


def test_results_unknown_status_does_not_count_as_present(result_database):
    with sqlite3.connect(result_database) as connection:
        connection.execute("UPDATE attempts SET status = 'pending' WHERE model = 'TUNED'")
    assert "a: 2 missing split coordinates" in doctor.check_results(result_database)[0]


@pytest.mark.parametrize("status", ["fail", "skip"])
def test_doctor_flags_non_skipping_models_when_dummy_does_not_pass(result_database, status):
    with sqlite3.connect(result_database) as connection:
        connection.execute(
            "UPDATE attempts SET status=? WHERE model='DUMMY' AND cell='a'", (status,)
        )
    assert any(
        "DUMMY eligibility violations" in problem
        for problem in doctor.check_results(result_database)
    )


def test_doctor_cli_results_check_exits_nonzero(result_database, monkeypatch, capsys):
    from tabbench_bio.cli import main

    with sqlite3.connect(result_database) as connection:
        connection.execute("DELETE FROM attempts WHERE model = 'TUNED'")
    monkeypatch.setattr(doctor, "check_metadata", lambda _: [])
    monkeypatch.setattr(doctor, "check_skill", lambda *a: [])
    monkeypatch.setattr(doctor, "check_cache", lambda _: [])
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "doctor", "--results", str(result_database)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    assert "FAIL Result split coverage" in capsys.readouterr().out
