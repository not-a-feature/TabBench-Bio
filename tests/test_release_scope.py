import hashlib
import json
import sqlite3

import pytest

from scripts.clean_release_scope import clean_release_scope


@pytest.mark.parametrize("record_disabled", [False, True])
def test_scope_removes_only_disabled_unrecorded_tasks_and_preserves_source(
    tmp_path, record_disabled
):
    source, output, registry = (
        tmp_path / name for name in ("source.sqlite", "output.sqlite", "registry.json")
    )
    registry.write_text(
        json.dumps(
            [
                {"bio_id": "active", "enabled": True},
                {"bio_id": "disabled", "enabled": False},
                {"bio_id": "not_run", "enabled": True},
            ]
        )
    )
    config = {
        "datasets_classification": ["active", "disabled", "not_run"],
        "datasets_regression": [],
        "models": ["DUMMY"],
        "budget": 3600,
    }
    text = json.dumps(config)
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE cells (cell TEXT, config_json TEXT, config_sha256 TEXT)")
        connection.execute(
            "INSERT INTO cells VALUES ('cell', ?, ?)",
            (text, hashlib.sha256(text.encode()).hexdigest()),
        )
        connection.execute("CREATE TABLE attempts (dataset TEXT, status TEXT)")
        connection.executemany(
            "INSERT INTO attempts VALUES (?, ?)",
            [("active_0", "pass"), ("active_0", "fail"), ("active_0", "skip")],
        )
        if record_disabled:
            connection.execute("INSERT INTO attempts VALUES ('disabled_0', 'fail')")
        connection.execute("CREATE TABLE blobs (payload BLOB)")
        connection.execute("INSERT INTO blobs VALUES (?)", (b"unchanged prediction",))
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    before = source.read_bytes()
    if record_disabled:
        with pytest.raises(AssertionError, match="recorded outcomes"):
            clean_release_scope(source, output, registry)
        assert not output.exists()
    else:
        report = clean_release_scope(source, output, registry)
        assert report["removed_datasets"] == ["disabled"]
        assert report["attempts_and_blobs_unchanged"]
        with sqlite3.connect(output) as connection:
            text, digest = connection.execute(
                "SELECT config_json,config_sha256 FROM cells"
            ).fetchone()
            config["datasets_classification"].remove("disabled")
            assert json.loads(text) == config
            assert hashlib.sha256(text.encode()).hexdigest() == digest
        assert json.loads(output.with_suffix(".scope.json").read_text())["changed_cells"]["cell"][
            "original_config"
        ]["datasets_classification"] == ["active", "disabled", "not_run"]
    assert source.read_bytes() == before
