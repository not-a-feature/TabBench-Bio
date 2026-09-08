import json
import sqlite3

from scripts.extend_frozen_cells_tabpfn_v3 import _registry_value, extend

MODEL = "TABPFN-WIDE-5K-NE3"
CELL = "cap_10000_n100"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_cell(root):
    cell = root / CELL
    cell.mkdir(parents=True)
    base = {
        "models": ["DUMMY"],
        "bio_max_features": 10_000,
        "train_subsample": 100,
        "output_dir": f"results/test/{CELL}",
        "cache_dir": ".cache/test",
        "dataset_names_classification": ["OpenML-1138"],
        "dataset_names_regression": [],
    }
    _write(cell / "config.json", base)
    for name, models in {
        "config_gpu_solo.json": [],
        "config_gpu_shared.json": [],
        "config_cpu.json": ["DUMMY"],
    }.items():
        tier = dict(base)
        tier["models"] = models
        _write(cell / name, tier)
    return base


def _make_bundle(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json, config_hash = _registry_value(config)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE cells (
                cell TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                config_sha256 TEXT NOT NULL
            );
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                cell TEXT NOT NULL REFERENCES cells(cell)
            );
            CREATE TABLE blobs (sha256 TEXT PRIMARY KEY);
            """
        )
        connection.execute("INSERT INTO cells VALUES (?, ?, ?)", (CELL, config_json, config_hash))
        connection.execute("INSERT INTO attempts VALUES ('kept-attempt', ?)", (CELL,))


def _assert_bundle_current(path, current):
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT config_json, config_sha256 FROM cells WHERE cell = ?", (CELL,)
        ).fetchone() == _registry_value(current)
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (1,)


def test_extends_configs_and_all_live_registries_without_changing_attempts(tmp_path):
    root = tmp_path / "results"
    previous = _make_cell(root)
    bundles = [root / "results.sqlite", root / "writers" / "spock.sqlite"]
    for bundle in bundles:
        _make_bundle(bundle, previous)

    extend(root, "FIRST", MODEL)

    current = json.loads((root / CELL / "config.json").read_text(encoding="utf-8"))
    assert current["models"] == ["DUMMY", MODEL]
    for bundle in bundles:
        _assert_bundle_current(bundle, current)
    registry_backup = root / "immutable_backups" / "pre_tabpfn_wide_5k_ne3_registry_FIRST"
    assert (registry_backup / "registry_rows.json").is_file()

    extend(root, "SECOND", MODEL)
    assert not (root / "immutable_backups" / "pre_tabpfn_wide_5k_ne3_registry_SECOND").exists()


def test_repairs_registry_after_config_only_interruption(tmp_path):
    root = tmp_path / "results"
    previous = _make_cell(root)

    extend(root, "CONFIG_ONLY", MODEL)
    current = json.loads((root / CELL / "config.json").read_text(encoding="utf-8"))
    bundle = root / "writers" / "spock.sqlite"
    _make_bundle(bundle, previous)

    extend(root, "RECOVERY", MODEL)

    _assert_bundle_current(bundle, current)
    assert (
        root
        / "immutable_backups"
        / "pre_tabpfn_wide_5k_ne3_registry_RECOVERY"
        / "registry_rows.json"
    ).is_file()
