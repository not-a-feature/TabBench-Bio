"""Recoverably add one GPU-solo model to an already-frozen feature-sweep grid."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

DEFAULT_MODEL = "TABPFN-V3"
TIER_FILES = ("config_gpu_solo.json", "config_gpu_shared.json", "config_cpu.json")
RUNTIME_CONFIG_KEYS = frozenset(
    {
        "output_dir",
        "cache_dir",
        "dataset_names_classification",
        "dataset_names_regression",
    }
)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_partition(cell: Path) -> tuple[dict, dict[str, dict]]:
    full = _read(cell / "config.json")
    tiers = {name: _read(cell / name) for name in TIER_FILES if (cell / name).is_file()}
    tier_models = [model for tier in tiers.values() for model in tier["models"]]
    assert len(tier_models) == len(set(tier_models)), f"Duplicate tier model in {cell}"
    assert set(tier_models) == set(full["models"]), f"Tier partition mismatch in {cell}"
    for tier in tiers.values():
        for key in ("bio_max_features", "train_subsample", "output_dir"):
            assert tier[key] == full[key], f"{key} mismatch in {cell}"
    assert "config_gpu_solo.json" in tiers, f"Missing GPU-solo tier in {cell}"
    return full, tiers


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_value(config: dict) -> tuple[str, str]:
    normalised = {key: value for key, value in config.items() if key not in RUNTIME_CONFIG_KEYS}
    config_json = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _pre_extension_config(config: dict, model: str) -> dict:
    previous = copy.deepcopy(config)
    assert previous["models"].count(model) == 1
    previous["models"].remove(model)
    return previous


def _verified_previous_config(cell: Path, backup_model: str, current: dict, model: str) -> dict:
    expected = _pre_extension_config(current, model)
    matches: list[tuple[Path, dict]] = []
    pattern = f"pre_{backup_model}_*"
    for backup in sorted((cell / "immutable_backups").glob(pattern)):
        config_path = backup / "config.json"
        manifest_path = backup / "sha256_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            continue
        manifest = _read(manifest_path)
        assert manifest.get("config.json") == _checksum(config_path), (
            f"Backup checksum mismatch in {backup}"
        )
        candidate = _read(config_path)
        if candidate == expected:
            matches.append((backup, candidate))
    assert matches, f"No verified pre-{model} config backup matches {cell / 'config.json'}"
    return matches[-1][1]


def _registry_paths(root: Path) -> list[Path]:
    paths = []
    canonical = root / "results.sqlite"
    if canonical.is_file():
        paths.append(canonical)
    writers = root / "writers"
    if writers.is_dir():
        paths.extend(sorted(writers.glob("*.sqlite")))
    return paths


def _table_count(connection: sqlite3.Connection, table: str) -> int | None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if exists is None:
        return None
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _migrate_registries(
    root: Path,
    timestamp: str,
    model: str,
    backup_model: str,
    cell_configs: dict[str, tuple[dict, dict]],
) -> tuple[int, int]:
    inspections: dict[Path, dict] = {}
    rows_to_change = 0
    rows_current = 0

    # Validate every bundle before writing any of them. Only the cells table may change;
    # attempt and blob counts are guarded again inside each write transaction.
    for path in _registry_paths(root):
        with sqlite3.connect(path, timeout=60) as connection:
            connection.execute("PRAGMA busy_timeout=60000")
            cells_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cells'"
            ).fetchone()
            if cells_table is None:
                continue
            bundle_rows: dict[str, tuple[str, str]] = {}
            bundle_updates: dict[str, tuple[str, str, str, str]] = {}
            for cell, (previous, current) in cell_configs.items():
                row = connection.execute(
                    "SELECT config_json, config_sha256 FROM cells WHERE cell = ?", (cell,)
                ).fetchone()
                if row is None:
                    continue
                previous_value = _registry_value(previous)
                current_value = _registry_value(current)
                assert row in (previous_value, current_value), (
                    f"Unexpected registered config for {cell} in {path}"
                )
                bundle_rows[cell] = row
                if row == previous_value:
                    bundle_updates[cell] = (*previous_value, *current_value)
                    rows_to_change += 1
                else:
                    rows_current += 1
            inspections[path] = {
                "rows": bundle_rows,
                "updates": bundle_updates,
                "attempts": _table_count(connection, "attempts"),
                "blobs": _table_count(connection, "blobs"),
            }

    if rows_to_change:
        backup = root / "immutable_backups" / f"pre_{backup_model}_registry_{timestamp}"
        assert not backup.exists(), f"Backup already exists: {backup}"
        backup.mkdir(parents=True)
        registry_manifest = {
            "model": model,
            "bundles": {
                path.relative_to(root).as_posix(): {
                    cell: {"config_json": row[0], "config_sha256": row[1]}
                    for cell, row in inspection["rows"].items()
                }
                for path, inspection in inspections.items()
            },
        }
        manifest_path = backup / "registry_rows.json"
        _write_atomic(manifest_path, registry_manifest)
        manifest_path.chmod(0o440)

    for path, inspection in inspections.items():
        if not inspection["updates"]:
            continue
        with sqlite3.connect(path, timeout=60) as connection:
            connection.execute("PRAGMA busy_timeout=60000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            before = {
                "attempts": _table_count(connection, "attempts"),
                "blobs": _table_count(connection, "blobs"),
            }
            assert before == {
                "attempts": inspection["attempts"],
                "blobs": inspection["blobs"],
            }, f"Result counts changed before registry migration in {path}"
            for cell, (old_json, old_hash, new_json, new_hash) in inspection["updates"].items():
                cursor = connection.execute(
                    "UPDATE cells SET config_json = ?, config_sha256 = ? "
                    "WHERE cell = ? AND config_json = ? AND config_sha256 = ?",
                    (new_json, new_hash, cell, old_json, old_hash),
                )
                assert cursor.rowcount == 1, f"Concurrent registry change for {cell} in {path}"
            after = {
                "attempts": _table_count(connection, "attempts"),
                "blobs": _table_count(connection, "blobs"),
            }
            assert after == before, f"Result rows changed during registry migration in {path}"
            connection.commit()

    # A prior interruption may have committed only some bundles. This verification plus
    # idempotent old-or-current handling makes the next invocation safely finish the rest.
    for path in inspections:
        with sqlite3.connect(path, timeout=60) as connection:
            for cell, (_, current) in cell_configs.items():
                row = connection.execute(
                    "SELECT config_json, config_sha256 FROM cells WHERE cell = ?", (cell,)
                ).fetchone()
                if row is not None:
                    assert row == _registry_value(current), (
                        f"Registry migration incomplete for {cell} in {path}"
                    )
    return rows_to_change, rows_current


def extend(root: Path, timestamp: str, model: str = DEFAULT_MODEL) -> None:
    assert model and all(character.isalnum() or character in "-_" for character in model)
    backup_model = (
        "tabpfn_v3_ferranti" if model == DEFAULT_MODEL else model.lower().replace("-", "_")
    )
    cells = sorted(path.parent for path in root.glob("cap_*/config.json"))
    assert cells, f"No frozen cells found below {root}"
    changed = 0
    already_extended = 0
    cell_configs: dict[str, tuple[dict, dict]] = {}

    for cell in cells:
        full, tiers = _validate_partition(cell)
        full_count = full["models"].count(model)
        solo_count = tiers["config_gpu_solo.json"]["models"].count(model)
        other_count = sum(
            tier["models"].count(model)
            for name, tier in tiers.items()
            if name != "config_gpu_solo.json"
        )
        assert full_count in (0, 1), f"Invalid {model} count in {cell / 'config.json'}"
        if full_count == 1:
            assert solo_count == 1 and other_count == 0, f"Invalid existing {model} tier in {cell}"
            previous_full = _verified_previous_config(cell, backup_model, full, model)
            cell_configs[cell.name] = (previous_full, full)
            already_extended += 1
            continue
        assert solo_count == 0 and other_count == 0, (
            f"Tier contains absent full-roster model in {cell}"
        )
        previous_full = copy.deepcopy(full)

        config_paths = [cell / "config.json", *(cell / name for name in tiers)]
        backup = cell / "immutable_backups" / f"pre_{backup_model}_{timestamp}"
        assert not backup.exists(), f"Backup already exists: {backup}"
        backup.mkdir(parents=True)
        manifest: dict[str, str] = {}
        for path in config_paths:
            destination = backup / path.name
            shutil.copy2(path, destination)
            destination.chmod(0o440)
            manifest[path.name] = _checksum(destination)
        manifest_path = backup / "sha256_manifest.json"
        _write_atomic(manifest_path, manifest)
        manifest_path.chmod(0o440)

        full["models"].append(model)
        tiers["config_gpu_solo.json"]["models"].append(model)
        _write_atomic(cell / "config.json", full)
        for name, tier in tiers.items():
            _write_atomic(cell / name, tier)

        updated_full, updated_tiers = _validate_partition(cell)
        assert updated_full["models"].count(model) == 1
        assert updated_tiers["config_gpu_solo.json"]["models"].count(model) == 1
        cell_configs[cell.name] = (previous_full, updated_full)
        changed += 1

    registry_changed, registry_current = _migrate_registries(
        root, timestamp, model, backup_model, cell_configs
    )
    print(
        f"cells={len(cells)} changed={changed} already_extended={already_extended} "
        f"registry_changed={registry_changed} registry_current={registry_current}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    extend(args.root, args.timestamp, args.model)


if __name__ == "__main__":
    main()
