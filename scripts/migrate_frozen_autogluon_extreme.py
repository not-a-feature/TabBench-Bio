"""Recoverably migrate frozen AutoGluon reference configs to native Extreme."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

MODEL = "AUTOGLUON"
TARGET_CELLS = ("cap_10000_n100", "cap_full")
CONFIG_FILES = (
    "config.json",
    "config_gpu_solo.json",
    "config_gpu_shared.json",
    "config_cpu.json",
)
OLD_PRESET = "best_quality"
NEW_PRESET = "extreme"
TIME_LIMIT_SECONDS = 3600


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


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate(root: Path, timestamp: str) -> None:
    changed_cells = 0
    changed_files = 0

    for cell_name in TARGET_CELLS:
        cell = root / cell_name
        assert (cell / "config.json").is_file(), f"Missing frozen cell: {cell}"
        paths = [cell / name for name in CONFIG_FILES if (cell / name).is_file()]
        payloads = {path: _read(path) for path in paths}

        for payload in payloads.values():
            assert payload["autogluon_time_limit"] == TIME_LIMIT_SECONDS, payload[
                "autogluon_time_limit"
            ]
            override = payload["model_overrides"][MODEL]
            assert override["ensemble"] is True, override
            assert override["presets"] in (OLD_PRESET, NEW_PRESET), override

        to_change = [
            path
            for path, payload in payloads.items()
            if payload["model_overrides"][MODEL]["presets"] == OLD_PRESET
        ]
        if not to_change:
            print(f"{cell_name}: already={NEW_PRESET} files={len(paths)}")
            continue

        backup = cell / "immutable_backups" / f"pre_autogluon_extreme_{timestamp}"
        assert not backup.exists(), f"Backup already exists: {backup}"
        backup.mkdir(parents=True)
        manifest: dict[str, str] = {}
        for path in paths:
            destination = backup / path.name
            shutil.copy2(path, destination)
            destination.chmod(0o440)
            manifest[path.name] = _checksum(destination)
        manifest_path = backup / "sha256_manifest.json"
        _write_atomic(manifest_path, manifest)
        manifest_path.chmod(0o440)

        for path in to_change:
            payloads[path]["model_overrides"][MODEL]["presets"] = NEW_PRESET
            _write_atomic(path, payloads[path])
            changed_files += 1

        for path in paths:
            updated = _read(path)
            assert updated["model_overrides"][MODEL]["presets"] == NEW_PRESET
            assert updated["autogluon_time_limit"] == TIME_LIMIT_SECONDS
        changed_cells += 1
        print(f"{cell_name}: changed={len(to_change)} backup={backup}")

    print(f"changed_cells={changed_cells} changed_files={changed_files}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    migrate(args.root, args.timestamp)


if __name__ == "__main__":
    main()
