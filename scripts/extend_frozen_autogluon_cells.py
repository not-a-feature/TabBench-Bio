"""Recoverably add AutoGluon to the two prespecified frozen result cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

MODEL = "AUTOGLUON"
TARGETS = {
    "cap_10000_n100": (10_000, 100),
    "cap_full": (None, None),
}
TIER_FILES = ("config_gpu_solo.json", "config_gpu_shared.json", "config_cpu.json")


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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


def _reference_override(roster_path: Path) -> dict:
    roster = _read(roster_path)
    matches = [entry for entry in roster if isinstance(entry, dict) and entry.get("key") == MODEL]
    assert len(matches) == 1, f"Expected exactly one {MODEL} roster entry in {roster_path}"
    entry = matches[0]
    override = {
        key: entry[key]
        for key in ("presets", "ensemble", "optimize", "num_hpo_trials")
        if key in entry
    }
    assert override.get("presets") == "extreme", override
    assert override.get("ensemble") is True, override
    return override


def _validate_partition(cell: Path) -> tuple[dict, dict[str, dict]]:
    full = _read(cell / "config.json")
    tiers = {name: _read(cell / name) for name in TIER_FILES if (cell / name).is_file()}
    tier_models = [model for tier in tiers.values() for model in tier["models"]]
    assert len(tier_models) == len(set(tier_models)), f"Duplicate tier model in {cell}"
    assert set(tier_models) == set(full["models"]), f"Tier partition mismatch in {cell}"
    assert "config_gpu_solo.json" in tiers, f"Missing GPU-solo tier in {cell}"
    for tier in tiers.values():
        for key in ("bio_max_features", "train_subsample", "output_dir"):
            assert tier[key] == full[key], f"{key} mismatch in {cell}"
    return full, tiers


def extend(root: Path, roster_path: Path, timestamp: str) -> None:
    override = _reference_override(roster_path)
    changed = 0
    already_configured = 0

    for name, (expected_cap, expected_samples) in TARGETS.items():
        cell = root / name
        assert (cell / "config.json").is_file(), f"Missing frozen cell: {cell}"
        full, tiers = _validate_partition(cell)
        assert full["bio_max_features"] == expected_cap, f"Unexpected feature cap in {cell}"
        assert full["train_subsample"] == expected_samples, f"Unexpected sample cap in {cell}"

        full_count = full["models"].count(MODEL)
        solo_count = tiers["config_gpu_solo.json"]["models"].count(MODEL)
        other_count = sum(
            tier["models"].count(MODEL)
            for tier_name, tier in tiers.items()
            if tier_name != "config_gpu_solo.json"
        )
        assert full_count in (0, 1), f"Invalid {MODEL} count in {cell / 'config.json'}"
        if full_count == 1:
            assert solo_count == 1 and other_count == 0, f"Invalid existing {MODEL} tier in {cell}"
            assert full.get("model_overrides", {}).get(MODEL) == override, (
                f"Unexpected existing {MODEL} override in {cell}"
            )
            already_configured += 1
            continue
        assert solo_count == 0 and other_count == 0, f"Tier contains absent {MODEL} in {cell}"

        config_paths = [cell / "config.json", *(cell / tier_name for tier_name in tiers)]
        backup = cell / "immutable_backups" / f"pre_autogluon_two_cell_{timestamp}"
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

        full["models"].append(MODEL)
        tiers["config_gpu_solo.json"]["models"].append(MODEL)
        for payload in (full, *tiers.values()):
            payload.setdefault("model_overrides", {})[MODEL] = override

        _write_atomic(cell / "config.json", full)
        for tier_name, tier in tiers.items():
            _write_atomic(cell / tier_name, tier)

        updated_full, updated_tiers = _validate_partition(cell)
        assert updated_full["models"].count(MODEL) == 1
        assert updated_tiers["config_gpu_solo.json"]["models"].count(MODEL) == 1
        assert updated_full["model_overrides"][MODEL] == override
        changed += 1

    print(f"cells={len(TARGETS)} changed={changed} already_configured={already_configured}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--roster", type=Path, default=Path("configs/models/all.json"))
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    extend(args.root, args.roster, args.timestamp)


if __name__ == "__main__":
    main()
