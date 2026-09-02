"""Move old-protocol AutoGluon artifacts aside before a clean, resumable rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL = "AUTOGLUON"
TARGET_CELLS = ("cap_10000_n100", "cap_full")
ARTIFACT_DIRS = ("logs", "predictions", "stats")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset(root: Path, timestamp: str, dry_run: bool) -> None:
    moved = 0
    for cell_name in TARGET_CELLS:
        cell = root / cell_name
        assert (cell / "config.json").is_file(), f"Missing configured cell: {cell}"
        candidates: list[Path] = []
        for seed_dir in sorted(cell.glob("seed_*")):
            for artifact_dir in ARTIFACT_DIRS:
                directory = seed_dir / artifact_dir
                if directory.is_dir():
                    candidates.extend(path for path in directory.glob(f"*{MODEL}*") if path.is_file())
        print(f"{cell_name}: artifacts={len(candidates)}")
        if dry_run or not candidates:
            continue

        backup = cell / "immutable_backups" / f"pre_autogluon_protocol_{timestamp}"
        assert not backup.exists(), f"Backup already exists: {backup}"
        manifest: dict[str, dict[str, str | int]] = {}
        for source in candidates:
            relative = source.relative_to(cell)
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            checksum = _checksum(source)
            size = source.stat().st_size
            shutil.move(source, destination)
            assert _checksum(destination) == checksum
            manifest[str(relative)] = {"sha256": checksum, "bytes": size}
            moved += 1
        manifest_path = backup / "sha256_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o440)

    print(f"moved={moved} dry_run={dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reset(args.root, args.timestamp, args.dry_run)


if __name__ == "__main__":
    main()
