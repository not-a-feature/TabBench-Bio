"""Safely reuse artifacts from a frozen run in an expanded benchmark tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from tabbench_bio.io_utils import atomic_write_json

_MARKER = "reuse_manifest.json"
_COPIED_DIRS = ("metrics",)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_content(left: Path, right: Path) -> bool:
    try:
        if os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return left.stat().st_size == right.stat().st_size and _digest(left) == _digest(right)


def _validate_cell_configs(source: dict, destination: dict, cell: str) -> None:
    assert set(source["datasets_classification"]).issubset(
        destination["datasets_classification"]
    ), f"{cell}: destination drops classification datasets from the source run"
    assert set(source["datasets_regression"]).issubset(destination["datasets_regression"]), (
        f"{cell}: destination drops regression datasets from the source run"
    )
    assert set(source["models"]).issubset(destination["models"]), (
        f"{cell}: destination drops models from the source run"
    )
    # Memory priors only decide which still-missing units are attempted. They do not alter
    # any passing source prediction, so a newer calibration is compatible with copied data.
    ignored = {
        "datasets_classification",
        "datasets_regression",
        "models",
        "model_limits",
        "output_dir",
    }
    source_shared = {key: value for key, value in source.items() if key not in ignored}
    destination_shared = {key: value for key, value in destination.items() if key not in ignored}
    assert source_shared == destination_shared, (
        f"{cell}: source and destination differ beyond dataset expansion/output path"
    )


def _copy_file_without_overwrite(source: Path, destination: Path) -> str:
    if destination.exists():
        assert destination.is_file() and _same_content(source, destination), (
            f"Refusing to replace differing destination artifact: {destination}"
        )
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "copied"


def fork_grid_results(
    source_root: str | os.PathLike[str], destination_root: str | os.PathLike[str]
) -> dict:
    """Reuse source artifacts in an already-configured expanded result tree.

    Existing destination files are never overwritten. Artifacts are copied rather than hard
    linked so no future writer in either tree can mutate the other tree's inode.
    """
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    assert source != destination, "Source and destination result roots must differ."
    assert source.is_dir(), f"Missing source result tree: {source}"
    assert destination.is_dir(), (
        f"Destination must first be configured by feature_sweep.py: {destination}"
    )

    marker_path = destination / _MARKER
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert Path(marker["source"]).resolve() == source, (
            f"Destination was seeded from {marker['source']}, not {source}."
        )
        return marker

    source_cells = {path.parent.name: path.parent for path in source.glob("cap_*/config.json")}
    destination_cells = {
        path.parent.name: path.parent for path in destination.glob("cap_*/config.json")
    }
    assert source_cells, f"No configured source cells under {source}"
    assert set(source_cells) == set(destination_cells), (
        "Source/destination grid coordinates differ; start a fresh run instead of reusing."
    )

    counts = {"copied": 0, "existing": 0}
    for name in sorted(source_cells):
        source_cell = source_cells[name]
        destination_cell = destination_cells[name]
        source_config = json.loads((source_cell / "config.json").read_text(encoding="utf-8"))
        destination_config = json.loads(
            (destination_cell / "config.json").read_text(encoding="utf-8")
        )
        _validate_cell_configs(source_config, destination_config, name)

        roots = [path for path in source_cell.glob("seed_*") if path.is_dir()]
        roots.extend(
            source_cell / directory
            for directory in _COPIED_DIRS
            if (source_cell / directory).is_dir()
        )
        for artifact_root in roots:
            for source_path in artifact_root.rglob("*"):
                if not source_path.is_file():
                    continue
                if source_path.name.startswith(".") and source_path.name.endswith(".tmp"):
                    continue
                relative = source_path.relative_to(source_cell)
                outcome = _copy_file_without_overwrite(source_path, destination_cell / relative)
                counts[outcome] += 1

    marker = {
        "source": str(source),
        "destination": str(destination),
        "created_at": datetime.now().isoformat(),
        "files": counts,
    }
    atomic_write_json(marker_path, marker)
    return marker
