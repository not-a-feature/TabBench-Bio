"""Frozen cross-validation identities shared by every cell in an experiment."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from tabbench_bio.io_utils import atomic_write_json

FILENAME = "split_manifest.json"


def split_versions() -> dict[str, str]:
    return {"numpy": np.__version__, "scikit-learn": sklearn.__version__}


def unit_id(seed: int, dataset: str) -> str:
    return json.dumps([int(seed), dataset], separators=(",", ":"))


def consistent_truth_hashes(attempts) -> dict[tuple[int, str], str]:
    """Require one held-out target artifact across all models and grid cells."""
    hashes = {}
    for attempt in attempts:
        if attempt.ground_truth_sha256 is None:
            continue
        key = (attempt.seed, attempt.dataset)
        if key in hashes:
            assert hashes[key] == attempt.ground_truth_sha256, (
                f"Conflicting held-out-target hashes across cells for {key}: "
                f"{attempt.cell}/{attempt.model}"
            )
        hashes[key] = attempt.ground_truth_sha256
    return hashes


def load_manifest(root: Path) -> dict | None:
    path = root / FILENAME
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["experiment_id"] == root.name
    return manifest


def validate_truth(manifest: dict, seed: int, dataset: str, digest: str) -> None:
    key = unit_id(seed, dataset)
    assert key in manifest["units"], f"Unregistered frozen split: {key}"
    assert manifest["units"][key]["ground_truth_sha256"] == digest, (
        f"Frozen split mismatch for {key}; quarantine incompatible results before resuming."
    )


def validate_prepared(
    manifest: dict,
    seed: int,
    dataset: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    check_versions: bool = True,
) -> None:
    assert not check_versions or manifest["versions"] == split_versions(), (
        f"Split environment changed: expected {manifest['versions']}, got {split_versions()}"
    )
    key = unit_id(seed, dataset)
    assert key in manifest["units"], f"Unregistered frozen split: {key}"
    unit = manifest["units"][key]
    truth = test[["target"]].sort_index()
    assert truth.index.tolist() == unit["test_indices"], f"Frozen test rows changed: {key}"
    assert train.index.is_unique and set(train.index) <= set(unit["train_indices"]), (
        f"Training rows outside frozen training partition: {key}"
    )
    buffer = io.StringIO(newline="")
    truth.to_csv(buffer, index=True, lineterminator="\n")
    validate_truth(manifest, seed, dataset, hashlib.sha256(buffer.getvalue().encode()).hexdigest())


def apply_frozen_split(
    manifest: dict, seed: int, dataset: str, frame: pd.DataFrame, groups=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the recorded partitions without invoking a version-dependent splitter."""
    key = unit_id(seed, dataset)
    assert key in manifest["units"], f"Unregistered frozen split: {key}"
    unit = manifest["units"][key]
    train_indices, test_indices = unit["train_indices"], unit["test_indices"]
    assert not set(train_indices) & set(test_indices), f"Overlapping frozen partitions: {key}"
    assert frame.index.is_unique and set(frame.index) == set(train_indices) | set(test_indices), (
        f"Dataset rows changed relative to frozen split: {key}"
    )
    train, test = frame.loc[train_indices].copy(), frame.loc[test_indices].copy()
    if groups is not None:
        aligned = pd.Series(np.asarray(groups), index=frame.index)
        assert not set(aligned.loc[train_indices]) & set(aligned.loc[test_indices]), (
            f"Biological groups cross frozen partitions: {key}"
        )
    validate_prepared(manifest, seed, dataset, train, test, check_versions=False)
    return train, test


def freeze_manifest(repository, *, folds: int) -> Path:
    """Freeze a unanimously consistent, complete set of CV test partitions."""
    assert folds >= 2
    attempts = repository.current_attempts()
    consistent_truth_hashes(attempts)
    representatives = {}
    for attempt in attempts:
        if attempt.ground_truth_sha256 is not None:
            representatives.setdefault((attempt.seed, attempt.dataset), attempt)
    assert representatives, "No passing split artifacts to freeze"
    frames = {key: repository.dataframe(a, "ground_truth") for key, a in representatives.items()}
    repeats = defaultdict(dict)
    for (seed, dataset), frame in frames.items():
        assert frame.index.is_unique
        repeats[(dataset, seed // folds)][seed % folds] = frame
    universes = {}
    target_fingerprints = {}
    for key, partitions in repeats.items():
        assert set(partitions) == set(range(folds)), f"Incomplete CV partition: {key}"
        combined = pd.concat([partitions[i] for i in range(folds)]).sort_index()
        assert combined.index.is_unique, f"Overlapping CV test folds: {key}"
        universes[key] = set(combined.index)
        target_fingerprints[json.dumps(key)] = hashlib.sha256(
            combined.to_csv(index=True, lineterminator="\n").encode()
        ).hexdigest()
    units = {}
    for (seed, dataset), frame in frames.items():
        test_indices = frame.sort_index().index.tolist()
        units[unit_id(seed, dataset)] = {
            "test_indices": test_indices,
            "train_indices": sorted(universes[(dataset, seed // folds)] - set(test_indices)),
            "ground_truth_sha256": representatives[(seed, dataset)].ground_truth_sha256,
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": repository.root.name,
        "versions": split_versions(),
        "cv_folds": folds,
        "target_fingerprints": target_fingerprints,
        "units": units,
    }
    path = repository.root / FILENAME
    if path.exists():
        assert load_manifest(repository.root) == manifest, (
            "Frozen split manifest cannot be replaced"
        )
    else:
        atomic_write_json(path, manifest)
    return path
