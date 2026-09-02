"""Synchronize the publication-relevant code from a TabArena-Bio checkout."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


PUBLIC_DIRECTORIES = (
    ".github",
    "configs",
    "src/tabbench_bio",
    "tests",
)
PUBLIC_FILES = (
    ".gitattributes",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "init.sh",
    "pyproject.toml",
    "requirements-autogluon-fork.txt",
    "scripts/audit_geo_covariates.py",
    "scripts/build_site.py",
    "scripts/extend_frozen_autogluon_cells.py",
    "scripts/extend_frozen_cells_tabpfn_v3.py",
    "scripts/feature_sweep.py",
    "scripts/fork_grid_results.py",
    "scripts/generate_brca_embeddings.py",
    "scripts/generate_brca_embeddings.sbatch",
    "scripts/generate_protein_embeddings.py",
    "scripts/generate_protein_embeddings.sbatch",
    "scripts/generate_social_preview.py",
    "scripts/materialize_null_dataset_skips.py",
    "scripts/migrate_frozen_autogluon_extreme.py",
    "scripts/reset_autogluon_reference.py",
    "scripts/resubmit_grid_array.sh",
    "scripts/run_grid.sbatch",
    "scripts/run_grid_array.sbatch",
    "scripts/run_grid_finalize.sbatch",
    "scripts/run_grid_full.sbatch",
    "scripts/run_grid_prepare.sbatch",
    "scripts/smoke_sqlite_api.py",
    "scripts/submit_grid_array.sh",
    "scripts/submit_grid_finalize.sh",
    "scripts/validate_bio_datasets.py",
)


def files_under(root: Path, relative: str) -> dict[str, Path]:
    base = root / relative
    assert base.is_dir(), f"Missing directory: {base}"
    return {
        path.relative_to(root).as_posix(): path
        for path in base.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def differences(source: Path, destination: Path) -> list[str]:
    mismatches: list[str] = []
    for relative in PUBLIC_DIRECTORIES:
        source_files = files_under(source, relative)
        destination_files = files_under(destination, relative)
        mismatches.extend(sorted(set(source_files) ^ set(destination_files)))
        for name in sorted(set(source_files) & set(destination_files)):
            if not filecmp.cmp(source_files[name], destination_files[name], shallow=False):
                mismatches.append(name)
    for relative in PUBLIC_FILES:
        source_file = source / relative
        destination_file = destination / relative
        assert source_file.is_file(), f"Missing file: {source_file}"
        if not destination_file.is_file() or not filecmp.cmp(
            source_file, destination_file, shallow=False
        ):
            mismatches.append(relative)
    return sorted(set(mismatches))


def synchronize(source: Path, destination: Path) -> None:
    for relative in PUBLIC_DIRECTORIES:
        source_path = source / relative
        destination_path = destination / relative
        assert source_path.is_dir(), f"Missing directory: {source_path}"
        shutil.rmtree(destination_path)
        shutil.copytree(
            source_path,
            destination_path,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for relative in PUBLIC_FILES:
        source_path = source / relative
        destination_path = destination / relative
        assert source_path.is_file(), f"Missing file: {source_path}"
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = Path(__file__).resolve().parents[1]
    assert source != destination, "Source and destination must differ"
    assert (source / "pyproject.toml").is_file(), f"Not a repository checkout: {source}"

    if not args.check:
        synchronize(source, destination)

    mismatches = differences(source, destination)
    assert not mismatches, "Public code differs:\n" + "\n".join(mismatches)
    print(f"Public code is synchronized with {source}")


if __name__ == "__main__":
    main()
