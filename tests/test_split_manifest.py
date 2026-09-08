import inspect
import json
import shutil
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from tabbench_bio.benchmark import TabBenchBio, configure_benchmark
from tabbench_bio.result_store import (
    ResultRepository,
    consolidate_results,
    import_legacy_results,
    install_snapshots,
    snapshot_writers,
)
from tabbench_bio.split_manifest import (
    apply_frozen_split,
    consistent_truth_hashes,
    freeze_manifest,
    load_manifest,
    validate_prepared,
)


def populated(root, monkeypatch, *, shifted=False):
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-a")
    cell = root / "cap_2000_n20"
    repo = ResultRepository(cell, {"models": ["LR"], "cv_folds": 2})
    for seed in range(2):
        truth = pd.DataFrame({"target": [0.0, 1.0]}, index=[2 * seed, 2 * seed + 1])
        if shifted:
            truth.index += 10
        repo.write(
            {"dataset": "toy_0", "model": "LR", "status": "pass"},
            seed=seed,
            prediction=truth,
            ground_truth=truth,
        )
    return repo


def test_freeze_and_validate_subsampled_training_rows(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    path = freeze_manifest(repo, folds=2)
    assert freeze_manifest(repo, folds=2) == path
    manifest = load_manifest(repo.root)
    test = pd.DataFrame({"target": [0.0, 1.0]}, index=[0, 1])
    train = pd.DataFrame({"target": [0.0]}, index=[2])
    validate_prepared(manifest, 0, "toy_0", train, test)
    with pytest.raises(AssertionError, match="Training rows"):
        validate_prepared(manifest, 0, "toy_0", test, test)
    with pytest.raises(AssertionError, match="Frozen test rows"):
        validate_prepared(manifest, 0, "toy_0", train, test.rename(index={0: 9}))
    manifest["versions"]["scikit-learn"] = "old"
    with pytest.raises(AssertionError, match="environment changed"):
        validate_prepared(manifest, 0, "toy_0", train, test)


def test_other_model_and_other_cell_cannot_write_wrong_split(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    freeze_manifest(repo, folds=2)
    other = ResultRepository(repo.root / "cap_10000_n20", {"models": ["CAT"]})
    wrong = pd.DataFrame({"target": [0.0, 1.0]}, index=[10, 11])
    with pytest.raises(AssertionError, match="Frozen split mismatch"):
        other.write(
            {"dataset": "toy_0", "model": "CAT", "status": "pass"},
            seed=0,
            prediction=wrong,
            ground_truth=wrong,
        )
    with sqlite3.connect(repo.writer_path) as c:
        assert c.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_incompatible_incoming_snapshot_rejected_before_replacement(tmp_path, monkeypatch):
    good = populated(tmp_path / "good" / "experiment", monkeypatch)
    freeze_manifest(good, folds=2)
    before = good.writer_path.read_bytes()
    bad = populated(tmp_path / "bad" / "experiment", monkeypatch, shifted=True)
    snapshots = snapshot_writers(bad.root).parent
    with pytest.raises(AssertionError, match="Frozen split mismatch"):
        install_snapshots(good.root, snapshots)
    assert good.writer_path.read_bytes() == before


def test_publication_checks_cross_cell_disagreement_without_manifest(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    other = ResultRepository(repo.root / "cap_10000_n20", {"models": ["CAT"]})
    truth = pd.DataFrame({"target": [0.0, 1.0]}, index=[10, 11])
    other.write(
        {"dataset": "toy_0", "model": "CAT", "status": "pass"},
        seed=0,
        prediction=truth,
        ground_truth=truth,
    )
    combined = ResultRepository.from_root(repo.root)
    with pytest.raises(AssertionError, match="across cells"):
        consistent_truth_hashes(combined.current_attempts())
    with pytest.raises(AssertionError, match="across cells"):
        freeze_manifest(combined, folds=2)


def test_freeze_rejects_incomplete_or_overlapping_folds(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    with pytest.raises(AssertionError, match="Incomplete CV"):
        freeze_manifest(repo, folds=3)
    with sqlite3.connect(repo.writer_path) as c:
        digest = c.execute("SELECT ground_truth_sha256 FROM attempts WHERE seed=0").fetchone()[0]
        c.execute("UPDATE attempts SET ground_truth_sha256=? WHERE seed=1", (digest,))
    with pytest.raises(AssertionError, match="Overlapping CV"):
        freeze_manifest(ResultRepository.from_root(repo.root), folds=2)


def test_prepared_cache_changes_when_split_environment_changes(tmp_path, monkeypatch):
    kwargs = {
        "cache_dir": str(tmp_path / "cache"),
        "dataset_names_classification": [],
        "dataset_names_regression": [],
    }
    before = TabBenchBio(**kwargs).cache_dir_processed
    monkeypatch.setattr("tabbench_bio.benchmark.split_versions", lambda: {"sklearn": "changed"})
    assert TabBenchBio(**kwargs).cache_dir_processed != before


def test_stale_bundle_cannot_be_read_or_consolidated(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    freeze_manifest(repo, folds=2)
    canonical = consolidate_results(repo.root)
    before = canonical.read_bytes()
    monkeypatch.setenv("TABBENCH_WRITER_ID", "node-b")
    incoming = ResultRepository(
        tmp_path / "unfrozen" / "experiment" / repo.cell, {"models": ["LR"], "cv_folds": 2}
    )
    wrong = pd.DataFrame({"target": [0.0, 1.0]}, index=[10, 11])
    incoming.write(
        {"dataset": "toy_0", "model": "CAT", "status": "pass"},
        seed=0,
        prediction=wrong,
        ground_truth=wrong,
    )
    shutil.copy2(incoming.writer_path, repo.root / "writers" / incoming.writer_path.name)
    with pytest.raises(AssertionError, match="Frozen split mismatch"):
        ResultRepository.from_root(repo.root)
    with pytest.raises(AssertionError, match="Frozen split mismatch"):
        consolidate_results(repo.root)
    assert canonical.read_bytes() == before


def test_quarantined_legacy_split_cannot_be_reimported(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    freeze_manifest(repo, folds=2)
    cell = repo.root / repo.cell
    stats = cell / "seed_0" / "stats"
    predictions = cell / "seed_0" / "predictions"
    stats.mkdir(parents=True)
    predictions.mkdir()
    (cell / "config.json").write_text(json.dumps({"models": ["LR"], "cv_folds": 2}))
    (stats / "toy_0_CAT.json").write_text(
        json.dumps(
            {
                "dataset": "toy_0",
                "model": "CAT",
                "status": "pass",
            }
        )
    )
    truth = pd.DataFrame({"target": [0.0, 1.0]}, index=[10, 11])
    truth.to_csv(predictions / "toy_0_CAT_predictions.csv")
    truth.to_csv(predictions / "toy_0_ground_truth.csv")
    with pytest.raises(AssertionError, match="Frozen split mismatch"):
        import_legacy_results(repo.root)
    with sqlite3.connect(repo.root / "writers" / "legacy.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_frozen_assignment_is_version_independent_and_checks_groups(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    freeze_manifest(repo, folds=2)
    manifest = load_manifest(repo.root)
    frame = pd.DataFrame({"target": [0.0, 1.0, 0.0, 1.0]})
    monkeypatch.setattr("tabbench_bio.split_manifest.split_versions", lambda: {"old": "version"})
    train, test = apply_frozen_split(manifest, 0, "toy_0", frame, groups=["a", "a", "b", "b"])
    assert train.index.tolist() == [2, 3]
    assert test.index.tolist() == [0, 1]
    with pytest.raises(AssertionError, match="groups cross"):
        apply_frozen_split(manifest, 0, "toy_0", frame, groups=["a", "b", "a", "b"])
    with pytest.raises(AssertionError, match="Dataset rows changed"):
        apply_frozen_split(manifest, 0, "toy_0", frame.iloc[:-1])


def test_configured_benchmark_uses_frozen_rows_and_separate_cache(tmp_path, monkeypatch):
    repo = populated(tmp_path / "experiment", monkeypatch)
    freeze_manifest(repo, folds=2)
    config = {name: p.default for name, p in inspect.signature(TabBenchBio).parameters.items()}
    config.update(
        output_dir=str(repo.root / repo.cell),
        cache_dir=str(tmp_path / "cache"),
        cv_folds=2,
        random_state=0,
        min_samples_per_class=1,
        dataset_names_classification=["toy"],
        dataset_names_regression=[],
    )
    base = TabBenchBio(**{name: config[name] for name in inspect.signature(TabBenchBio).parameters})
    benchmark = configure_benchmark(config, init_benchmark=False)
    assert benchmark.cache_dir_processed.startswith(base.cache_dir_processed + "_frozen_")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [0.0, 1.0, 0.0, 1.0]})
    monkeypatch.setattr(
        benchmark,
        "_load_dataset",
        lambda _name: SimpleNamespace(n_targets=1, to_dataframe=lambda _idx: frame, groups=None),
    )
    monkeypatch.setattr(benchmark, "_fit_apply_features", lambda train, test, _key: (train, test))

    def forbidden_split(*_args):
        raise AssertionError("Library splitter must not run with a frozen manifest")

    monkeypatch.setattr(benchmark, "_split", forbidden_split)
    train, test = benchmark._load_dataset_from_key("toy_0")
    assert train.index.tolist() == [2, 3]
    assert test.index.tolist() == [0, 1]
    benchmark._save_dataset("toy_0", train, test)
    cached_train, cached_test = benchmark._load_dataset_from_cache("toy_0")
    pd.testing.assert_frame_equal(cached_train, train)
    pd.testing.assert_frame_equal(cached_test, test)
