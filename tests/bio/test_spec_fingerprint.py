from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.bio import adapter, cache, datasets, fingerprint
from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.loaders.base import BioRawDataset


@pytest.fixture
def spec():
    return BioDatasetSpec("toy", "local", "table.csv", target="y", problem_type="binary")


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "other"),
        ("problem_type", "regression"),
        ("fetch_id", "new.csv"),
        ("group_column", "patient"),
        ("source_max_features", 5),
        ("embedding_column", "embedding"),
    ],
)
def test_registry_changes_invalidate_raw_and_split_cache(tmp_path, monkeypatch, spec, field, value):
    raw = BioRawDataset(
        "toy", pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1]), "binary", "", "", "", {}
    )
    loader = Mock()
    loader.fetch.return_value = raw
    monkeypatch.setattr(adapter, "get_loader", lambda *a, **kw: loader)
    monkeypatch.setattr(adapter, "get_spec", lambda _: spec)
    fetched = adapter.load_bio_dataset("toy", cache_dir=str(tmp_path))
    assert fetched.metadata["spec_sha256"] == fingerprint.spec_fingerprint(spec)
    adapter.load_bio_dataset("toy", cache_dir=str(tmp_path))
    assert loader.fetch.call_count == 1
    changed = replace(spec, **{field: value})
    monkeypatch.setattr(adapter, "get_spec", lambda _: changed)
    with pytest.raises(AssertionError, match="stale dataset cache"):
        adapter.load_bio_dataset("toy", cache_dir=str(tmp_path))
    assert loader.fetch.call_count == 1
    refreshed = adapter.load_bio_dataset("toy", cache_dir=str(tmp_path), force_refetch=True)
    assert refreshed.metadata["spec_sha256"] == fingerprint.spec_fingerprint(changed)
    monkeypatch.setattr("tabbench_bio.benchmark.reload_bio_registry", lambda: None)
    monkeypatch.setitem(datasets.BIO_DATASETS, "toy", spec)
    first = TabBenchBio(["toy"], [], cache_dir=str(tmp_path))
    monkeypatch.setitem(datasets.BIO_DATASETS, "toy", changed)
    second = TabBenchBio(["toy"], [], cache_dir=str(tmp_path))
    assert first.cache_dir_processed == second.cache_dir_processed
    monkeypatch.setitem(datasets.BIO_DATASETS, "toy", spec)
    original_paths = first._get_cache_paths("toy_0")
    monkeypatch.setitem(datasets.BIO_DATASETS, "toy", changed)
    assert second._get_cache_paths("toy_0") != original_paths


def test_loader_version_invalidates_fingerprint(spec, monkeypatch):
    old = fingerprint.spec_fingerprint(spec)
    monkeypatch.setattr(fingerprint, "LOADER_CACHE_VERSION", fingerprint.LOADER_CACHE_VERSION + 1)
    assert fingerprint.spec_fingerprint(spec) != old


def test_existing_split_metadata_is_validated_without_writing(tmp_path, monkeypatch):
    initial = TabBenchBio([], [], cache_dir=str(tmp_path))
    path = Path(initial.cache_dir_processed) / "split_params.json"
    original = path.read_bytes()

    def forbidden_write(*args, **kwargs):
        raise AssertionError("Reusing a cache must not rewrite its metadata")

    monkeypatch.setattr("tabbench_bio.benchmark.atomic_write_json", forbidden_write)
    reused = TabBenchBio([], [], cache_dir=str(tmp_path))
    assert reused.cache_dir_processed == initial.cache_dir_processed
    assert path.read_bytes() == original
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="Split cache metadata mismatch"):
        TabBenchBio([], [], cache_dir=str(tmp_path))


def test_interrupted_pickle_write_preserves_previous_cache(tmp_path, monkeypatch):
    raw = BioRawDataset("toy", pd.DataFrame({"x": [1]}), pd.Series([0]), "binary", "", "", "", {})
    path = cache.save_cached_raw(tmp_path, raw)
    previous = path.read_bytes()

    def broken_write(data, handle):
        handle.write(b"partial pickle")
        raise OSError("interrupted")

    monkeypatch.setattr(pd, "to_pickle", broken_write)
    with pytest.raises(OSError, match="interrupted"):
        cache.save_cached_raw(tmp_path, raw)
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "changes",
    [
        {"notes": "new notes"},
        {"display_name": "New name"},
        {"license": "EUPL-1.2"},
        {"enabled": False},
        {"data_type": "Other"},
        {"eval_metric": "accuracy"},
        {"redistributable": False},
    ],
)
def test_provenance_changes_do_not_invalidate_data(spec, changes):
    assert fingerprint.spec_fingerprint(spec) == fingerprint.spec_fingerprint(
        replace(spec, **changes)
    )


def test_dataset_selection_does_not_change_existing_split_paths(tmp_path):
    first = TabBenchBio(["OpenML-1138"], [], cache_dir=str(tmp_path))
    second = TabBenchBio(["OpenML-1138", "OpenML-1083"], [], cache_dir=str(tmp_path))
    assert first._get_cache_paths("OpenML-1138_0") == second._get_cache_paths("OpenML-1138_0")


def test_prevalence_only_invalidates_its_dataset(spec):
    original = fingerprint.spec_fingerprint(spec)
    assert fingerprint.spec_fingerprint(replace(spec, train_prevalence_filter=None)) == original
    assert fingerprint.spec_fingerprint(replace(spec, train_prevalence_filter=0.1)) != original
