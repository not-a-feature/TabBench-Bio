import pickle
import sys
from types import ModuleType

import pandas as pd
import pytest

from tabbench_bio.bio.cache import dataset_cache_path, load_cached_raw, save_cached_raw
from tabbench_bio.bio.loaders.base import BioRawDataset


@pytest.mark.parametrize("legacy", [False, True])
def test_cached_dataset_survives_package_rename(tmp_path, monkeypatch, legacy):
    raw = BioRawDataset(
        bio_id="example",
        X=pd.DataFrame({"feature": [1.0, 2.0]}),
        y=pd.Series([0, 1]),
        problem_type="binary",
        license="example",
        source_url="example",
        citation="example",
        metadata={},
    )
    with monkeypatch.context() as patch:
        if legacy:
            for name in ("tabarena_bio", "tabarena_bio.bio", "tabarena_bio.bio.loaders"):
                patch.setitem(sys.modules, name, ModuleType(name))
            name = "tabarena_bio.bio.loaders.base"
            module = ModuleType(name)
            module.BioRawDataset = BioRawDataset
            patch.setitem(sys.modules, name, module)
            patch.setattr(BioRawDataset, "__module__", name)
        path = save_cached_raw(tmp_path, raw)
    original = path.read_bytes()
    loaded = load_cached_raw(tmp_path, "example")
    assert isinstance(loaded, BioRawDataset)
    pd.testing.assert_frame_equal(loaded.X, raw.X)
    pd.testing.assert_series_equal(loaded.y, raw.y)
    assert path.read_bytes() == original


def test_unreadable_cache_is_preserved(tmp_path):
    path = dataset_cache_path(tmp_path, "example")
    path.parent.mkdir()
    path.write_bytes(b"invalid pickle")
    with pytest.raises(pickle.UnpicklingError):
        load_cached_raw(tmp_path, "example")
    assert path.read_bytes() == b"invalid pickle"
