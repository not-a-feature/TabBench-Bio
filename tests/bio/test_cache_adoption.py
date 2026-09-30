import hashlib
import json

import pandas as pd
import pytest

from tabbench_bio.bio import cache, migration
from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.fingerprint import validate_cached_spec
from tabbench_bio.bio.loaders.base import BioRawDataset


@pytest.mark.parametrize("manifest_state", ["absent", "matching", "mismatched"])
def test_adoption_preserves_data_and_checks_frozen_targets(tmp_path, monkeypatch, manifest_state):
    spec = BioDatasetSpec("toy", "local", "toy.csv", target="y", problem_type="binary")
    monkeypatch.setattr(migration, "get_spec", lambda _: spec)
    raw = BioRawDataset(
        "toy", pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]), "binary", "", "", "", {}
    )
    path = cache.save_cached_raw(tmp_path, raw)
    original = path.read_bytes()
    manifest = None
    if manifest_state != "absent":
        manifest = tmp_path / "split_manifest.json"
        truth = pd.DataFrame({"target": [0, 1]})
        digest = hashlib.sha256(truth.to_csv(index=True, lineterminator="\n").encode()).hexdigest()
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cv_folds": 2,
                    "target_fingerprints": {
                        json.dumps(["toy_0", 0]): digest
                        if manifest_state == "matching"
                        else "wrong"
                    },
                    "units": {
                        json.dumps([0, "toy_0"]): {"test_indices": [0]},
                        json.dumps([1, "toy_0"]): {"test_indices": [1]},
                    },
                }
            )
        )
    if manifest_state == "mismatched":
        with pytest.raises(AssertionError, match="Frozen target fingerprint differs"):
            migration.adopt_cached_dataset(
                "toy", tmp_path, reason="Checked original registry", manifest=manifest
            )
        assert path.read_bytes() == original
        return
    migration.adopt_cached_dataset(
        "toy", tmp_path, reason="Checked original registry", manifest=manifest
    )
    adopted = cache.load_cached_raw(tmp_path, "toy")
    validate_cached_spec(adopted, spec)
    pd.testing.assert_frame_equal(adopted.X, raw.X)
    pd.testing.assert_series_equal(adopted.y, raw.y)
    assert adopted.metadata["cache_adoption"]["reason"] == "Checked original registry"
    with pytest.raises(AssertionError, match="Only unversioned"):
        migration.adopt_cached_dataset("toy", tmp_path, reason="No restamping")
