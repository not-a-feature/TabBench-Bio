import json

import pytest

from tabbench_bio.evaluation import _build_excluded_keys


@pytest.mark.parametrize("field", ["exclude_datasets", "exclude_targets"])
def test_configured_exclusions_require_stats(tmp_path, field):
    config = dict(
        output_dir=str(tmp_path), exclude_keys=[], exclude_datasets=[], exclude_targets=[]
    )
    config[field] = ["excluded"]
    with pytest.raises(AssertionError, match="dataset_stats.json"):
        _build_excluded_keys(config)


def test_exclusion_keys_and_stats_are_combined(tmp_path):
    config = dict(
        output_dir=str(tmp_path),
        exclude_keys=["direct_0"],
        exclude_datasets=["whole"],
        exclude_targets=["age"],
    )
    (tmp_path / "dataset_stats.json").write_text(
        json.dumps(
            {
                "whole": {"target_names": ["a", "b"]},
                "partial": {"target_names": ["age", "weight"]},
            }
        )
    )
    assert _build_excluded_keys(config) == {"direct_0", "whole_0", "whole_1", "partial_0"}
    config.update(exclude_datasets=[], exclude_targets=[])
    (tmp_path / "dataset_stats.json").unlink()
    assert _build_excluded_keys(config) == {"direct_0"}
