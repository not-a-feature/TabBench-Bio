"""Installed users can create workspaces and supply models outside the package."""

import json

import pytest

from tabbench_bio import resources
from tabbench_bio.evaluate import load_model_factory


def test_workspace_exports_versioned_profiles_and_protects_existing_files(tmp_path):
    destination = tmp_path / "workspace"
    resources.init_workspace(destination)
    grid = json.loads((destination / "configs/grid_sweep_all.json").read_text())
    for field in ("models", "datasets", "datasets_regression"):
        assert (destination / "configs" / grid[field]).is_file()
    for profile in (destination / "environments").glob("*.txt"):
        content = profile.read_text()
        assert "-e ." not in content
        assert f"]=={resources.__version__}" in content
    assert (destination / "requirements-autogluon-fork.txt").is_file()
    assert callable(load_model_factory(str(destination / "my_model.py")))
    with pytest.raises(AssertionError, match="new workspace"):
        resources.init_workspace(destination)


def test_external_factory_supports_sibling_imports_and_custom_function(tmp_path):
    (tmp_path / "custom_backend.py").write_text("from sklearn.dummy import DummyClassifier\n")
    model = tmp_path / "custom.py"
    model.write_text(
        "from custom_backend import DummyClassifier\n"
        "def build(task):\n"
        "    assert task == 'classification'\n"
        "    return DummyClassifier()\n"
    )
    estimator = load_model_factory(f"{model}:build")("classification")
    estimator.fit([[1], [2]], [0, 1])
    assert len(estimator.predict([[3]])) == 1
    assert load_model_factory(f"{model}:build").__module__ == model.stem
