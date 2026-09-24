"""Model dependencies stay in the environment selected by the registry."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabbench_bio import model_run
from tabbench_bio.cli import main
from tabbench_bio.models.custom import CUSTOM_MODELS


@pytest.mark.parametrize(
    "platform, executable", [("win32", "Scripts/python.exe"), ("linux", "bin/python")]
)
def test_profile_selects_platform_interpreter(tmp_path, monkeypatch, platform, executable):
    monkeypatch.setattr(model_run, "CHECKOUT", tmp_path)
    monkeypatch.setattr(model_run.sys, "platform", platform)
    profile = tmp_path / "environments/example.txt"
    profile.parent.mkdir()
    profile.write_text("-e .[bio,autogluon]\n")
    python = tmp_path / ".venvs/example" / executable
    python.parent.mkdir(parents=True)
    python.touch()
    assert model_run.model_python("example") == str(python)


@pytest.mark.parametrize("profile", ["../other", "", "a/b", "a\\b", "/absolute"])
def test_profile_cannot_escape_environment_directory(profile):
    with pytest.raises(AssertionError, match="Invalid environment profile"):
        model_run.model_python(profile)


def test_missing_profile_and_environment_have_setup_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(model_run, "CHECKOUT", tmp_path)
    with pytest.raises(AssertionError, match="Missing environment profile"):
        model_run.model_python("example")
    profile = tmp_path / "environments/example.txt"
    profile.parent.mkdir()
    profile.write_text("-e .[bio,autogluon]\n")
    with pytest.raises(AssertionError, match="uv venv .venvs/example"):
        model_run.model_python("example")


def test_missing_environment_stops_before_probe_or_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_run.sys, "argv", ["tabbench-bio", "EXAMPLE"])
    monkeypatch.setattr(model_run, "CHECKOUT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/grid_sweep_all.json").write_text(
        '{"models": [{"key": "EXAMPLE", "environment": "example", "device": "cpu"}]}'
    )
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments/example.txt").touch()
    monkeypatch.setattr(
        model_run.subprocess, "run", lambda *a, **kw: pytest.fail("Started a worker")
    )
    with pytest.raises(AssertionError, match="not installed"):
        main()
    assert not (tmp_path / "results").exists()


def test_probe_uses_selected_environment_and_preserves_gpu_allocation(tmp_path, monkeypatch):
    python = tmp_path / ".venvs/example/Scripts/python.exe"
    monkeypatch.setenv("VIRTUAL_ENV", "old-environment")
    monkeypatch.setenv("PYTHONHOME", "old-python")
    monkeypatch.setenv("PATH", "old-path")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated,GPU-other")

    def probe(command, **kwargs):
        assert command[0] == str(python)
        env = kwargs["env"]
        assert env["VIRTUAL_ENV"] == str(python.parent.parent)
        assert env["PATH"] == str(python.parent) + os.pathsep + "old-path"
        assert "PYTHONHOME" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-allocated"
        return SimpleNamespace(stdout='{"gpu":"NVIDIA L40S"}')

    monkeypatch.setattr(model_run.subprocess, "run", probe)
    model_run.worker_environment("gpu", 32, str(python))


def test_registered_models_have_profiles():
    root = Path(__file__).resolve().parents[1]
    roster = json.loads((root / "configs/models/all.json").read_text())
    for entry in [*roster, *CUSTOM_MODELS.values()]:
        assert (root / "environments" / f"{entry['environment']}.txt").is_file()
