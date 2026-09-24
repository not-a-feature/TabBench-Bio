"""Short commands preserve the protocol and distinguish CPUs from model threads."""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tabbench_bio import model_run
from tabbench_bio.cli import main
from tabbench_bio.result_store import ResultRepository


@pytest.fixture(autouse=True)
def installed_model_environment(monkeypatch):
    monkeypatch.setattr(model_run, "CHECKOUT", Path(__file__).resolve().parents[1])
    monkeypatch.setattr(model_run, "model_python", lambda profile: model_run.sys.executable)


@pytest.mark.parametrize(
    "cpus,threads,gpu,warning",
    [
        (18, 32, "NVIDIA L40S", False),
        (32, 32, "NVIDIA L40S", True),
        (18, 16, "NVIDIA L40S", True),
        (18, 32, "NVIDIA A100", True),
        (18, 32, None, True),
    ],
)
def test_hardware_warning_and_single_gpu(monkeypatch, capsys, cpus, threads, gpu, warning):
    monkeypatch.setattr(model_run.sys, "platform", "win32")
    monkeypatch.setattr(model_run.os, "cpu_count", lambda: 64)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(cpus))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    seen = []

    def probe(command, **kwargs):
        seen.append(kwargs["env"])
        return SimpleNamespace(stdout=json.dumps({"gpu": gpu}))

    monkeypatch.setattr(model_run.subprocess, "run", probe)
    if gpu is None:
        with pytest.raises(AssertionError, match="requires a GPU"):
            model_run.worker_environment("gpu", threads)
    else:
        env, hardware = model_run.worker_environment("gpu", threads)
        assert hardware["threads"] == threads
        assert hardware["available_threads"] == cpus
        assert env["TABBENCH_MODEL_CPUS"] == str(threads)
        assert env["OMP_NUM_THREADS"] == str(threads)
    assert seen[0]["CUDA_VISIBLE_DEVICES"] == "GPU-first"
    assert ("less comparable" in capsys.readouterr().err) == warning


def test_cpu_override_hides_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        model_run.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout='{"gpu":null}')
    )
    env, _ = model_run.worker_environment("cpu", 1)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == "1"


def test_reference_then_full_grid_resume_and_unknown_model(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []
    python = str(tmp_path / ".venvs/standard/Scripts/python.exe")

    def select_profile(profile):
        assert profile == "standard"
        return python

    monkeypatch.setattr(model_run, "model_python", select_profile)
    monkeypatch.setattr(
        model_run, "worker_environment", lambda *a: ({}, {"gpu": "NVIDIA L40S", "threads": 32})
    )

    def worker(command, **kwargs):
        assert command[0] == python
        assert json.loads(kwargs["env"]["TABBENCH_MODEL_HARDWARE"])["environment"] == "standard"
        if "--config" in command:
            path = Path(command[command.index("--config") + 1])
            config = json.loads(path.read_text())
            calls.append(config)
            repository = ResultRepository(path.parent, config)
            if repository.current(0, "toy_0", "RF") is None:
                repository.write(
                    {
                        "dataset": "toy_0",
                        "model": "RF",
                        "status": "skip",
                        "reason": "model_limit",
                    },
                    seed=0,
                )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(model_run, "subprocess", SimpleNamespace(run=worker))
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "RF"])
    main()
    assert len(calls) == 1
    assert calls[0]["bio_max_features"] == 10000 and calls[0]["train_subsample"] == 100
    root = tmp_path / "results/rf"
    frozen = (root / "cap_10000_n100/config.json").read_bytes()
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "RF", "--full-grid"])
    main()
    assert len(calls) == 29
    assert len(list(root.glob("cap_*/config.json"))) == 28
    assert (root / "cap_10000_n100/config.json").read_bytes() == frozen
    assert len(ResultRepository.from_root(root).attempts()) == 28
    assert json.loads((root / "hardware.jsonl").read_text().splitlines()[0])["python"] == python
    assert "Database:" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "NO_SUCH_MODEL"])
    with pytest.raises(AssertionError, match="Unknown model"):
        main()


def test_changed_frozen_settings_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(model_run, "worker_environment", lambda *a: ({}, {}))
    monkeypatch.setattr(model_run.subprocess, "run", lambda *a, **kw: None)
    cell = tmp_path / "cap_10000_n100"
    cell.mkdir()
    path = cell / "config.json"
    path.write_text('{"models":["WRONG"]}')
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "RF", "--output", str(tmp_path)])
    with pytest.raises(AssertionError, match="Settings changed"):
        main()
    assert path.read_text() == '{"models":["WRONG"]}'


def test_autogluon_native_preset_passes_preflight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_run, "worker_environment", lambda *a: ({}, {}))
    model_module = ModuleType("tabbench_bio.model")

    def resolve(*args):
        pytest.fail("The native AUTOGLUON preset must not resolve to a single model class")

    model_module._resolve_hyperparameters = resolve
    monkeypatch.setitem(model_run.sys.modules, "tabbench_bio.model", model_module)

    def worker(command, **kwargs):
        if "-c" in command:
            with monkeypatch.context() as context:
                context.setattr(model_run.sys, "argv", ["-c", *command[3:]])
                exec(command[2], {})
        else:
            path = Path(command[command.index("--config") + 1])
            config = json.loads(path.read_text())
            assert config["model_overrides"]["AUTOGLUON"] == {
                "ensemble": True,
                "presets": "extreme",
            }
            repository = ResultRepository(path.parent, config)
            repository.write(
                {
                    "dataset": "toy_0",
                    "model": "AUTOGLUON",
                    "status": "skip",
                    "reason": "model_limit",
                },
                seed=0,
            )

    monkeypatch.setattr(model_run, "subprocess", SimpleNamespace(run=worker))
    monkeypatch.setattr("sys.argv", ["tabbench-bio", "AUTOGLUON"])
    main()
    assert (tmp_path / "results/autogluon/results.sqlite").is_file()


def test_leaderboard_rejects_conflicting_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["tabbench-bio", "leaderboard", "first.sqlite", "--sqlite", "second.sqlite"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
