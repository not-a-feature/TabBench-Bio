import signal
import subprocess

import pytest

from scripts import feature_sweep


class _FakeProcess:
    active = 0
    max_active = 0
    commands = []
    returncodes = []

    def __init__(self, command, env):
        self.command = command
        self.env = env
        self.returncode = None
        self.polls = 0
        self.final_returncode = self.returncodes.pop(0)
        self.commands.append((command, env))
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)

    def poll(self):
        self.polls += 1
        if self.polls == 1:
            return None
        self.returncode = self.final_returncode
        type(self).active -= 1
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self):
        return self.returncode


def _units(count):
    spec = {
        "cpu_cfg": "config_cpu.json",
        "full_cfg": "config.json",
        "tag": "cap 2000/n=20",
    }
    return [(spec, seed) for seed in range(count)]


def _reset_fake(returncodes):
    _FakeProcess.active = 0
    _FakeProcess.max_active = 0
    _FakeProcess.commands = []
    _FakeProcess.returncodes = list(returncodes)


def test_cpu_scheduler_limits_parallelism_and_finishes_in_main_thread(monkeypatch):
    _reset_fake([0, 0, 0, 0, 0])
    monkeypatch.setattr(feature_sweep.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(feature_sweep.time, "sleep", lambda _: None)
    finished = []

    feature_sweep._run_cpu_units(
        _units(5),
        2,
        12,
        ("--include-model", "CAT"),
        lambda full_cfg, tag: finished.append((full_cfg, tag)),
    )

    assert _FakeProcess.max_active == 2
    assert len(_FakeProcess.commands) == 5
    assert all(env["CUDA_VISIBLE_DEVICES"] == "" for _, env in _FakeProcess.commands)
    assert all(env["SLURM_CPUS_PER_TASK"] == "12" for _, env in _FakeProcess.commands)
    assert all(command[-2:] == ["--include-model", "CAT"] for command, _ in _FakeProcess.commands)
    assert finished == [("config.json", "cap 2000/n=20")] * 5


def test_cpu_scheduler_completes_other_units_before_propagating_failure(monkeypatch):
    _reset_fake([1, 0, 0])
    monkeypatch.setattr(feature_sweep.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(feature_sweep.time, "sleep", lambda _: None)
    finished = []

    with pytest.raises(subprocess.CalledProcessError):
        feature_sweep._run_cpu_units(
            _units(3),
            2,
            8,
            (),
            lambda full_cfg, tag: finished.append((full_cfg, tag)),
        )

    assert len(_FakeProcess.commands) == 3
    assert finished == [("config.json", "cap 2000/n=20")] * 2


def test_empty_device_pool_schedules_only_cpu_units(monkeypatch):
    spec = {
        "full_cfg": "config.json",
        "gpu_solo_cfg": "config_gpu_solo.json",
        "gpu_shared_cfg": "config_gpu_shared.json",
        "cpu_cfg": "config_cpu.json",
        "n_splits": 2,
        "tag": "cap 2000/n=20",
    }
    scheduled = []
    metrics = []

    def run_cpu_units(units, _workers, _cpus, _args, finish):
        scheduled.extend(units)
        for unit_spec, _seed_index in units:
            finish(unit_spec["full_cfg"], unit_spec["tag"])

    monkeypatch.setattr(feature_sweep, "_run_cpu_units", run_cpu_units)
    monkeypatch.setattr(
        feature_sweep,
        "_cell_metrics",
        lambda full_cfg, tag: metrics.append((full_cfg, tag)),
    )

    feature_sweep.run_grid_parallel(
        [spec],
        devices=[],
        workers_per_device=1,
        solo_workers=1,
        gpu_worker_cpus=1,
        cpu_workers=2,
        cpu_worker_cpus=4,
    )

    assert scheduled == [(spec, 0), (spec, 1)]
    assert metrics == [("config.json", "cap 2000/n=20")]


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is POSIX-only")
def test_stack_dump_signal_is_registered(monkeypatch):
    calls = []
    monkeypatch.setattr(
        feature_sweep.faulthandler,
        "register",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    feature_sweep._enable_stack_dumps()

    assert calls == [((signal.SIGUSR1,), {"all_threads": True})]
