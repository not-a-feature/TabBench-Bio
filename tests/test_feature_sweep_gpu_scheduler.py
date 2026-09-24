import json
import subprocess
import threading
from pathlib import Path

import pytest

from scripts import feature_sweep


@pytest.mark.parametrize("devices", [["0"], ["0", "1"]])
def test_gpu_workers_are_exclusive_and_metrics_wait_for_all_units(monkeypatch, devices):
    spec = {
        "full_cfg": "config.json",
        "gpu_cfgs": ["config_gpu_solo.json", "config_gpu_shared.json"],
        "cpu_cfg": "config_cpu.json",
        "n_splits": 4,
        "tag": "test",
    }
    lock = threading.Lock()
    barrier = threading.Barrier(len(devices))
    active = set()
    completed = []
    metrics = []

    def run(command, *, env, check):
        device = env["CUDA_VISIBLE_DEVICES"]
        if device:
            with lock:
                assert device not in active
                active.add(device)
            barrier.wait(timeout=5)
            with lock:
                active.remove(device)
        with lock:
            completed.append((command[command.index("--config") + 1], command[-1]))

    def finish_metrics(*args):
        assert len(completed) == 12
        assert not active
        metrics.append(args)

    monkeypatch.setattr(feature_sweep.subprocess, "run", run)
    monkeypatch.setattr(feature_sweep, "_cell_metrics", finish_metrics)
    # Obsolete environment settings cannot enable GPU sharing.
    monkeypatch.setenv("GPU_WORKERS_PER_DEVICE", "8")
    monkeypatch.setenv("GPU_SOLO_WORKERS", "8")
    feature_sweep.run_grid_parallel(
        [spec], devices=devices, gpu_worker_cpus=32, cpu_workers=2, cpu_worker_cpus=8
    )
    assert len(set(completed)) == 12
    assert metrics == [("config.json", "test")]


def test_failed_gpu_unit_propagates_and_prevents_cell_metrics(monkeypatch):
    spec = {
        "full_cfg": "config.json",
        "gpu_cfgs": ["config_gpu.json"],
        "cpu_cfg": None,
        "n_splits": 2,
        "tag": "test",
    }
    completed = []
    metrics = []

    def run(command, **kwargs):
        completed.append(command)
        if command[-1] == "0":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(feature_sweep.subprocess, "run", run)
    monkeypatch.setattr(feature_sweep, "_cell_metrics", lambda *args: metrics.append(args))
    with pytest.raises(subprocess.CalledProcessError):
        feature_sweep.run_grid_parallel(
            [spec], devices=["0"], gpu_worker_cpus=32, cpu_workers=1, cpu_worker_cpus=8
        )
    assert len(completed) == 2
    assert not metrics


def test_new_gpu_config_and_legacy_resume(tmp_path):
    kwargs = dict(
        datasets=["toy"],
        datasets_regression=[],
        models=[
            {"key": "TABM", "device": "gpu", "solo": False},
            {"key": "TABDPT", "device": "gpu", "solo": True},
            {"key": "RF", "device": "cpu"},
        ],
        n_rep=None,
        cv_folds=2,
        time_limit=60,
        out_root=str(tmp_path),
        cache_dir=".cache",
        test_size=0.2,
        random_state=42,
        min_samples_per_class=2,
    )
    spec = feature_sweep._write_cell_config(2000, 100, **kwargs)
    gpu_path = Path(spec["gpu_cfgs"][0])
    assert gpu_path.name == "config_gpu.json"
    gpu_config = json.loads(gpu_path.read_text())
    assert gpu_config["models"] == ["TABM", "TABDPT"]
    assert feature_sweep._write_cell_config(2000, 100, **kwargs) == spec

    # Recreate an existing run's split GPU configs, then resume without rewriting them.
    for name, key in [("solo", "TABDPT"), ("shared", "TABM")]:
        gpu_path.with_name(f"config_gpu_{name}.json").write_text(
            json.dumps({**gpu_config, "models": [key]})
        )
    gpu_path.unlink()
    original = {path: path.read_bytes() for path in gpu_path.parent.glob("*.json")}
    resumed = feature_sweep._write_cell_config(2000, 100, **kwargs)
    assert [Path(path).name for path in resumed["gpu_cfgs"]] == [
        "config_gpu_solo.json",
        "config_gpu_shared.json",
    ]
    assert {path: path.read_bytes() for path in gpu_path.parent.glob("*.json")} == original
    assert resumed["n_splits"] == 2
