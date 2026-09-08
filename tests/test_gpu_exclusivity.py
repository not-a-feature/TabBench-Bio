from pathlib import Path

import pytest

from tabbench_bio import gpu_exclusivity


def test_gpu_status_distinguishes_own_and_foreign_processes(monkeypatch, tmp_path: Path):
    (tmp_path / "101").mkdir()
    (tmp_path / "202").mkdir()
    (tmp_path / "101" / "environ").write_bytes(
        b"CUDA_VISIBLE_DEVICES=0\0TABBENCH_BIO_GPU_GUARD_TOKEN=run-1\0"
    )

    responses = iter(
        [
            [["0", "GPU-a", "70000", "81559"]],
            [
                ["GPU-a", "101", "python", "1000"],
                ["GPU-a", "202", "python", "12000"],
            ],
        ]
    )
    monkeypatch.setattr(gpu_exclusivity, "_query", lambda fields: next(responses))

    status = gpu_exclusivity.gpu_status(("0",), "run-1", proc_root=tmp_path)["0"]

    assert status.free_mib == 70000
    assert status.total_mib == 81559
    assert status.foreign_processes == ("pid=202 python (12000 MiB)",)


def test_gpu_status_accepts_uuid(monkeypatch, tmp_path: Path):
    responses = iter(
        [
            [["0", "GPU-a", "80000", "81559"]],
            [],
        ]
    )
    monkeypatch.setattr(gpu_exclusivity, "_query", lambda fields: next(responses))

    status = gpu_exclusivity.gpu_status(("GPU-a",), "run-1", proc_root=tmp_path)

    assert status["GPU-a"].exclusive


def test_gpu_status_ignores_stale_driver_process(monkeypatch, tmp_path: Path):
    responses = iter(
        [
            [["0", "GPU-a", "78000", "81559"]],
            [["GPU-a", "999", "[Not Found]", "3390"]],
        ]
    )
    monkeypatch.setattr(gpu_exclusivity, "_query", lambda fields: next(responses))

    status = gpu_exclusivity.gpu_status(("0",), "run-1", proc_root=tmp_path)

    assert status["0"].exclusive


def test_gpu_status_ignores_small_missing_pid_with_unknown_name(monkeypatch, tmp_path: Path):
    responses = iter(
        [
            [["0", "GPU-a", "78000", "81559"]],
            [["GPU-a", "999", "[No data]", "4096"]],
        ]
    )
    monkeypatch.setattr(gpu_exclusivity, "_query", lambda fields: next(responses))

    status = gpu_exclusivity.gpu_status(("0",), "run-1", proc_root=tmp_path)

    assert status["0"].exclusive


def test_gpu_status_rejects_large_missing_pid_allocation(monkeypatch, tmp_path: Path):
    responses = iter(
        [
            [["0", "GPU-a", "76000", "81559"]],
            [["GPU-a", "999", "[Not Found]", "10241"]],
        ]
    )
    monkeypatch.setattr(gpu_exclusivity, "_query", lambda fields: next(responses))

    status = gpu_exclusivity.gpu_status(("0",), "run-1", proc_root=tmp_path)

    assert status["0"].foreign_processes == ("pid=999 [Not Found] (10241 MiB)",)


def test_gpu_status_allows_up_to_ten_gib_foreign_usage(monkeypatch, tmp_path: Path):
    (tmp_path / "202").mkdir()
    responses = iter(
        [
            [["0", "GPU-a", "71000", "81559"]],
            [["GPU-a", "202", "python", "10240"]],
        ]
    )
    monkeypatch.setattr(gpu_exclusivity, "_query", lambda fields: next(responses))

    status = gpu_exclusivity.gpu_status(("0",), "run-1", proc_root=tmp_path)

    assert status["0"].exclusive


def test_gpu_status_rejects_unknown_device(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        gpu_exclusivity,
        "_query",
        lambda fields: [["0", "GPU-a", "80000", "81559"]],
    )

    with pytest.raises(AssertionError, match="Unknown GPU"):
        gpu_exclusivity.gpu_status(("2",), "run-1", proc_root=tmp_path)
