"""Fail-closed detection of foreign CUDA processes on benchmark GPUs."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

TOKEN_ENV = "TABBENCH_BIO_GPU_GUARD_TOKEN"
DEFAULT_MAX_STALE_MIB = 4_096
DEFAULT_MAX_FOREIGN_MIB = 10_240


class GpuContentionError(RuntimeError):
    """A foreign CUDA process is using a guarded benchmark GPU."""


@dataclass(frozen=True)
class GpuStatus:
    free_mib: int
    total_mib: int
    foreign_processes: tuple[str, ...]

    @property
    def exclusive(self) -> bool:
        return not self.foreign_processes


def _query(fields: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{fields.split(':', maxsplit=1)[0]}={fields.split(':', maxsplit=1)[1]}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [[value.strip() for value in row] for row in csv.reader(StringIO(completed.stdout))]


def _owned_by_run(pid: int, run_token: str, proc_root: Path) -> bool:
    if pid == os.getpid():
        return True
    try:
        environment = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    marker = f"{TOKEN_ENV}={run_token}".encode()
    return marker in environment


def gpu_status(
    devices: tuple[str, ...],
    run_token: str,
    *,
    proc_root: Path = Path("/proc"),
    max_stale_mib: int = DEFAULT_MAX_STALE_MIB,
    max_foreign_mib: int = DEFAULT_MAX_FOREIGN_MIB,
) -> dict[str, GpuStatus]:
    assert devices
    assert max_stale_mib >= 0
    assert max_foreign_mib >= 0
    gpu_rows = _query("gpu:index,uuid,memory.free,memory.total")
    gpu_by_token: dict[str, tuple[str, int, int]] = {}
    for index, uuid, free_mib, total_mib in gpu_rows:
        values = (uuid, int(free_mib), int(total_mib))
        gpu_by_token[index] = values
        gpu_by_token[uuid] = values
    missing = [device for device in devices if device not in gpu_by_token]
    assert not missing, f"Unknown GPU device token(s): {missing}"

    target = {gpu_by_token[device][0]: device for device in devices}
    foreign: dict[str, list[tuple[str, int]]] = {device: [] for device in devices}
    for uuid, pid_text, process_name, used_mib in _query(
        "compute-apps:gpu_uuid,pid,process_name,used_memory"
    ):
        if uuid not in target:
            continue
        pid = int(pid_text)
        if not (proc_root / str(pid)).exists() and int(used_mib) <= max_stale_mib:
            continue
        if not _owned_by_run(pid, run_token, proc_root):
            device = target[uuid]
            foreign[device].append((f"pid={pid} {process_name} ({used_mib} MiB)", int(used_mib)))

    return {
        device: GpuStatus(
            free_mib=gpu_by_token[device][1],
            total_mib=gpu_by_token[device][2],
            foreign_processes=(
                ()
                if sum(used_mib for _, used_mib in foreign[device]) <= max_foreign_mib
                else tuple(description for description, _ in foreign[device])
            ),
        )
        for device in devices
    }


def assert_exclusive_from_environment() -> None:
    run_token = os.environ[TOKEN_ENV]
    devices = tuple(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    status = gpu_status(devices, run_token)
    conflicts = {
        device: current.foreign_processes
        for device, current in status.items()
        if not current.exclusive
    }
    if conflicts:
        raise GpuContentionError(f"Foreign CUDA process detected: {conflicts}")


def _format(status: dict[str, GpuStatus]) -> str:
    return "; ".join(
        f"GPU {device}: {current.free_mib}/{current.total_mib} MiB free, "
        f"foreign={list(current.foreign_processes)}"
        for device, current in status.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=75_000)
    parser.add_argument("--stable-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--max-stale-mib", type=int, default=DEFAULT_MAX_STALE_MIB)
    parser.add_argument("--max-foreign-mib", type=int, default=DEFAULT_MAX_FOREIGN_MIB)
    args = parser.parse_args()

    devices = tuple(args.devices.split(","))
    stable_since = None
    previous = None
    while True:
        status = gpu_status(
            devices,
            args.run_token,
            max_stale_mib=args.max_stale_mib,
            max_foreign_mib=args.max_foreign_mib,
        )
        rendered = _format(status)
        if rendered != previous:
            print(rendered, flush=True)
            previous = rendered
        ready = all(
            current.exclusive and current.free_mib >= args.min_free_mib
            for current in status.values()
        )
        if not args.wait:
            raise SystemExit(0 if all(current.exclusive for current in status.values()) else 1)
        now = time.monotonic()
        stable_since = now if ready and stable_since is None else stable_since
        stable_since = None if not ready else stable_since
        if stable_since is not None and now - stable_since >= args.stable_seconds:
            print("GPU exclusivity stable; starting benchmark.", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
