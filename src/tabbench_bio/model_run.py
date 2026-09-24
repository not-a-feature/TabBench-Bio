"""Run one model through the standard grid from an editable checkout."""

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from tabbench_bio.config import (
    cell_name,
    config_for_cell,
    model_limits,
    model_overrides,
    resolve_list,
)
from tabbench_bio.io_utils import atomic_write_json
from tabbench_bio.models.custom import CUSTOM_MODELS
from tabbench_bio.result_store import ResultRepository, consolidate_results

CHECKOUT = Path(__file__).resolve().parents[2]


def model_python(profile: str) -> str:
    """Find the installed interpreter for a model's environment profile."""
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]*", profile), (
        f"Invalid environment profile: {profile!r}"
    )
    requirements = CHECKOUT / "environments" / f"{profile}.txt"
    assert requirements.is_file(), f"Missing environment profile: {requirements}"
    environment = CHECKOUT / ".venvs" / profile
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    assert executable.is_file(), (
        f"Environment {profile!r} is not installed. From {CHECKOUT}, run:\n"
        f"uv venv .venvs/{profile} --python 3.12\n"
        f'uv pip install --python "{executable}" -r environments/{profile}.txt'
    )
    return str(executable)


def worker_environment(
    device: str, requested_threads: int, python: str = sys.executable
) -> tuple[dict, dict]:
    """Select one GPU and configure model threads independently of CPU allocation."""
    available = len(os.sched_getaffinity(0)) if sys.platform == "linux" else os.cpu_count()
    assert available is not None and available > 0, "Cannot determine CPU allocation"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        available = min(available, int(os.environ["SLURM_CPUS_PER_TASK"]))
    assert requested_threads > 0, "--threads must be positive"
    threads = requested_threads
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(Path(python).parent.parent)
    env["PATH"] = str(Path(python).parent) + os.pathsep + env["PATH"]
    if "PYTHONHOME" in env:
        del env["PYTHONHOME"]
    for key in ("TABBENCH_BIO_RESOURCE_TIER", "TABBENCH_RESOURCE_PROFILE"):
        if key in env:
            del env[key]
    visible = env["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in env else "0"
    env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else visible.split(",")[0]
    for key in (
        "TABBENCH_MODEL_CPUS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        env[key] = str(threads)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    # Probe in a child: CUDA visibility and thread limits must precede library imports.
    probe = subprocess.run(
        [
            python,
            "-c",
            "import json,torch; print(json.dumps({'gpu': torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else None}))",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    hardware = json.loads(probe.stdout)
    hardware.update(threads=threads, available_threads=available)
    env["TABBENCH_MODEL_HARDWARE"] = json.dumps(hardware)
    allocation_matches = available == 18 if "SLURM_CPUS_PER_TASK" in env else available >= 32
    if (
        not allocation_matches
        or threads != 32
        or hardware["gpu"] is None
        or "L40S" not in hardware["gpu"].upper()
    ):
        print(
            f"WARNING: using {threads} model threads on {available} available CPUs "
            f"and {hardware['gpu'] or 'no GPU'}. "
            "The TCML reference uses 18 allocated CPUs, 32 model threads and one NVIDIA L40S; results, "
            "especially timings and time-limited fits, are less comparable.",
            file=sys.stderr,
            flush=True,
        )
    if device == "gpu":
        assert hardware["gpu"] is not None, (
            "This model requires a GPU. Allocate one, or pass --device cpu to try CPU execution."
        )
    return env, hardware


def run_model(args) -> None:
    key = args.model_key.upper()
    assert re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]*", key), f"Invalid model key: {key}"
    grid_path = CHECKOUT / "configs/grid_sweep_all.json"
    assert grid_path.is_file(), "Use an editable clone: uv pip install -e '.[bio,autogluon]'"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    roster = resolve_list(grid["models"], str(grid_path.parent))
    entries = {entry["key"]: entry for entry in roster}
    assert key in entries or key in CUSTOM_MODELS, (
        f"Unknown model {key}. Add its adapter to models/custom.py."
    )
    entry = dict(entries[key]) if key in entries else {"key": key}
    if key in CUSTOM_MODELS:
        entry.update(CUSTOM_MODELS[key])
    assert "environment" in entry, f"Model {key} must declare an environment profile"
    profile = entry["environment"]
    python = model_python(profile)
    device = args.device if args.device else entry["device"]
    assert device in ("cpu", "gpu"), f"Invalid device for {key}: {device}"
    print(f"{key}: environment {profile} ({python})", flush=True)
    env, hardware = worker_environment(device, args.threads, python)
    hardware.update(environment=profile, python=python)
    env["TABBENCH_MODEL_HARDWARE"] = json.dumps(hardware)
    # Resolve the adapter before downloading datasets or creating result files.
    subprocess.run(
        [
            python,
            "-c",
            "import sys; from tabbench_bio.model import _resolve_hyperparameters; "
            "hp={} if sys.argv[1]=='AUTOGLUON' else "
            "_resolve_hyperparameters([sys.argv[1]],int(sys.argv[2])); "
            "assert all(not isinstance(k,str) for k in hp), 'Unknown AutoGluon model'",
            key,
            str(int(device == "gpu")),
        ],
        env=env,
        check=True,
    )
    root = Path(args.output or f"results/{key.lower()}").resolve()
    cache = args.cache_dir
    if not cache and "TABBENCH_CACHE_DIR" in env:
        cache = env["TABBENCH_CACHE_DIR"]
    cache = str(Path(cache or ".cache/grid_all").resolve())
    settings = {
        "datasets": resolve_list(grid["datasets"], str(grid_path.parent)),
        "datasets_regression": resolve_list(grid["datasets_regression"], str(grid_path.parent)),
        "models": [key],
        "limits": model_limits([entry]),
        "overrides": model_overrides([entry]),
        "n_rep": 1,
        "cv_folds": grid["cv_folds"],
        "time_limit": grid["time_limit"],
        "cache_dir": cache,
        "test_size": grid["test_size"],
        "random_state": grid["random_state"],
        "min_samples_per_class": grid["min_samples_per_class"],
    }
    caps, samples = (grid["caps"], grid["samples"]) if args.full_grid else ([10000], [100])
    cells = []
    for cap in caps:
        for samples_per_fold in samples:
            name = cell_name(cap, samples_per_fold)
            output = root / name
            config = config_for_cell(
                cap,
                samples_per_fold,
                out_dir=str(output),
                **settings,
            )
            path = output / "config.json"
            if path.is_file():
                frozen = json.loads(path.read_text(encoding="utf-8"))
                assert {**frozen, "cache_dir": cache} == config, (
                    f"Settings changed for {name}; use a new --output directory"
                )
                config = frozen
            cells.append((path, config))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "hardware.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), **hardware}) + "\n")
    print(f"{key}: {len(cells)} cells, five folds, one fit at a time. Results: {root}", flush=True)
    for index, (path, config) in enumerate(cells, 1):
        if not path.exists():
            atomic_write_json(path, config)
        print(f"[{index}/{len(cells)}] {path.parent.name}", flush=True)
        subprocess.run(
            [
                python,
                "-m",
                "tabbench_bio.cli",
                "run",
                "--config",
                str(path),
                "--cache-dir",
                cache,
            ],
            env=env,
            check=True,
        )
    database = consolidate_results(root)
    status = ResultRepository.from_root(root).current_frame()
    print(status["status"].value_counts().to_string() if not status.empty else "No result units.")
    print(
        f"Database: {database}\nMerge this file with the reference results to compute anchored Elo."
    )
