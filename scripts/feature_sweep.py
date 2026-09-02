"""Feature x sample sweep — model performance across feature caps AND training-set sizes.

Runs the benchmark over a 2-D grid: ``bio_max_features`` (the feature cap) crossed with
``train_subsample`` (the number of stratified training rows). Both are fit on the training
split only and applied to a fixed held-out test set, so every cell is a leak-free
learning-curve point (``docs/integrity_review.md``). ``train_subsample = null`` reuses the
1-D feature sweep's full-sample cells (``cap_<cap>/``).

All knobs live in ``configs/grid_sweep.json``. ``datasets`` and ``models`` each take an
inline list or a path to a JSON array file (relative to the config). Schema::

    {
      "caps":     [2000, 10000, 25000, null],       # null = keep every feature
      "samples":  [20, 50, 100, 200, 500, null],    # null = full training set
      "datasets": "datasets/bio_classification.json",   # JSON path or inline bio-id list
      "models":   "models/all.json",                    # JSON path or inline key list
      "cv_folds": 5, "time_limit": 300, "test_size": 0.2, "random_state": 42,
      "min_samples_per_class": 10, "output": "results/feature_sweep", "cache_dir": ".cache"
    }

``cv_folds`` runs every cell x dataset as its own stratified k-fold CV (k splits/cell); the
fold partition seed is the CV *repeat* (not the cell), so all cells share one partition and
the learning curve varies only the feature/sample budget on a fixed test set. Set
``cv_folds`` to ``null`` for the legacy repeated-holdout sweep with ``n_rep`` repetitions per cell.

Parallelism: the model list is split by device tag (``configs/models/all.json``) and streamed
through three concurrent pools — GPU ``solo`` models (memory-heavy foundation models) pinned
one-per-GPU so they never share VRAM, the remaining light GPU models packed several per GPU,
and CPU-tagged models in a separate GPU-free pool — so GPUs and idle cores saturate at once
without a co-tenant fit OOMing or silently degrading a foundation model's context. Each
``(cell, seed, tier)`` unit is leak-free and disjoint on disk (split-cache key folds in cap,
train_subsample and seed), so a crashed unit just leaves a gap a rerun resumes. GPUs come from
the environment (``CUDA_VISIBLE_DEVICES`` / ``SLURM_GPUS_ON_NODE``, or ``NUM_GPUS`` to force a
count); ``GPU_WORKERS_PER_DEVICE`` packs several *shared* (non-solo) units per GPU;
``CPU_POOL_WORKERS`` sets the CPU pool width. A cell's metrics run once (CPU-only) after every
pool finishes its seeds. Tag a model ``"solo": true`` in the model list to give it a whole GPU.

Usage (from repo root, with the project venv)::

    .venv/bin/python scripts/feature_sweep.py --grid-config configs/grid_sweep.json
    .venv/bin/python scripts/feature_sweep.py --grid-config configs/grid_sweep.json --skip-run

Every knob comes from the grid-config (or an explicit CLI flag, which overrides it); a knob
set by neither is a fatal error — there are no code defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from tabbench_bio.config import model_limits, model_overrides, parse_models, resolve_list
from tabbench_bio.io_utils import atomic_to_csv, atomic_write_json
from tabbench_bio.result_fork import fork_grid_results
from tabbench_bio.sample_fallback import resolve_sample_fallbacks


def log(msg: str) -> None:
    """Print one wall-clock-stamped line, matching the model log's timestamp format."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def _fmt_dur(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))

# Metrics summarised per cell; the site picks any of them for the Elo/surface plots.
# f1_macro is the primary ranking metric (tabbench_bio.metrics.PRIMARY_CLF_METRIC).
REPORT_METRICS = ["f1_macro", "balanced_accuracy", "matthews_corrcoef", "f1_score", "roc_auc"]


def _cell_dirname(cap: int | None, n_train: int | None) -> str:
    """Per-cell results dir. Full-sample cells reuse the 1-D sweep's ``cap_<cap>/``;
    an uncapped feature run (``cap is None``) lives under ``cap_full/``."""
    cap_s = "full" if cap is None else str(cap)
    return f"cap_{cap_s}" if n_train is None else f"cap_{cap_s}_n{n_train}"


def _models_for_cell(models: list, model_cells: dict[str, list[str]], cap, n_train) -> list:
    """Return the roster allowed at one operating point.

    Models absent from ``model_cells`` run across the full grid. A listed model runs only in
    the explicitly named cell directories. This keeps resource-intensive references sparse
    without hard-coding a model name or silently projecting one cell's score onto another.
    """
    cell = _cell_dirname(cap, n_train)
    return [
        entry
        for entry in models
        if (entry if isinstance(entry, str) else entry["key"]) not in model_cells
        or cell in model_cells[entry if isinstance(entry, str) else entry["key"]]
    ]


def _config_for_cell(
    cap, n_train, *, datasets, datasets_regression, models, limits, overrides, n_rep, cv_folds,
    time_limit, out_dir, cache_dir, test_size, random_state, min_samples_per_class,
):
    # cv_folds set => stratified k-fold per cell x dataset (n_repetitions pinned to 1);
    # None => legacy holdout with n_rep repetitions.
    return {
        "datasets_classification": datasets,
        "datasets_regression": datasets_regression,
        "test_size": test_size,
        "n_repetitions": 1 if cv_folds is not None else n_rep,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "cache_dir": cache_dir,
        "output_dir": out_dir,
        "models": models,
        "model_limits": limits,
        "model_overrides": overrides,
        "autogluon_time_limit": time_limit,
        # Run-wide fitting regime: one library-default configuration per model, no HPO, no
        # bagging. Roster entries may override it (see config.model_overrides); AUTOGLUON does,
        # which is why it is reported as a best-case AutoML reference and not a ranked peer.
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "num_hpo_trials": 0,
        "min_samples_per_class": min_samples_per_class,
        "group_regression_splits": False,
        "bio_max_features": cap,
        "max_classes": None,
        # Sample axis: cap TRAINING rows (stratified, train-only). null = all rows.
        "train_subsample": n_train,
        "subsample": None,
        # KNN's AutoGluon preprocessor drops every column carrying a NaN, which on the sparse
        # metagenomic abundance matrices leaves it nothing to fit ("No valid features to train
        # KNeighbors"); an explicit train-fitted median makes the unit measure the model.
        "nan_policy": {"default": "native", "KNN": "median"},
        "exclude_keys": [],
        "exclude_datasets": [],
        "exclude_targets": [],
    }


def _gpu_devices() -> list[str]:
    """Device tokens for ``CUDA_VISIBLE_DEVICES`` pinning, read from the environment (never
    importing torch, so the parent stays CUDA-free). ``NUM_GPUS`` forces a count, else the
    exported CUDA tokens, then ``SLURM_GPUS_ON_NODE``, else a single (possibly CPU-only) device."""
    if os.environ.get("NUM_GPUS"):
        return [str(i) for i in range(int(os.environ["NUM_GPUS"]))]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible.split(",")
    n = os.environ.get("SLURM_GPUS_ON_NODE")
    return [str(i) for i in range(int(n))] if n else ["0"]


def _write_cell_config(
    cap, n_train, *, datasets, datasets_regression, models, n_rep, cv_folds, time_limit,
    out_root, cache_dir, test_size, random_state, min_samples_per_class,
) -> dict:
    """Write a grid cell's configs and return its scheduling spec.

    Four configs share one ``out_dir``: ``config.json`` (full model list — drives the
    once-per-cell metrics step) plus the device/tier subsets the pools fit in parallel, each
    written only when non-empty — ``config_gpu_solo.json`` (memory-heavy models pinned
    one-per-GPU), ``config_gpu_shared.json`` (light GPU models packed several per GPU) and
    ``config_cpu.json`` (the GPU-free pool). ``n_splits`` is the CV fold count when
    ``cv_folds`` is set, else the holdout repetition count.
    """
    out_dir = os.path.join(out_root, _cell_dirname(cap, n_train))
    os.makedirs(out_dir, exist_ok=True)

    # A result directory defines a frozen experiment. Registry/model-list growth after the
    # run began must not rewrite its configs and silently mix new tasks into old outputs.
    existing_full_path = os.path.join(out_dir, "config.json")
    if os.path.isfile(existing_full_path):
        config_paths = {
            "full_cfg": existing_full_path,
            "gpu_solo_cfg": os.path.join(out_dir, "config_gpu_solo.json"),
            "gpu_shared_cfg": os.path.join(out_dir, "config_gpu_shared.json"),
            "cpu_cfg": os.path.join(out_dir, "config_cpu.json"),
        }
        with open(existing_full_path, encoding="utf-8") as handle:
            frozen = json.load(handle)
        assert frozen["bio_max_features"] == cap, (
            f"Existing cell {out_dir} has feature cap {frozen['bio_max_features']}, not {cap}."
        )
        assert frozen["train_subsample"] == n_train, (
            f"Existing cell {out_dir} has sample cap {frozen['train_subsample']}, not {n_train}."
        )
        assert os.path.normpath(frozen["output_dir"]) == os.path.normpath(out_dir), (
            f"Existing cell config points to {frozen['output_dir']}, not {out_dir}."
        )

        tier_models = []
        for key in ("gpu_solo_cfg", "gpu_shared_cfg", "cpu_cfg"):
            path = config_paths[key]
            if not os.path.isfile(path):
                config_paths[key] = None
                continue
            with open(path, encoding="utf-8") as handle:
                tier = json.load(handle)
            for invariant in ("bio_max_features", "train_subsample", "output_dir"):
                assert tier[invariant] == frozen[invariant], (
                    f"Frozen tier config {path} differs from config.json on {invariant}."
                )
            tier_models.extend(tier["models"])
        assert len(tier_models) == len(set(tier_models)), (
            f"Frozen tier configs under {out_dir} schedule a model more than once."
        )
        assert set(tier_models) == set(frozen["models"]), (
            f"Frozen tier configs under {out_dir} do not partition config.json models."
        )

        requested_pairs = parse_models(models)
        requested = _config_for_cell(
            cap,
            n_train,
            datasets=datasets,
            datasets_regression=datasets_regression,
            models=[key for key, *_ in requested_pairs],
            limits=model_limits(models),
            overrides=model_overrides(models),
            n_rep=n_rep,
            cv_folds=cv_folds,
            time_limit=time_limit,
            out_dir=out_dir,
            cache_dir=cache_dir,
            test_size=test_size,
            random_state=random_state,
            min_samples_per_class=min_samples_per_class,
        )
        drift = sorted(
            key for key in set(frozen) | set(requested) if frozen.get(key) != requested.get(key)
        )
        if drift:
            log(
                f"  resume {os.path.basename(out_dir)} from frozen configs; ignoring current "
                f"run-definition drift in: {', '.join(drift)}"
            )
        frozen_splits = frozen["cv_folds"] or frozen["n_repetitions"]
        label = "full" if n_train is None else f"n={n_train}"
        cap_label = "full" if cap is None else cap
        return {
            **config_paths,
            "n_splits": frozen_splits,
            "tag": f"cap {cap_label}/{label}",
        }

    pairs = parse_models(models)
    limits = model_limits(models)  # per-model size skips; same map in every tier's config
    overrides = model_overrides(models)  # per-model fitting-regime opt-outs (AutoGluon reference)
    gpu_solo = [k for k, dev, solo in pairs if dev == "gpu" and solo]
    gpu_shared = [k for k, dev, solo in pairs if dev == "gpu" and not solo]
    cpu_models = [k for k, dev, _ in pairs if dev == "cpu"]

    def _dump(name, model_keys):
        path = os.path.join(out_dir, name)
        payload = _config_for_cell(
            cap,
            n_train,
            datasets=datasets,
            datasets_regression=datasets_regression,
            models=model_keys,
            limits=limits,
            overrides=overrides,
            n_rep=n_rep,
            cv_folds=cv_folds,
            time_limit=time_limit,
            out_dir=out_dir,
            cache_dir=cache_dir,
            test_size=test_size,
            random_state=random_state,
            min_samples_per_class=min_samples_per_class,
        )
        atomic_write_json(path, payload)
        return path

    full_cfg = _dump("config.json", [k for k, *_ in pairs])
    gpu_solo_cfg = _dump("config_gpu_solo.json", gpu_solo) if gpu_solo else None
    gpu_shared_cfg = _dump("config_gpu_shared.json", gpu_shared) if gpu_shared else None
    cpu_cfg = _dump("config_cpu.json", cpu_models) if cpu_models else None

    label = "full" if n_train is None else f"n={n_train}"
    cap_lbl = "full" if cap is None else cap
    return {"full_cfg": full_cfg, "gpu_solo_cfg": gpu_solo_cfg,
            "gpu_shared_cfg": gpu_shared_cfg, "cpu_cfg": cpu_cfg,
            "n_splits": cv_folds if cv_folds is not None else n_rep,
            "tag": f"cap {cap_lbl}/{label}"}


def _cli(cfg_path: str, *args: str) -> list[str]:
    return [sys.executable, "-m", "tabbench_bio.cli", "run", "--config", cfg_path, *args]


def _cell_metrics(cfg_path: str, tag: str):
    """Compute one cell's metrics (CPU-only, idempotent, aggregates across its seeds)."""
    subprocess.run(_cli(cfg_path, "--step", "metrics"), check=True)


class _GpuAllocator:
    """Weighted per-device slot allocator for the grid's GPU pool.

    Each device has ``capacity`` (= ``GPU_WORKERS_PER_DEVICE``) slots. A *shared* unit takes
    one slot, so up to ``capacity`` light GPU models pack a device; a *solo* unit takes the
    whole device (all ``capacity`` slots), so a memory-heavy foundation model never shares
    VRAM with a co-tenant fit. ``solo_workers`` further caps how many solo units run
    concurrently across *all* devices (default = one per device); lowering it runs the heavy
    foundation models (MITRA, ...) fewer-at-a-time to ease host-RAM pressure, leaving the
    spare GPUs to the shared tier. Solo requests take priority — new shared acquisitions wait
    while a queued solo unit could still start — so a steady trickle of shared units can't
    starve a solo unit of a fully-free GPU, yet shared units keep packing GPUs once the solo
    cap is saturated. Progress is guaranteed: running units are never blocked from releasing,
    so a waiting solo unit always drains its device eventually."""

    def __init__(self, devices, capacity, solo_workers=None):
        self._free = {d: capacity for d in devices}
        self._cap = capacity
        self._solo_cap = len(devices) if solo_workers is None else solo_workers
        self._solo_running = 0
        self._solo_waiting = 0
        self._cv = threading.Condition()

    def acquire(self, solo):
        """Block until a device is free, claim it, and return its ``CUDA_VISIBLE_DEVICES`` token."""
        with self._cv:
            if solo:
                self._solo_waiting += 1
                try:
                    while True:
                        if self._solo_running < self._solo_cap:  # under the concurrent-solo cap
                            for d in self._free:
                                if self._free[d] == self._cap:  # whole GPU free
                                    self._free[d] = 0
                                    self._solo_running += 1
                                    return d
                        self._cv.wait()
                finally:
                    self._solo_waiting -= 1
            while True:
                # Yield a fully-free GPU only to a solo unit that could actually start now;
                # once the solo cap is saturated a queued solo can't run, so pack shared units.
                solo_could_start = self._solo_waiting > 0 and self._solo_running < self._solo_cap
                if not solo_could_start:
                    for d in self._free:
                        if self._free[d] >= 1:
                            self._free[d] -= 1
                            return d
                self._cv.wait()

    def release(self, dev, solo):
        with self._cv:
            self._free[dev] += self._cap if solo else 1
            if solo:
                self._solo_running -= 1
            self._cv.notify_all()


def run_grid_parallel(cell_specs, *, devices, workers_per_device, solo_workers,
                      gpu_worker_cpus, cpu_workers, cpu_worker_cpus):
    """Stream the grid through three concurrent pools so GPUs and CPUs saturate at once, then
    finalize each cell's metrics the instant its last unit lands.

    GPU units flow through a weighted :class:`_GpuAllocator`: ``solo``-tagged (memory-heavy)
    models take a whole GPU (at most ``solo_workers`` concurrently), while the remaining light
    GPU models pack ``workers_per_device`` per device — so foundation models never share VRAM
    (no OOM/silent-context-degradation) yet the cheap GPU models still parallelize. Lowering
    ``solo_workers`` runs the heavy models fewer-at-a-time. CPU-tagged models flow through a separate pool
    of ``cpu_workers`` GPU-free workers, so they never hold a GPU hostage. Each
    ``(cell, seed, tier)`` unit writes its own disjoint split-cache and predictions, so the
    pools never collide and a failed unit just leaves a gap a rerun resumes. A cell's metrics
    (CPU-only) run once its ``n_splits`` units in every non-empty pool finish."""
    from concurrent.futures import ThreadPoolExecutor

    alloc = _GpuAllocator(devices, workers_per_device, solo_workers)

    # A cell needs every split of every non-empty pool done before its metrics run.
    remaining = {
        s["full_cfg"]: s["n_splits"] * (
            bool(s["gpu_solo_cfg"]) + bool(s["gpu_shared_cfg"]) + bool(s["cpu_cfg"]))
        for s in cell_specs
    }
    lock = threading.Lock()

    def _finish(full_cfg, tag):
        with lock:
            remaining[full_cfg] -= 1
            cell_done = remaining[full_cfg] == 0
        if cell_done:
            log(f"  metrics {tag} (all pools/seeds done)")
            _cell_metrics(full_cfg, tag)

    def run_gpu(spec, seed_index, cfg_key, solo):
        dev = alloc.acquire(solo)  # blocks until a GPU (whole device if solo) frees
        kind = "GPU-solo" if solo else "GPU"
        try:
            log(f"  start {kind} {spec['tag']} seed{seed_index} -> GPU {dev}")
            t0 = time.monotonic()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev,
                   "SLURM_CPUS_PER_TASK": str(gpu_worker_cpus)}
            subprocess.run(
                _cli(spec[cfg_key], "--step", "predictions", "--seed-index", str(seed_index)),
                env=env,
                check=True,
            )
            log(f"  done  {kind} {spec['tag']} seed{seed_index} -> GPU {dev}: "
                f"ok in {_fmt_dur(time.monotonic() - t0)}")
        finally:
            alloc.release(dev, solo)
        _finish(spec["full_cfg"], spec["tag"])

    def run_cpu(spec, seed_index):
        log(f"  start CPU {spec['tag']} seed{seed_index}")
        t0 = time.monotonic()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
               "SLURM_CPUS_PER_TASK": str(cpu_worker_cpus)}
        subprocess.run(
            _cli(spec["cpu_cfg"], "--step", "predictions", "--seed-index", str(seed_index)),
            env=env,
            check=True,
        )
        log(f"  done  CPU {spec['tag']} seed{seed_index}: "
            f"ok in {_fmt_dur(time.monotonic() - t0)}")
        _finish(spec["full_cfg"], spec["tag"])

    solo_units = [(s, i, "gpu_solo_cfg", True)
                  for s in cell_specs if s["gpu_solo_cfg"] for i in range(s["n_splits"])]
    shared_units = [(s, i, "gpu_shared_cfg", False)
                    for s in cell_specs if s["gpu_shared_cfg"] for i in range(s["n_splits"])]
    cpu_units = [(s, i) for s in cell_specs if s["cpu_cfg"] for i in range(s["n_splits"])]

    # Both pools run at the same time: submit to each executor, then wait for all of it. Solo
    # units are submitted first so they claim whole GPUs before shared units pack the rest.
    # Keep extra threads for queued solo waiters so they cannot starve shared-tier dispatch.
    gpu_max = max(1, len(devices) * workers_per_device + solo_workers)
    with ThreadPoolExecutor(max_workers=gpu_max) as gpu_ex, \
            ThreadPoolExecutor(max_workers=cpu_workers) as cpu_ex:
        futures = [gpu_ex.submit(run_gpu, *u) for u in solo_units + shared_units]
        futures += [cpu_ex.submit(run_cpu, *u) for u in cpu_units]
        for fut in futures:
            fut.result()


def aggregate(out_root, cells):
    """Build strict and memory-adaptive views without changing any cell artifact."""
    names = [_cell_dirname(cap, n_train) for cap, n_train in cells]
    strict, adaptive, manifest = resolve_sample_fallbacks(out_root, names)
    if all(frame.empty for frame in adaptive.values()):
        raise SystemExit("No metrics found to aggregate. Run the sweep first (omit --skip-run).")
    return strict, adaptive, manifest


def _write_summary(raw: pd.DataFrame, path: str) -> None:
    if raw.empty:
        atomic_to_csv(pd.DataFrame(), path, index=False)
        return
    metrics = [metric for metric in REPORT_METRICS if metric in raw.columns]
    group = ["dataset", "model", "max_features", "n_train"]
    summary = raw.groupby(group, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["__".join(column).rstrip("_") for column in summary.columns]
    atomic_to_csv(summary, path, index=False)


def summarise(strict: dict, adaptive: dict, manifest: pd.DataFrame, out_root: str):
    """Write machine-readable CSVs only (no rendered tables) for later aggregation/plotting.

    Existing filenames remain the adaptive classification view for downstream compatibility.
    Strict files preserve the original completed-cell outcomes, and ``sample_fallbacks.csv``
    records every resolved or unresolved memory fallback. Per-cell files remain read-only.
    """
    outputs = {
        "sweep_metrics.csv": adaptive["classification"],
        "sweep_metrics_strict.csv": strict["classification"],
        "sweep_metrics_regression.csv": adaptive["regression"],
        "sweep_metrics_regression_strict.csv": strict["regression"],
        "sample_fallbacks.csv": manifest,
    }
    for filename, frame in outputs.items():
        atomic_to_csv(frame, os.path.join(out_root, filename), index=False)
    _write_summary(adaptive["classification"], os.path.join(out_root, "sweep_summary.csv"))
    _write_summary(strict["classification"], os.path.join(out_root, "sweep_summary_strict.csv"))
    log(
        f"Wrote adaptive/strict sweep metrics and {len(manifest)} sample-fallback record(s) "
        f"under {out_root}"
    )


def _norm_axis(values) -> list[int | None]:
    """Normalise a grid axis: ``null`` / 'full' / 'none' / '' -> None (full set / no cap), else int."""
    out: list[int | None] = []
    for v in values:
        if v is None or (isinstance(v, str) and v.lower() in ("full", "none", "null", "")):
            out.append(None)
        else:
            out.append(int(v))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-config", default=None,
                   help="JSON config holding the whole grid (caps/samples/datasets/models/...). "
                        "Explicit CLI flags below override individual keys.")
    p.add_argument("--caps", nargs="+", default=None,
                   help="feature caps; use 'full' (or 'none') for no cap (every feature kept)")
    p.add_argument("--samples", nargs="+", default=None,
                   help="training-set sizes; use 'full' (or 'none') for all rows")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="classification datasets (inline ids or a JSON-array path)")
    p.add_argument("--datasets-regression", nargs="+", default=None,
                   help="regression datasets (inline ids or a JSON-array path); default none")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n-rep", type=int, default=None, help="holdout repetitions per cell (cv_folds=null)")
    p.add_argument("--cv-folds", type=int, default=None,
                   help="stratified k-fold CV per cell x dataset; k splits/cell (overrides holdout)")
    p.add_argument("--time-limit", type=int, default=None, help="AutoGluon time_limit per cell (s)")
    p.add_argument("--output", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--test-size", type=float, default=None)
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument("--min-samples-per-class", type=int, default=None)
    p.add_argument("--skip-run", action="store_true", help="aggregate existing results only")
    p.add_argument("--prepare-only", action="store_true",
                   help="warm the shared raw-dataset cache, then exit without fitting")
    p.add_argument("--skip-prepare", action="store_true",
                   help="skip raw-cache warming (use only after a successful --prepare-only job)")
    p.add_argument("--shard-count", type=int, default=1,
                   help="split grid cells across this many collision-free workers")
    p.add_argument("--shard-index", type=int, default=0,
                   help="zero-based worker index when --shard-count is greater than one")
    args = p.parse_args()

    gc = {}
    config_dir = ""
    if args.grid_config:
        config_dir = os.path.dirname(os.path.abspath(args.grid_config))
        with open(args.grid_config) as f:
            gc = json.load(f)

    def pick(name, cli):
        """CLI flag wins, else the grid-config key; set by neither is fatal — no code defaults."""
        if cli is not None:
            return cli
        assert name in gc, (
            f"'{name}' is unset: pass --{name.replace('_', '-')} or add it to --grid-config"
        )
        return gc[name]

    caps = _norm_axis(pick("caps", args.caps))
    samples = _norm_axis(pick("samples", args.samples))
    datasets = resolve_list(pick("datasets", args.datasets), config_dir)
    datasets_regression = resolve_list(pick("datasets_regression", args.datasets_regression), config_dir)
    models = resolve_list(pick("models", args.models), config_dir)
    # Resource-intensive references can be restricted to named operating points. Models not
    # listed here retain the ordinary full-grid policy.
    model_cells = gc.get("model_cells", {})
    assert isinstance(model_cells, dict), "'model_cells' must map model keys to cell-name lists"
    roster_keys = {entry if isinstance(entry, str) else entry["key"] for entry in models}
    unknown_restricted_models = set(model_cells) - roster_keys
    assert not unknown_restricted_models, (
        f"'model_cells' names models outside the roster: {sorted(unknown_restricted_models)}"
    )
    for model, allowed_cells in model_cells.items():
        assert isinstance(allowed_cells, list) and allowed_cells, (
            f"'model_cells[{model}]' must be a non-empty list"
        )
        assert len(allowed_cells) == len(set(allowed_cells)), (
            f"'model_cells[{model}]' contains duplicate cells"
        )
    # cv_folds set => stratified k-fold (k units/cell); null => legacy holdout with n_rep reps
    # (n_rep required only in that mode).
    cv_folds = pick("cv_folds", args.cv_folds)
    n_rep = pick("n_rep", args.n_rep) if cv_folds is None else None
    n_splits = cv_folds if cv_folds is not None else n_rep
    time_limit = pick("time_limit", args.time_limit)
    output = pick("output", args.output)
    cache_dir = pick("cache_dir", args.cache_dir)
    test_size = pick("test_size", args.test_size)
    random_state = pick("random_state", args.random_state)
    mspc = pick("min_samples_per_class", args.min_samples_per_class)

    cells = [(cap, n) for cap in caps for n in samples]
    configured_cell_names = {_cell_dirname(cap, n) for cap, n in cells}
    for model, allowed_cells in model_cells.items():
        unknown_cells = set(allowed_cells) - configured_cell_names
        assert not unknown_cells, (
            f"'model_cells[{model}]' names cells outside the configured grid: "
            f"{sorted(unknown_cells)}"
        )
    if args.shard_count < 1:
        p.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        p.error("--shard-index must satisfy 0 <= index < shard-count")
    if args.prepare_only and args.skip_run:
        p.error("--prepare-only and --skip-run are mutually exclusive")
    shard_cells = [cell for i, cell in enumerate(cells) if i % args.shard_count == args.shard_index]
    if not shard_cells:
        p.error("this shard has no grid cells; reduce --shard-count")
    devices = _gpu_devices()
    workers_per_device = max(1, int(os.environ.get("GPU_WORKERS_PER_DEVICE", "1")))
    # Cap concurrent solo (whole-GPU foundation-model) units across all devices; default one
    # per GPU. Lower it to run MITRA/... fewer-at-a-time and ease host-RAM pressure.
    solo_workers = max(1, int(os.environ.get("GPU_SOLO_WORKERS", str(len(devices)))))
    gpu_slots = len(devices) * workers_per_device
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

    # Two concurrent pools share the node's cores; CPU_POOL_WORKERS sets the CPU pool width
    # (default ~1 per 16 cores). Cores split evenly across all workers so the node is not
    # oversubscribed.
    pairs = parse_models(models)
    n_gpu_solo = sum(dev == "gpu" and solo for _, dev, solo in pairs)
    n_gpu_shared = sum(dev == "gpu" and not solo for _, dev, solo in pairs)
    n_cpu_models = sum(dev == "cpu" for _, dev, _ in pairs)
    cpu_workers = max(1, int(os.environ.get("CPU_POOL_WORKERS", str(max(1, total_cpus // 16)))))
    per_worker_cpus = max(1, total_cpus // (gpu_slots + cpu_workers))
    split_lbl = f"{cv_folds}-fold CV" if cv_folds is not None else f"{n_rep} holdout(s)"
    print(f"Grid: {len(caps)} cap(s) x {len(samples)} sample size(s) = {len(cells)} cells "
          f"over {len(datasets)}+{len(datasets_regression)} clf+reg dataset(s) x "
          f"{len(models)} model(s) ({n_gpu_solo} GPU-solo + {n_gpu_shared} GPU-shared + "
          f"{n_cpu_models} CPU) x {split_lbl} ({n_splits} split(s)/cell) | "
          f"GPU pool: {len(devices)} GPU(s), solo={solo_workers} concurrent (1/GPU), "
          f"shared={workers_per_device}/GPU ({gpu_slots} shared slot(s)); "
          f"CPU pool: {cpu_workers} worker(s); {per_worker_cpus} CPU(s) each.",
          flush=True)

    os.makedirs(output, exist_ok=True)
    if not args.skip_run:
        kwargs = dict(datasets=datasets, datasets_regression=datasets_regression,
                      n_rep=n_rep, cv_folds=cv_folds, time_limit=time_limit,
                      out_root=output, cache_dir=cache_dir, test_size=test_size,
                      random_state=random_state, min_samples_per_class=mspc)

        # Grid cells are the unit of sharding. Different workers therefore write to disjoint
        # cap_<cap>[_n<samples>] directories while sharing only the pre-warmed read-only raw cache.
        cell_specs = [
            _write_cell_config(
                cap,
                n,
                models=_models_for_cell(models, model_cells, cap, n),
                **kwargs,
            )
            for cap, n in shard_cells
        ]

        reuse_results_from = gc.get("reuse_results_from")
        if reuse_results_from:
            marker = fork_grid_results(reuse_results_from, output)
            log(
                f"Reused frozen artifacts from {marker['source']} "
                f"({marker['files']}) without modifying the source run."
            )

        if not args.skip_prepare:
            # Warm the raw dataset caches once, single-threaded, so parallel workers don't race
            # to build them (raw caches are seed/cap-independent; split caches are per-unit).
            log("Warming raw dataset cache (single-threaded) ...")
            subprocess.run(
                _cli(cell_specs[0]["full_cfg"], "--step", "prepare", "--seed-index", "0"),
                check=True,
            )
        if args.prepare_only:
            log("Raw dataset cache ready; --prepare-only requested, stopping before fits.")
            return

        run_grid_parallel(cell_specs, devices=devices,
                          workers_per_device=workers_per_device, solo_workers=solo_workers,
                          gpu_worker_cpus=per_worker_cpus, cpu_workers=cpu_workers,
                          cpu_worker_cpus=per_worker_cpus)

        # A shard cannot aggregate until all sibling shards have finished. A dependent finalizer
        # runs this script with --skip-run and no sharding to validate and aggregate every cell.
        if args.shard_count > 1:
            log(f"Shard {args.shard_index + 1}/{args.shard_count} complete; aggregation deferred.")
            return

    strict, adaptive, manifest = aggregate(output, cells)
    summarise(strict, adaptive, manifest, output)


if __name__ == "__main__":
    main()
