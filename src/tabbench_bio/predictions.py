"""Prediction generation step for the benchmark pipeline (Step 1).

Trains each (model, dataset, seed) combination and saves predictions as CSV
files.  Supports resume: existing prediction files are skipped unless
``overwrite=True``.  Measures train/inference time, peak memory, GPU power
(pynvml), and CPU energy (Intel RAPL).

Usage
-----
::

    from tabbench_bio.config import load_config
    from tabbench_bio.predictions import compute_predictions

    config = load_config("configs/benchmark_v0.1.json")
    compute_predictions(config)
"""

import gc
import json
import logging
import os
import random
import shutil
import signal
import threading
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tabbench_bio.benchmark import configure_benchmark
from tabbench_bio.coverage import DESIGN_SKIPS
from tabbench_bio.dataset import TaskType
from tabbench_bio.io_utils import archive_existing, atomic_to_csv, atomic_write_json
from tabbench_bio.logging_utils import LOG_FORMAT, run_file_logger
from tabbench_bio.sample_fallback import log_has_memory_failure
from tabbench_bio.seeds import get_seeds

try:
    from tabbench_bio.model import AutoGluonModel
except ImportError:
    AutoGluonModel = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Models whose upstream package implements classification only: a regression unit for one of
# these is excluded by design (``classification_only``), not counted as a model failure.
CLASSIFICATION_ONLY_MODELS: set[str] = {"TABPFN-WIDE"}
# Bump only when memory handling changes materially. OOMs written by an older version are
# retried once; current-version OOMs remain terminal on ordinary restarts.
MEMORY_RETRY_VERSION = 1
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data hash (staleness detection)
# ---------------------------------------------------------------------------


def _data_hash(df: pd.DataFrame) -> str:
    import hashlib

    label_col = df.columns[-1]
    h = hashlib.md5()
    h.update(str(len(df)).encode())
    h.update("|".join(sorted(df[label_col].astype(str))).encode())
    return h.hexdigest()


def _assert_ground_truth_compatible(path: str, data_test: pd.DataFrame) -> None:
    """Refuse to reuse results if the persisted held-out target has changed."""
    saved = pd.read_csv(path, index_col=0).sort_index()
    expected = data_test[["target"]].sort_index()
    # CSV inference converts numeric-looking string class labels (for example
    # OpenML-1084's "1", "2", "3") to integers. Normalize only when the in-memory
    # target is genuinely string-valued; numeric regression targets retain strict
    # numeric comparison semantics.
    expected_non_null = expected["target"].dropna()
    if (
        not expected_non_null.empty
        and expected_non_null.map(lambda value: isinstance(value, str)).all()
    ):
        saved = saved.copy()
        expected = expected.copy()
        saved["target"] = saved["target"].astype("string")
        expected["target"] = expected["target"].astype("string")
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)


def _saved_predictions_are_valid(
    pred_path: str,
    proba_path: str,
    data_test: pd.DataFrame,
    task_type: TaskType,
) -> bool:
    """Validate persisted result structure before treating a pass record as complete."""
    try:
        expected_index = data_test.sort_index().index
        prediction = pd.read_csv(pred_path, index_col=0).sort_index()
        if "target" not in prediction or not prediction.index.equals(expected_index):
            return False
        if task_type == TaskType.Classification:
            probability = pd.read_csv(proba_path, index_col=0).sort_index()
            if probability.empty or not probability.index.equals(expected_index):
                return False
        return True
    except (OSError, ValueError, pd.errors.ParserError):
        return False


def _is_current_memory_failure_record(
    record: dict, current_model_limit: dict | None = None
) -> bool:
    current_oom = (
        record.get("reason") in {"fit_oom", "inference_oom"}
        and record.get("memory_retry_version") == MEMORY_RETRY_VERSION
    )
    current_prior = (
        current_model_limit is not None
        and "enforce_memory_prior" in current_model_limit
        and current_model_limit["enforce_memory_prior"]
        and record.get("reason") == "model_limit"
        and record.get("memory_prior_max_cells") == current_model_limit["max_cells"]
        and record.get("memory_prior_version") == current_model_limit["memory_prior_version"]
    )
    return current_oom or current_prior


def _memory_prior_skip_details(
    model_limits: dict,
    model_name: str,
    n_train: int,
    n_features: int,
) -> dict | None:
    """Return a calibrated pre-fit skip when this unit exceeds a model's safe boundary."""
    assert n_train > 0 and n_features > 0
    if model_name not in model_limits:
        return None
    prior = model_limits[model_name]
    # Frozen configs written before memory priors became advisory contain only
    # ``max_cells``. Treat absence of an explicit opt-in as advisory so an old run can
    # continue under newer code without turning historical observations into skips.
    if "enforce_memory_prior" not in prior or not prior["enforce_memory_prior"]:
        return None
    max_cells = prior["max_cells"]
    prior_version = prior["memory_prior_version"]
    assert isinstance(max_cells, int) and max_cells > 0
    assert isinstance(prior_version, int) and prior_version > 0
    assert prior["enforce_memory_prior"] is True
    required_cells = n_features * n_train
    if required_cells <= max_cells:
        return None
    return {
        "n_features": n_features,
        "required_cells": required_cells,
        "memory_prior_max_cells": max_cells,
        "memory_prior_version": prior_version,
        "recommended_max_n_train": max_cells // n_features,
    }


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False


class _PsutilMemoryTracker:
    _POLL_S = 0.05

    def __init__(self):
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        proc = _psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / (1024**2)
                if rss > self._peak_mb:
                    self._peak_mb = rss
            except Exception:
                break
            self._stop.wait(self._POLL_S)

    def __enter__(self):
        self._stop.clear()
        self._peak_mb = _psutil.Process().memory_info().rss / (1024**2)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


class _TracemallocMemoryTracker:
    _peak_mb: float = 0.0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *_):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_mb = peak / (1024**2)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


def _memory_tracker():
    return _PsutilMemoryTracker() if _HAS_PSUTIL else _TracemallocMemoryTracker()


# ---------------------------------------------------------------------------
# Power / energy tracking
# ---------------------------------------------------------------------------

try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _pynvml = None
    _HAS_PYNVML = False

_RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
_RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"
_HAS_RAPL = os.path.isfile(_RAPL_PATH)


def _read_rapl():
    try:
        with open(_RAPL_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return None


class _PowerTracker:
    _POLL_S = 0.1

    def __init__(self):
        self._gpu_samples: list[float] = []
        self._gpu_handle = None
        self._elapsed_s = 0.0
        self._cpu_energy_j: float | None = None
        self._stop = threading.Event()
        self._thread = None

    def _poll_gpu(self):
        while not self._stop.is_set():
            try:
                mw = _pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                self._gpu_samples.append(mw / 1000.0)
            except Exception:
                pass
            self._stop.wait(self._POLL_S)

    def __enter__(self):
        self._start = time.perf_counter()
        self._gpu_samples = []
        if _HAS_PYNVML:
            try:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                self._stop.clear()
                self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
                self._thread.start()
            except Exception:
                self._gpu_handle = None
        self._cpu_start = _read_rapl() if _HAS_RAPL else None
        return self

    def __exit__(self, *_):
        self._elapsed_s = time.perf_counter() - self._start
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if _HAS_RAPL and self._cpu_start is not None:
            end = _read_rapl()
            if end is not None:
                delta = end - self._cpu_start
                if delta < 0:
                    try:
                        with open(_RAPL_MAX_PATH) as f:
                            delta += int(f.read().strip())
                    except Exception:
                        pass
                self._cpu_energy_j = delta / 1e6

    @property
    def gpu_mean_power_w(self):
        return (
            round(sum(self._gpu_samples) / len(self._gpu_samples), 2) if self._gpu_samples else None
        )

    @property
    def gpu_energy_j(self):
        if self._gpu_samples and self._elapsed_s > 0:
            return round(sum(self._gpu_samples) / len(self._gpu_samples) * self._elapsed_s, 2)
        return None

    @property
    def cpu_energy_j(self):
        return round(self._cpu_energy_j, 2) if self._cpu_energy_j is not None else None


# ---------------------------------------------------------------------------
# Subsampling (OOM guard for specific model/dataset combinations)
# ---------------------------------------------------------------------------


def _maybe_subsample(
    data_train: pd.DataFrame,
    model_name: str,
    key: str,
    task_type,
    subsample_config: dict | None,
    seed: int,
) -> pd.DataFrame:
    """Subsample *data_train* for listed (model, dataset) pairs.

    Applies uniform spectral downsampling (``max_features``) and/or stratified
    sample subsampling (``max_samples``) as an OOM guard for large datasets.
    Returns *data_train* unchanged if the pair is not listed.
    """
    if subsample_config is None:
        return data_train

    combos = subsample_config.get("combinations", {})
    if key not in combos.get(model_name, []):
        return data_train

    label_col = data_train.columns[-1]
    feature_cols = [c for c in data_train.columns if c != label_col]
    result = data_train

    max_f = subsample_config.get("max_features")
    if max_f and len(feature_cols) > max_f:
        step = len(feature_cols) / max_f
        selected = [feature_cols[round(i * step)] for i in range(max_f)]
        result = result[selected + [label_col]]

    max_n = subsample_config.get("max_samples", 10_000)
    if len(result) > max_n:
        if task_type == TaskType.Classification:
            _, result = train_test_split(
                result, test_size=max_n, random_state=seed, stratify=result[label_col]
            )
        else:
            result = result.sample(n=max_n, random_state=seed)

    return result


# ---------------------------------------------------------------------------
# Per-model NaN handling
# ---------------------------------------------------------------------------

#: Valid per-model missing-value policies. ``native``/``none`` keep NaNs so the model
#: (or AutoGluon's internal pipeline) handles them; the rest fill from *training*
#: statistics. The benchmark's drop-high-missing + variance cap run upstream and are
#: shared across models; only this fill step is per-model.
_NAN_POLICIES = ("native", "none", "median", "mean", "zero")

#: Policy for models without a ``nan_policy`` entry: keep NaNs for AutoGluon to impute.
DEFAULT_NAN_POLICY = "native"


def _resolve_nan_policy(model_name: str, nan_policy: dict | None) -> str:
    """Pick the NaN policy for *model_name*: explicit entry, else the ``default`` key, else native."""
    if not nan_policy:
        return DEFAULT_NAN_POLICY
    return nan_policy.get(model_name, nan_policy.get("default", DEFAULT_NAN_POLICY))


def _apply_nan_policy(
    train: pd.DataFrame, test: pd.DataFrame, policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill feature NaNs per *policy*, fitting fill values on TRAIN only (no leakage).

    ``native``/``none`` is a no-op — NaNs are left for the model / AutoGluon to handle
    internally. ``median``/``mean`` fill from the training-column statistic; ``zero``
    fills with 0. The identical per-column fill is applied to *test*. Returns new frames
    (originals untouched); the label column is never imputed.
    """
    if policy in ("native", "none"):
        return train, test
    if policy not in _NAN_POLICIES:
        raise ValueError(f"Unknown nan policy {policy!r}; expected one of {_NAN_POLICIES}.")

    label_col = train.columns[-1]
    feat = [c for c in train.columns if c != label_col]
    X_train = train[feat]
    if policy == "median":
        fill = X_train.median(numeric_only=True)
    elif policy == "mean":
        fill = X_train.mean(numeric_only=True)
    else:  # "zero"
        fill = pd.Series(0.0, index=feat)

    train = train.copy()
    train[feat] = X_train.fillna(fill)
    test = test.copy()
    test_feat = [c for c in feat if c in test.columns]
    test[test_feat] = test[test_feat].fillna(fill[test_feat])
    return train, test


# ---------------------------------------------------------------------------
# Adaptive inference batching
# ---------------------------------------------------------------------------


def _is_cuda_oom(error: BaseException) -> bool:
    """Return whether *error* or its explicit cause is a CUDA OOM."""
    current: BaseException | None = error
    while current is not None:
        if "cuda out of memory" in str(current).lower():
            return True
        try:
            import torch

            if isinstance(current, torch.cuda.OutOfMemoryError):
                return True
        except ImportError:
            pass
        current = current.__cause__
    return False


def _is_stage_oom(error: BaseException, stage: str, active_log_path: str) -> bool:
    """Recognise direct CUDA OOMs and AutoGluon's generic fit wrapper."""
    return _is_cuda_oom(error) or (
        stage == "fit"
        and "no models were trained successfully" in str(error).lower()
        and log_has_memory_failure(active_log_path)
    )


def _clear_cuda_after_oom() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _predict_with_oom_batching(
    predict_fn,
    data_test: pd.DataFrame,
) -> tuple[pd.Series | pd.DataFrame, int, int]:
    """Predict in order, halving the query batch after a CUDA OOM.

    The first attempt uses the complete test set. Halving continues until a
    batch succeeds or batch size one also OOMs. Every retry remains inside the
    caller's inference time budget. Training data and model state are unchanged.
    """
    assert len(data_test) > 0, "Cannot predict an empty test set."

    batch_size = len(data_test)
    oom_retries = 0
    while True:
        chunks = []
        try:
            for start in range(0, len(data_test), batch_size):
                stop = min(start + batch_size, len(data_test))
                chunks.append(predict_fn(data_test.iloc[start:stop]))
            predictions = pd.concat(chunks)
            assert predictions.index.equals(data_test.index), (
                "Batched prediction changed the test-row index or order."
            )
            return predictions, batch_size, oom_retries
        except Exception as error:
            if not _is_cuda_oom(error) or batch_size == 1:
                raise
            previous_batch_size = batch_size
            batch_size = max(1, batch_size // 2)
            oom_retries += 1
            del chunks
            _clear_cuda_after_oom()
            logger.warning(
                "CUDA OOM during inference; reducing query batch from %d to %d (retry %d).",
                previous_batch_size,
                batch_size,
                oom_retries,
            )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def _timed():
    t = [0.0]
    start = time.perf_counter()
    try:
        yield t
    finally:
        t[0] = time.perf_counter() - start


class UnitTimeoutError(Exception):
    """A unit exceeded the run's declared per-unit wall-clock budget."""


@contextmanager
def _time_budget(seconds: int):
    """Raise :class:`UnitTimeoutError` once *seconds* of wall clock have elapsed in the block.

    The run declares one time budget for every model (``autogluon_time_limit``), but a model
    only honours it if its own fit loop checks the clock: measured over this grid, CatBoost
    stops at the budget while RealTabPFN-2.5 ran 26x past it, so the leaderboard was ranking
    models that were not given the same budget. SIGALRM enforces it on the units the model
    does not stop itself, and a unit killed this way is recorded ``fail`` / ``time_limit`` —
    the outcome ``coverage.impute_failures`` scores at chance level.
    """

    def _fire(signum, frame):
        raise UnitTimeoutError(f"exceeded the {seconds}s per-unit wall-clock budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_splits(config, seed_index: int | None = None):
    """Materialize every train/test split into the on-disk cache (no model fitting).

    Run once, single-threaded, before launching multi-GPU shard workers. The processed-split
    cache is shared by all workers (same seed + param hash) but its pickle writes are not
    concurrency safe, so the workers must only ever *read* it. Iterating the benchmark
    computes and caches each split (see ``TabBenchBio.__getitem__``); here we exhaust the
    iterator and discard the data, leaving a fully warmed cache for the parallel workers.
    """
    logger.info("=" * 60 + "\nSTEP 0: Preparing (caching) dataset splits")
    seeds = get_seeds(config)
    if seed_index is not None:
        seeds = [seeds[seed_index]]
    for seed in seeds:
        logger.info("--- Seed %s (prepare) ---", seed)
        config["random_state"] = seed
        _set_global_seeds(seed)
        benchmark = configure_benchmark(config)
        n_cached = sum(1 for _ in benchmark)
        logger.info("Cached %d split(s) for seed %s", n_cached, seed)


def compute_predictions(
    config,
    seed_index: int | None = None,
    overwrite: bool = False,
    reverse: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
):
    """Train all models and save predictions to CSV.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration.
    seed_index : int | None
        Run only this zero-based seed index (for parallel seed execution).
    overwrite : bool
        Re-run even when a prediction file already exists.
    reverse : bool
        Iterate datasets in reverse order (useful for forward+reverse parallel jobs).
    num_shards, shard_index : int
        Split the (model x dataset) grid across ``num_shards`` workers; this process runs
        only the cells with ``cell_index % num_shards == shard_index``. Used for multi-GPU
        execution — launch one worker per GPU (each pinned with ``CUDA_VISIBLE_DEVICES``)
        with the same ``num_shards`` and a distinct ``shard_index``. The cell ordering is
        deterministic, so workers partition the grid without overlap. Warm the split cache
        first (``--step prepare``) so the workers only read it (the cache is not concurrency
        safe to write). Results aggregate naturally — every worker writes into the same
        ``output_dir``, and a cell already on disk is skipped.
    """
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    logger.info("=" * 60 + "\nSTEP 1: Computing Predictions")

    mem_backend = "psutil" if _HAS_PSUTIL else "tracemalloc"
    output_dir = config["output_dir"]

    cache_dir = config["cache_dir"]
    autogluon_path = (
        os.path.join(cache_dir, "autogluon") if cache_dir else os.path.join(".cache", "autogluon")
    )
    os.makedirs(autogluon_path, exist_ok=True)

    seeds = get_seeds(config)
    if seed_index is not None:
        seeds = [seeds[seed_index]]

    model_names = config["models"]
    autogluon_time_limit = config["autogluon_time_limit"]
    autogluon_presets = config["autogluon_presets"]
    optimize = config["optimize"]
    ensemble = config["ensemble"]
    num_hpo_trials = config["num_hpo_trials"]
    subsample_config = config["subsample"]
    model_size_limits = config["model_limits"]
    model_overrides = config["model_overrides"]
    # Grid sample-axis budget for this cell; used to recognise cells whose budget exceeds
    # the data and are therefore duplicates of the full-sample cell.
    train_subsample = config["train_subsample"]

    # Per-model NaN handling: ``nan_policy`` maps model key -> fill policy, with an
    # optional "default" entry; unlisted models keep NaNs ("native") for AutoGluon to
    # impute internally. The fill is fit on each model's training split at predict time.
    nan_policy = config["nan_policy"]
    if nan_policy:
        logger.info("NaN handling policy: %s", nan_policy)

    for seed in seeds:
        logger.info("--- Seed %s ---", seed)
        config["random_state"] = seed
        _set_global_seeds(seed)
        benchmark = configure_benchmark(config)

        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        predictions_dir = os.path.join(seed_dir, "predictions")
        logs_dir = os.path.join(seed_dir, "logs")
        stats_dir = os.path.join(seed_dir, "stats")
        for d in (predictions_dir, logs_dir, stats_dir):
            os.makedirs(d, exist_ok=True)
        validated_truth_paths: set[str] = set()

        if reverse:
            benchmark._key_list = list(reversed(benchmark._key_list))

        pbar = tqdm(total=len(benchmark) * len(model_names), desc=f"Seed {seed}")
        results = []

        # Deterministic grid index, reset per seed, so multi-GPU shard workers (each with
        # the same num_shards but a distinct shard_index) partition the cells with no overlap.
        cell_idx = -1
        for model_name in model_names:
            for data_train, data_test, key, task_type in benchmark:
                cell_idx += 1
                if num_shards > 1 and cell_idx % num_shards != shard_index:
                    pbar.update(1)
                    continue
                pbar.set_description(f"Seed {seed} | {key} | {model_name}")

                if data_train is None or data_test is None:
                    pbar.update(1)
                    continue

                # An empty train or test split can't be fit (AutoGluon's internal
                # holdout split raises on n_samples=0). Record the skip and move on with
                # a single line instead of letting fit() emit a full traceback.
                if len(data_train) == 0 or len(data_test) == 0:
                    logger.warning(
                        "Skipping %s / %s: empty split (train=%d, test=%d).",
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                    )
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        "empty_split",
                        f"empty_split: train={len(data_train)}, test={len(data_test)}",
                    )
                    pbar.update(1)
                    continue

                # Duplicate grid cell: this cell's sample budget exceeds the data, so
                # _subsample_train was a no-op and the fit is byte-identical to the
                # full-sample cell's. Running it again would spend compute to produce a
                # second copy of one result, and reporting it would draw a learning curve
                # that appears to extend past the data and flatten there.
                if train_subsample is not None and len(data_train) < train_subsample:
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        "duplicate_cell",
                        f"duplicate_cell: only {len(data_train)} training rows available for "
                        f"train_subsample={train_subsample}; identical to the full-sample cell",
                    )
                    pbar.update(1)
                    continue

                # Skip classification-only models for regression datasets
                if task_type == TaskType.Regression and model_name in CLASSIFICATION_ONLY_MODELS:
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        "classification_only",
                        f"{model_name} only supports classification tasks",
                    )
                    pbar.update(1)
                    continue

                # Skip constant-target datasets for regression
                if task_type == TaskType.Regression:
                    label_col = data_train.columns[-1]
                    if data_train[label_col].std() == 0:
                        _write_skip_record(
                            stats_dir,
                            key,
                            model_name,
                            len(data_train),
                            len(data_test),
                            "constant_target",
                            "constant_target: all training target values are equal",
                        )
                        pbar.update(1)
                        continue

                pred_path = os.path.join(predictions_dir, f"{key}_{model_name}_predictions.csv")
                proba_path = os.path.join(predictions_dir, f"{key}_{model_name}_proba.csv")
                truth_path = os.path.join(predictions_dir, f"{key}_ground_truth.csv")
                stats_path = os.path.join(stats_dir, f"{key}_{model_name}.json")
                log_path = os.path.join(logs_dir, f"{key}_{model_name}.log")
                current_model_limit = (
                    model_size_limits[model_name] if model_name in model_size_limits else None
                )
                prior_pass = False
                prior_current_memory_failure = False
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path) as f:
                            prior_record = json.load(f)
                        prior_pass = prior_record.get("status") == "pass"
                        prior_current_memory_failure = _is_current_memory_failure_record(
                            prior_record, current_model_limit
                        )
                    except (OSError, ValueError):
                        prior_pass = False
                        prior_current_memory_failure = False
                if os.path.exists(truth_path) and truth_path not in validated_truth_paths:
                    _assert_ground_truth_compatible(truth_path, data_test)
                    validated_truth_paths.add(truth_path)
                # For classification, both files must exist to skip — otherwise
                # rerun so the missing proba file gets written and downstream
                # log_loss / ROC-AUC become available.
                if task_type == TaskType.Classification:
                    already_complete = (
                        os.path.exists(pred_path)
                        and os.path.exists(proba_path)
                        and os.path.exists(truth_path)
                        and prior_pass
                        and _saved_predictions_are_valid(
                            pred_path, proba_path, data_test, task_type
                        )
                    )
                else:
                    already_complete = (
                        os.path.exists(pred_path)
                        and os.path.exists(truth_path)
                        and prior_pass
                        and _saved_predictions_are_valid(
                            pred_path, proba_path, data_test, task_type
                        )
                    )
                if already_complete and not overwrite:
                    pbar.update(1)
                    continue
                # Start from the observed-safe sample region instead of spending a full fit
                # on a unit already beyond this model's calibrated feature x row boundary.
                # A saved pass remains authoritative, while --overwrite deliberately bypasses
                # the prior so a changed implementation can probe and recalibrate it.
                n_features = data_train.shape[1] - 1
                memory_skip = _memory_prior_skip_details(
                    model_size_limits, model_name, len(data_train), n_features
                )
                if memory_skip is not None and not overwrite and not prior_pass:
                    max_cells = memory_skip["memory_prior_max_cells"]
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        "model_limit",
                        f"exceeds calibrated {model_name} memory prior: {n_features} features x "
                        f"{len(data_train)} train rows = {memory_skip['required_cells']} > "
                        f"{max_cells} cells; start at <= "
                        f"{memory_skip['recommended_max_n_train']} train rows",
                        **memory_skip,
                    )
                    pbar.update(1)
                    continue
                # Legacy OOMs and typed OOMs from an older policy version are retried once.
                # Preserve current-version OOMs so restarts do not repeatedly burn GPU; a
                # future fix can bump MEMORY_RETRY_VERSION to target only these units.
                if prior_current_memory_failure and not overwrite:
                    pbar.update(1)
                    continue

                # Retried units are the only source artifacts we replace automatically.
                # Preserve their prior ledger and any partial predictions for audit/recovery;
                # successful units never reach this branch unless overwrite=True was explicit.
                for artifact in (stats_path, pred_path, proba_path):
                    archive_existing(artifact)

                # Per-model fitting-regime override from the roster (configs/models/*.json).
                # Every benchmarked model runs the run-wide regime — one default
                # configuration, no HPO, no bagging — so the ranking compares models rather
                # than tuning budgets. AUTOGLUON declares its own regime there instead, which
                # is what makes it a best-case AutoML *reference* rather than a peer entry.
                over = model_overrides[model_name] if model_name in model_overrides else {}
                with run_file_logger(log_path) as active_log_path:
                    model = AutoGluonModel(
                        ensemble=over["ensemble"] if "ensemble" in over else ensemble,
                        optimize=over["optimize"] if "optimize" in over else optimize,
                        models=[model_name],
                        task_type=task_type,
                        autogluon_time_limit=autogluon_time_limit,
                        autogluon_presets=(
                            over["presets"] if "presets" in over else autogluon_presets
                        ),
                        autogluon_path=os.path.join(autogluon_path, key),
                        num_hpo_trials=(
                            over["num_hpo_trials"] if "num_hpo_trials" in over else num_hpo_trials
                        ),
                    )

                    nan_pol = _resolve_nan_policy(model_name, nan_policy)

                    record = {
                        "dataset": key,
                        "model": model_name,
                        "nan_policy": nan_pol,
                        "n_train_samples": len(data_train),
                        "n_test_samples": len(data_test),
                        "status": "pass",
                        # Machine-readable skip/fail category consumed by
                        # tabbench_bio.coverage; empty on a passing run.
                        "reason": "",
                        "error": "",
                        "timestamp": datetime.now().isoformat(),
                        "train_time_s": None,
                        "inference_time_s": None,
                        "inference_time_per_sample_ms": None,
                        "inference_batch_size": None,
                        "inference_proba_batch_size": None,
                        "inference_oom_retries": None,
                        "train_peak_memory_mb": None,
                        "inference_peak_memory_mb": None,
                        "memory_backend": mem_backend,
                        "n_models_trained": None,
                        "n_base_models": None,
                        "ag_total_fit_time_s": None,
                        "ag_time_per_model_s": None,
                        "train_gpu_power_w": None,
                        "train_gpu_energy_j": None,
                        "train_cpu_energy_j": None,
                        "inference_gpu_power_w": None,
                        "inference_gpu_energy_j": None,
                        "inference_cpu_energy_j": None,
                    }

                    stage = "fit"
                    try:
                        data_train_fit = _maybe_subsample(
                            data_train, model_name, key, task_type, subsample_config, seed
                        )
                        # Per-model NaN policy: fit fill on this model's train, apply to test.
                        data_train_fit, data_test_fit = _apply_nan_policy(
                            data_train_fit, data_test, nan_pol
                        )

                        with _timed() as tt, _memory_tracker() as tm, _PowerTracker() as tp:
                            with _time_budget(autogluon_time_limit):
                                model.fit(data_train_fit)
                        record["train_time_s"] = round(tt[0], 3)
                        record["train_peak_memory_mb"] = tm.peak_mb
                        record["train_gpu_power_w"] = tp.gpu_mean_power_w
                        record["train_gpu_energy_j"] = tp.gpu_energy_j
                        record["train_cpu_energy_j"] = tp.cpu_energy_j
                        record.update(model.get_fit_stats())

                        stage = "inference"
                        with _timed() as it, _memory_tracker() as im, _PowerTracker() as ip:
                            with _time_budget(autogluon_time_limit):
                                if task_type == TaskType.Classification:
                                    y_pred, inference_batch_size, label_oom_retries = (
                                        _predict_with_oom_batching(
                                            model.predict,
                                            data_test_fit,
                                        )
                                    )
                                    y_proba, proba_batch_size, proba_oom_retries = (
                                        _predict_with_oom_batching(
                                            model.predict_proba,
                                            data_test_fit,
                                        )
                                    )
                                    inference_oom_retries = label_oom_retries + proba_oom_retries
                                else:
                                    y_pred, inference_batch_size, inference_oom_retries = (
                                        _predict_with_oom_batching(
                                            model.predict,
                                            data_test_fit,
                                        )
                                    )
                        record["inference_time_s"] = round(it[0], 3)
                        record["inference_peak_memory_mb"] = im.peak_mb
                        record["inference_batch_size"] = inference_batch_size
                        if task_type == TaskType.Classification:
                            record["inference_proba_batch_size"] = proba_batch_size
                        record["inference_oom_retries"] = inference_oom_retries
                        record["inference_time_per_sample_ms"] = round(
                            it[0] / len(data_test) * 1000, 4
                        )
                        record["inference_gpu_power_w"] = ip.gpu_mean_power_w
                        record["inference_gpu_energy_j"] = ip.gpu_energy_j
                        record["inference_cpu_energy_j"] = ip.cpu_energy_j

                        atomic_to_csv(y_pred.sort_index(), pred_path, index=True)

                        # For classification, also persist class probabilities
                        # so downstream metrics (log_loss, ROC-AUC) can be
                        # computed without rerunning the fitted model.
                        if task_type == TaskType.Classification:
                            atomic_to_csv(y_proba.sort_index(), proba_path, index=True)

                        if not os.path.exists(truth_path):
                            atomic_to_csv(
                                data_test[["target"]].sort_index(), truth_path, index=True
                            )
                            validated_truth_paths.add(truth_path)

                    except Exception as e:
                        # AutoGluon's internal holdout split raises this when an
                        # already-non-empty training set collapses to nothing after its
                        # rare-class filtering — common on tiny subsample cells. It's a
                        # benign skip, not a failure: log one line (no traceback) so the
                        # logs aren't flooded. The pre-fit check at the top of the loop
                        # only catches splits that are empty before AutoGluon runs.
                        if isinstance(e, ValueError) and "train set will be empty" in str(e):
                            logger.info(
                                "Skipping %s / %s: empty split after AutoGluon class filtering.",
                                key,
                                model_name,
                            )
                            record["status"] = "skip"
                            record["reason"] = "empty_split_after_filtering"
                            record["error"] = "empty_split_after_filtering"
                        elif isinstance(e, UnitTimeoutError):
                            # Delivered no prediction inside the budget every model was given.
                            logger.error("Timeout for %s / %s: %s", key, model_name, e)
                            record["status"] = "fail"
                            record["reason"] = "time_limit"
                            record["error"] = str(e)
                        elif _is_stage_oom(e, stage, active_log_path):
                            logger.error(
                                "CUDA OOM during %s for %s / %s: %s", stage, key, model_name, e
                            )
                            record["status"] = "fail"
                            record["reason"] = f"{stage}_oom"
                            record["memory_retry_version"] = MEMORY_RETRY_VERSION
                            record["error"] = str(e)
                        else:
                            logger.error("Error for %s / %s: %s", key, model_name, e, exc_info=True)
                            record["status"] = "fail"
                            record["reason"] = "fit_error"
                            record["error"] = str(e)

                    atomic_write_json(stats_path, record)

                # Free GPU memory between runs
                _cleanup_model(model)
                del model
                gc.collect()
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                results.append(record)
                pbar.update(1)

        pbar.close()
        _log_seed_summary(seed, results, stats_dir)
        failed = [r for r in results if r.get("status") == "fail"]
        if failed:
            # A failed unit is a recorded model outcome, not a harness crash: coverage.py
            # imputes it at chance level so a model cannot improve its standing by crashing
            # on the targets it finds hard, and assert_complete refuses to publish a run
            # whose units left no record at all. Raising here instead aborted the whole
            # shard, so the cell's metrics never ran and every passing unit in it was
            # discarded — the ledger exists precisely so that is not the response.
            names = ", ".join(f"{r['dataset']}/{r['model']}" for r in failed[:10])
            suffix = " ..." if len(failed) > 10 else ""
            logger.error(
                "%d model run(s) failed for seed %s: %s%s. Recorded in %s; scored at chance "
                "by coverage.impute_failures.",
                len(failed),
                seed,
                names,
                suffix,
                stats_dir,
            )


def _cleanup_model(model):
    """Clear AutoGluon internals and delete the on-disk predictor dir.

    The in-memory teardown breaks reference cycles so the predictor can be GC'd.
    The on-disk predictor directory is then removed: it is write-once / read-never
    once predictions, probabilities, and stats are saved (resume keys off the result
    CSVs, not this dir, and every new run gets a fresh ``uuid`` path), so leaving it
    behind only accumulates disk. Runs unconditionally after save, so it also sweeps
    the partial directory left by a failed ``fit()``.
    """
    try:
        if model.predictor is not None:
            # _learner / _trainer are AutoGluon internals: navigate defensively.
            learner = getattr(model.predictor, "_learner", None)
            trainer = getattr(learner, "_trainer", None) if learner else None
            if trainer is not None and hasattr(trainer, "models"):
                trainer.models.clear()
            model.predictor = None
    except Exception:
        pass

    path = model.autogluon_path
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _write_skip_record(stats_dir, key, model_name, n_train, n_test, reason, error, **details):
    """Record a unit the benchmark excluded by design.

    ``reason`` is one of :data:`tabbench_bio.coverage.DESIGN_SKIPS` — the machine-readable
    category the leaderboard reads to keep the skip out of the score rather than treating
    it as a loss; ``error`` carries the human-readable detail.
    """
    assert reason in DESIGN_SKIPS, (
        f"undeclared skip reason {reason!r}; add it to coverage.DESIGN_SKIPS"
    )
    record = {
        "dataset": key,
        "model": model_name,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "status": "skip",
        "reason": reason,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    assert not set(record).intersection(details), "Skip details cannot replace core record fields."
    record.update(details)
    path = os.path.join(stats_dir, f"{key}_{model_name}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("status") == "pass":
            raise AssertionError(
                f"Refusing to replace passing result with skip record: {path} ({reason})"
            )
        stable_keys = set(record) - {"timestamp"}
        if all(existing.get(field) == record.get(field) for field in stable_keys):
            return
        archive_existing(path)
    atomic_write_json(path, record)


def _log_seed_summary(seed, results, stats_dir):
    all_records = []
    for fname in sorted(os.listdir(stats_dir)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(stats_dir, fname)) as f:
                    all_records.append(json.load(f))
            except Exception:
                pass

    n_total = len(all_records)
    n_failed = sum(r.get("status") == "fail" for r in all_records)
    n_skipped = sum(r.get("status") == "skip" for r in all_records)
    n_this = len(results)
    n_this_failed = sum(r.get("status") == "fail" for r in results)
    n_this_skipped = sum(r.get("status") == "skip" for r in results)

    if n_total:
        logger.info(
            "Seed %s: %d this run (%d passed, %d skipped, %d failed) | "
            "%d pre-existing (%d skipped, %d failed). Stats: %s",
            seed,
            n_this,
            n_this - n_this_failed - n_this_skipped,
            n_this_skipped,
            n_this_failed,
            n_total - n_this,
            n_skipped - n_this_skipped,
            n_failed - n_this_failed,
            stats_dir,
        )
