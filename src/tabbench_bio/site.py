"""Static leaderboard site generator (GitHub Pages-ready).

Turns a pipeline results directory (the ``metrics/*.csv`` + per-run ``stats/*.json``
written by :mod:`tabbench_bio.evaluation` / :mod:`tabbench_bio.predictions`) into a
self-contained static web page — a RamanBench-style leaderboard with sortable,
filterable tables and client-rendered (Plotly) figures — that can be committed under
``docs/`` and published with GitHub Pages ("Deploy from branch → main, /docs").

The static shell — ``index.html``, ``style.css``, ``app.js`` and the vendored
``assets/plotly.min.js`` — lives in the output directory itself (committed under ``docs/``
and hand-maintained); there is no build step and no CDN. This builder regenerates only the
*data*: ``app.js`` ``fetch()``-es ``<out_dir>/data/leaderboard.json`` at runtime and draws
every chart itself from the raw arrays in that payload; supporting CSVs land in
``<out_dir>/data/``.

Because browsers block ``fetch()`` over ``file://``, preview locally through a server
(``python -m http.server`` in *out_dir*); on GitHub Pages it is served over HTTP and just
works.

The canonical site is built end-to-end by ``run_grid.sbatch`` from the grid sweep's
untruncated full-sample cell (``results/feature_sweep/cap_full``); the calls below rebuild
the page from any results directory by hand.

Usage
-----
::

    from tabbench_bio.site import build_site

    build_site(
        "results/feature_sweep/cap_full",
        out_dir="docs",
        config_path="results/feature_sweep/cap_full/config.json",
    )

or via the CLI::

    tabbench-bio site --results-dir results/feature_sweep/cap_full --out docs \
        --config results/feature_sweep/cap_full/config.json
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from tabbench_bio.coverage import assert_complete, coverage_counts, impute_failures, load_status
from tabbench_bio.elo import compute_elo, score_table, win_counts
from tabbench_bio.leaderboard import Leaderboard
from tabbench_bio.metrics import PRIMARY_CLF_METRIC
from tabbench_bio.seeds import get_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model presentation: family (category) + a nicer display name
# ---------------------------------------------------------------------------

#: AutoGluon model key → algorithmic family. Keys not listed fall back to "Other".
MODEL_CATEGORY: dict[str, str] = {
    "LR": "Traditional ML",
    "KNN": "Traditional ML",
    "RF": "Tree-based",
    "XT": "Tree-based",
    "GBM": "Gradient Boosting",
    "XGB": "Gradient Boosting",
    "CAT": "Gradient Boosting",
    "NN_TORCH": "Deep Learning",
    "FASTAI": "Deep Learning",
    "REALMLP": "Deep Learning",
    "TABPFN": "Tabular Foundation",
    "REALTABPFN-V2": "Tabular Foundation",
    "REALTABPFN-V2.5": "Tabular Foundation",
    "TABPFN-V3": "Tabular Foundation",
    "TABPFN-WIDE": "Tabular Foundation",
    "TABFM": "Tabular Foundation",
    "TABDPT": "Tabular Foundation",
    "TABICL": "Tabular Foundation",
    "TABM": "Tabular Foundation",
    "MITRA": "Tabular Foundation",
    "AUTOGLUON": "AutoML",
    "DUMMY": "Baseline",
}

#: Family → display colour (used for badges in the table and bars/points in figures).
CATEGORY_COLOR: dict[str, str] = {
    "Traditional ML": "#3b82f6",
    "Tree-based": "#22c55e",
    "Gradient Boosting": "#f59e0b",
    "Deep Learning": "#ef4444",
    "Tabular Foundation": "#a855f7",
    "AutoML": "#111827",
    "Baseline": "#9ca3af",
    "Other": "#9ca3af",
}

#: Colours cycled over biological modalities in the positioning figure. Distinct from
#: CATEGORY_COLOR, which encodes model families — the two never share an axis.
_MODALITY_PALETTE: list[str] = [
    "#e11d48",  # rose
    "#0891b2",  # cyan
    "#7c3aed",  # violet
    "#ca8a04",  # amber
    "#059669",  # emerald
    "#db2777",  # pink
    "#2563eb",  # blue
    "#ea580c",  # orange
]

#: AutoGluon model key → human-friendly display name. Falls back to the key itself.
MODEL_DISPLAY: dict[str, str] = {
    "LR": "Logistic Reg.",
    "KNN": "KNN",
    "RF": "Random Forest",
    "XT": "Extra Trees",
    "GBM": "LightGBM",
    "XGB": "XGBoost",
    "CAT": "CatBoost",
    "NN_TORCH": "MLP",
    "FASTAI": "FastAI",
    "REALMLP": "RealMLP",
    "TABPFN": "TabPFN",
    "REALTABPFN-V2": "TabPFN v2",
    "REALTABPFN-V2.5": "TabPFN v2.5",
    "TABPFN-V3": "TabPFN v3",
    "TABPFN-WIDE": "TabPFN-Wide",
    "TABFM": "TabFM",
    "TABDPT": "TabDPT",
    "TABICL": "TabICL",
    "TABM": "TabM",
    "MITRA": "MITRA",
    "AUTOGLUON": "AutoGluon",
    "DUMMY": "Random",
}


def _category_of(model_id: str) -> str:
    return MODEL_CATEGORY.get(model_id, "Other")


def _display_of(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id, model_id)


#: Page-only data-source label overrides. The underlying ``spec.source`` (and dataset keys,
#: configs, caches) are unchanged — this only renames how a source is shown on the site.
_SOURCE_DISPLAY: dict[str, str] = {"OPENML": "Misc"}


def _display_source(source: str | None) -> str | None:
    """Page-facing data-source label (e.g. ``openml`` → ``"Misc"``); other sources unchanged."""
    if not source:
        return source
    up = source.upper()
    return _SOURCE_DISPLAY.get(up, up)


# ---------------------------------------------------------------------------
# Aggregating the per-run stats JSONs (timing / memory)
# ---------------------------------------------------------------------------


def _aggregate_run_stats(results_dir: str) -> pd.DataFrame:
    """Mean train time / inference cost / peak memory per model from stats JSONs.

    Reads every ``seed_*/stats/*.json`` written by the predictions step and averages
    the passing runs per model. ``Infer. s/1K`` is the mean inference time per 1,000
    samples in seconds (numerically equal to the recorded ms-per-sample).
    """
    rows: list[dict] = []
    for path in glob.glob(os.path.join(results_dir, "seed_*", "stats", "*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get("status") != "pass":
            continue
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["model_id", "Train Time s", "Infer. s/1K", "Peak Mem MB"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("model")
        .agg(
            train_time_s=("train_time_s", "mean"),
            infer_ms_per_sample=("inference_time_per_sample_ms", "mean"),
            peak_mem_mb=("train_peak_memory_mb", "mean"),
        )
        .reset_index()
        .rename(columns={"model": "model_id"})
    )
    agg["Train Time s"] = agg["train_time_s"].round(1)
    agg["Infer. s/1K"] = agg["infer_ms_per_sample"].round(2)  # ms/sample == s/1k samples
    agg["Peak Mem MB"] = agg["peak_mem_mb"].round(0)
    return agg[["model_id", "Train Time s", "Infer. s/1K", "Peak Mem MB"]]


def _expected_keys(config: dict) -> list[str]:
    """Dataset target keys the config scheduled, resolving ``null`` against the registry.

    Bio datasets are single-target, so a configured dataset contributes exactly one key.
    """
    from tabbench_bio.benchmark import TabBenchBio
    from tabbench_bio.bio import bio_dataset_names

    clf = config["dataset_names_classification"]
    reg = config["dataset_names_regression"]
    if clf is None:
        clf = bio_dataset_names("binary") + bio_dataset_names("multiclass")
    if reg is None:
        reg = bio_dataset_names("regression")
    return sorted(TabBenchBio.get_key(name, 0) for name in [*clf, *reg])


def _build_display_meta(results_dir: str, model_ids: list[str]) -> pd.DataFrame:
    """display_meta frame (model_id, Model, Category + timing + coverage) for the Leaderboard.

    Carries the per-model failure/design-skip counts so every published table shows how much
    of the grid a model's ``Score`` was actually computed on.
    """
    stats = _aggregate_run_stats(results_dir).set_index("model_id")
    cov = coverage_counts(load_status(results_dir))
    cov = cov.set_index("model_id") if not cov.empty else cov
    rows = []
    for mid in model_ids:
        row = {
            "model_id": mid,
            "Model": _display_of(mid),
            "Category": _category_of(mid),
        }
        if mid in stats.index:
            for col in ("Train Time s", "Infer. s/1K", "Peak Mem MB"):
                row[col] = stats.loc[mid, col]
        if not cov.empty and mid in cov.index:
            for col in ("# Failed", "# Skipped"):
                row[col] = int(cov.loc[mid, col])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-task mean metrics + dataset coverage (merged into the rank tables)
# ---------------------------------------------------------------------------


def _mean_metrics(metrics_df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Mean of selected raw metric columns per model, renamed to display labels.

    *cols* maps raw column name → display column name (e.g. ``{"f1_score": "F1"}``).
    Coverage is reported by the leaderboard's ``# Targets`` column instead of being
    recomputed here, so the tables carry one definition of it rather than two.
    """
    present = {raw: disp for raw, disp in cols.items() if raw in metrics_df.columns}
    if metrics_df.empty or "model" not in metrics_df.columns:
        return pd.DataFrame(columns=["model_id", *present.values()])

    out = metrics_df.groupby("model")[list(present)].mean().rename(columns=present).round(4)
    return out.reset_index().rename(columns={"model": "model_id"})


# ---------------------------------------------------------------------------
# Dataset overview tab (registry + results)
# ---------------------------------------------------------------------------


def _nan_fracs(config: dict | None, keys: list[str]) -> dict[str, float]:
    """Native missing-value fraction per key, from the unified bio source cache (no fetch).

    Reads the cached :class:`BioRawDataset` (``<cache_dir>/bio/datasets/<id>.pkl``) and
    returns the fraction of missing cells in its *native* feature matrix — a stable dataset
    property (like its native dimensionality), independent of any per-run capping/imputation.
    Falls back to omitting a key (so its missingness bucket is unknown) when the cache is
    absent.
    """
    cache_dir = (config.get("cache_dir") if config else None) or ".cache"
    bio_root = os.path.join(cache_dir, "bio")
    try:
        from tabbench_bio.bio.cache import load_cached_raw
    except Exception:  # pragma: no cover - bio extra not installed
        return {}

    fracs: dict[str, float] = {}
    for key in keys:
        dataset_name, _ = _split_key(key)
        try:
            raw = load_cached_raw(bio_root, dataset_name)
            if raw is None or getattr(raw, "X", None) is None:
                continue
            miss = np.asarray(pd.isna(raw.X))
            if miss.size:
                fracs[key] = float(miss.mean())
        except Exception:
            continue
    return fracs


def _dataset_overview(
    results_dir: str,
    config: dict | None,
    names: list[str],
    name_to_key: dict[str, str],
) -> list[dict]:
    """Per-dataset metadata for every dataset the site knows about: type / source / task /
    target / size / classes / license.

    *names* is the union of datasets across the leaderboard run and the feature-grid sweep;
    only those present in the dataset registry (:func:`is_bio_dataset`) are shown, so the
    registry is the single place to add or remove datasets. Newly registered datasets (e.g.
    the embedding sets, only present in the grid) therefore appear automatically. Descriptive
    fields come from the registry (:func:`get_spec`); size / class / feature counts use the
    run-effective values from the results stats when the dataset was in the leaderboard run
    (``name_to_key``), otherwise the dataset's native counts straight from the cache.
    """
    try:
        from tabbench_bio.bio.datasets import get_spec, is_bio_dataset
    except Exception:  # pragma: no cover - bio extra not installed
        return []
    try:
        from tabbench_bio.bio.cache import load_cached_raw
    except Exception:  # pragma: no cover - bio extra not installed
        load_cached_raw = None  # type: ignore[assignment]

    cache_dir = (config.get("cache_dir") if config else None) or ".cache"
    bio_root = os.path.join(cache_dir, "bio")

    rows = []
    for name in sorted(set(names)):
        if not is_bio_dataset(name):
            continue
        spec = get_spec(name)
        source = _display_source(spec.source)
        data_type = spec.data_type
        task = spec.problem_type
        target = spec.target
        license_ = spec.license

        raw = None
        if load_cached_raw is not None:
            try:
                raw = load_cached_raw(bio_root, name)
            except Exception:
                raw = None

        key = name_to_key.get(name)
        n_samples = _key_n_samples(results_dir, key) if key else None
        n_classes = (
            _key_n_classes(results_dir, key) if key and task in ("binary", "multiclass") else None
        )
        n_features = None
        if raw is not None and getattr(raw, "X", None) is not None:
            if n_samples is None:
                n_samples = int(raw.X.shape[0])
            n_features = int(raw.X.shape[1])
            if (
                n_classes is None
                and task in ("binary", "multiclass")
                and getattr(raw, "y", None) is not None
            ):
                n_classes = int(pd.Series(raw.y).nunique())

        rows.append(
            {
                "Dataset": spec.display_name or name,
                "Type": data_type or "—",
                "Source": source or "—",
                "Task": task or "—",
                "Target": target or "—",
                "# Samples": n_samples,
                "# Classes": n_classes,
                "# Features": n_features,
                "License": license_ or "—",
            }
        )
    return rows


def _key_n_samples(results_dir: str, key: str) -> int | None:
    """Total samples (train + test) for a key, from any of its stats JSONs."""
    for path in glob.glob(os.path.join(results_dir, "seed_*", "stats", f"{key}_*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
            n = (rec.get("n_train_samples") or 0) + (rec.get("n_test_samples") or 0)
            if n:
                return int(n)
        except Exception:
            continue
    return None


def _key_n_classes(results_dir: str, key: str) -> int | None:
    """Distinct class count for a key, from its saved ground-truth file."""
    for path in glob.glob(
        os.path.join(results_dir, "seed_*", "predictions", f"{key}_ground_truth.csv")
    ):
        try:
            gt = pd.read_csv(path, index_col=0)
            return int(gt["target"].nunique())
        except Exception:
            continue
    return None


def _split_key(key: str) -> tuple[str, int]:
    parts = key.split("_")
    return "_".join(parts[:-1]), int(parts[-1])


def _dataset_display(name: str) -> str:
    """Human-friendly dataset name from the registry (``display_name``), else the id."""
    try:
        from tabbench_bio.bio.datasets import get_spec

        return get_spec(name).display_name or name
    except Exception:
        return name


# ---------------------------------------------------------------------------
# Benchmark breakdown (per dataset type / size / class count: model × category scores)
# ---------------------------------------------------------------------------

#: Canonical bucket orders, used to order the dropdown options sensibly.
_SIZE_ORDER = ["≤ 200", "201-1000", "> 1000"]
_CLASS_ORDER = ["Binary", "Multiclass (≤ 10)", "Many (> 10)", "Regression"]
_NAN_ORDER = ["None", "Sparse (≤ 5%)", "Heavy (> 5%)"]


def _bucket_size(n: int | None) -> str | None:
    """Sample-size bucket (matches PLAN.md axes: ≤200 / 200–1000 / >1000)."""
    if n is None:
        return None
    if n <= 200:
        return "≤ 200"
    if n <= 1000:
        return "201-1000"
    return "> 1000"


def _bucket_classes(n: int | None, task: str | None) -> str | None:
    """Class-count bucket: binary / multiclass-≤10 / many, or regression."""
    if task == "regression":
        return "Regression"
    if n is None:
        return None
    if n <= 2:
        return "Binary"
    if n <= 10:
        return "Multiclass (≤ 10)"
    return "Many (> 10)"


def _bucket_nan(frac: float | None) -> str | None:
    """Missingness bucket from a dataset's native NaN fraction: none / sparse (≤5%) / heavy."""
    if frac is None:
        return None
    if frac <= 0:
        return "None"
    if frac <= 0.05:
        return "Sparse (≤ 5%)"
    return "Heavy (> 5%)"


def _normalized_scores(score_all: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset min-max scores (best model = 1, worst = 0).

    Same normalization as the per-dataset score heatmap, so a model's breakdown score
    is comparable across datasets of differing difficulty and averaging over a profile's
    datasets is meaningful.
    """
    norm = score_all.astype(float).copy()
    for col in norm.columns:
        vals = norm[col]
        worst = vals.min()
        denom = vals.max() - worst
        if denom and denom > 0:
            norm[col] = (vals - worst) / denom
        else:
            # Tied observed results score one; an unrun pairing remains missing.
            norm[col] = vals.where(vals.isna(), 1.0)
    return norm


def _build_breakdown(
    score_all: pd.DataFrame,
    results_dir: str,
    model_ids: list[str],
    config: dict | None = None,
) -> dict | None:
    """Benchmark-breakdown payload: dataset profile axes → per-model, per-dataset scores.

    For every dataset target (``key``) this emits its profile buckets — biological data
    type (modality), sample-size bucket, class-count bucket and missingness bucket (native
    NaN fraction: none / sparse / heavy) — together with each model's
    normalized score on it. The client groups the datasets by a chosen axis and shows a
    model × category table of mean scores, so you can see how each model performs across
    dataset *types* (not a single recommendation). A pure group-by + mean over precomputed
    data, so it updates instantly with no server (matches the static-site design).

    Returns ``None`` when there is no score table to build on.
    """
    if score_all is None or score_all.empty or score_all.shape[0] < 1:
        return None
    try:
        from tabbench_bio.bio.datasets import get_spec
    except Exception:  # pragma: no cover - bio extra not installed
        get_spec = None  # type: ignore[assignment]

    norm = _normalized_scores(score_all)
    nan_fracs = _nan_fracs(config, list(norm.columns))
    datasets: list[dict] = []
    for key in norm.columns:
        name, _ = _split_key(key)
        data_type = task = None
        if get_spec is not None:
            try:
                spec = get_spec(name)
                # Group by the curated biological modality; fall back to the source label
                # for datasets that have no data_type curated yet.
                data_type = spec.data_type or _display_source(spec.source)
                task = spec.problem_type
            except Exception:
                pass
        n_samples = _key_n_samples(results_dir, key)
        n_classes = _key_n_classes(results_dir, key) if task in ("binary", "multiclass") else None
        col = norm[key]
        scores = {mid: round(float(col[mid]), 4) for mid in col.index if pd.notna(col[mid])}
        datasets.append(
            {
                "key": key,
                "name": name,
                "data_type": data_type or "—",
                "size": _bucket_size(n_samples),
                "classes": _bucket_classes(n_classes, task),
                "nan": _bucket_nan(nan_fracs.get(key)),
                "scores": scores,
            }
        )

    def _options(field: str, order: list[str] | None = None) -> list[str]:
        present = {d[field] for d in datasets if d[field]}
        return [v for v in order if v in present] if order else sorted(present)

    axes = [
        {"key": "data_type", "label": "Dataset type", "options": _options("data_type")},
        {"key": "size", "label": "Sample size", "options": _options("size", _SIZE_ORDER)},
        {"key": "classes", "label": "Class count", "options": _options("classes", _CLASS_ORDER)},
        {"key": "nan", "label": "Missingness", "options": _options("nan", _NAN_ORDER)},
    ]
    # Keep only axes that actually vary in this run (an axis with no observed value is noise).
    axes = [a for a in axes if a["options"]]
    models = {
        mid: {"display": _display_of(mid), "category": _category_of(mid)} for mid in model_ids
    }
    return {"axes": axes, "datasets": datasets, "models": models}


# ---------------------------------------------------------------------------
# Feature x sample grid (HDLSS sweep): per-cell Elo + performance-surface figures
# ---------------------------------------------------------------------------

#: Metrics offered in the grid Elo metric selector (each ranked higher-is-better). The
#: leaderboard's primary metric leads, so it is the default; only those present in the sweep
#: summary are shown. The primary metric is also the metric AutoGluon fits and selects on
#: (model.py), so the ranking reports what the models were optimised for; the rest re-rank a
#: fixed set of fits and are secondary by construction.
_GRID_ELO_METRICS = [PRIMARY_CLF_METRIC, "balanced_accuracy", "matthews_corrcoef", "roc_auc"]

#: Metrics for which the client draws the per-dataset performance surfaces.
_GRID_FIG_METRICS = ["f1_macro", "balanced_accuracy", "matthews_corrcoef", "roc_auc"]


def _sample_sort_key(v):
    """Sort key for a grid axis (n_train or feature cap): numeric ascending, 'full' last."""
    s = str(v)
    if s.lower() == "full":
        return (1, math.inf)
    try:
        return (0, float(s))
    except ValueError:
        return (1, math.inf)


def _build_grid(out_dir: str) -> dict | None:
    """Grid-tab payload from the committed feature×sample sweep aggregates.

    Reads ``<out_dir>/data/feature_grid_summary.csv`` (written by the grid sbatch from
    :mod:`scripts.feature_sweep`) and emits, per (feature-cap, training-size) cell, a
    pairwise Elo ranking across the grid datasets — reusing the same
    :func:`score_table` / :func:`compute_elo` machinery as the main leaderboard. Also
    embeds, per dataset/metric/model, the mean-metric grid (indexed ``[sample][cap]``) the
    client draws as the performance-surface heatmaps and learning curves. Returns ``None``
    when the sweep aggregates are absent, so the grid section simply does not appear.
    """
    summary_path = os.path.join(out_dir, "data", "feature_grid_summary.csv")
    if not os.path.exists(summary_path):
        return None
    df = pd.read_csv(summary_path)
    if df.empty or not {"dataset", "model", "max_features", "n_train"}.issubset(df.columns):
        return None
    elo_metrics = [m for m in _GRID_ELO_METRICS if f"{m}__mean" in df.columns]
    if not elo_metrics:
        return None

    # Show only datasets present in the dataset registry, so the registry is the single place
    # to add/remove datasets (sweep aggregates for unregistered datasets are ignored).
    try:
        from tabbench_bio.bio.datasets import is_bio_dataset

        df = df[df["dataset"].map(is_bio_dataset)]
    except Exception:  # pragma: no cover - bio extra not installed
        pass
    if df.empty:
        return None

    df["n_train"] = df["n_train"].astype(str)  # 'full' + numeric-as-string
    df["max_features"] = df["max_features"].astype(str)  # 'full' + numeric-as-string
    caps = sorted({str(c) for c in df["max_features"].dropna()}, key=_sample_sort_key)
    samples = sorted({str(v) for v in df["n_train"]}, key=_sample_sort_key)

    # Map each grid dataset to its biological domain (the curated data_type, falling back to
    # the source label) so the client can restrict the Elo to one domain. Mirrors the
    # Breakdown tab's grouping.
    domain_of: dict[str, str] = {}
    try:
        from tabbench_bio.bio.datasets import get_spec

        for ds in df["dataset"].unique():
            spec = get_spec(ds)
            domain_of[ds] = spec.data_type or _display_source(spec.source) or "—"
    except Exception:  # pragma: no cover - bio extra not installed
        pass
    df["__domain"] = df["dataset"].map(lambda d: domain_of.get(d, "—"))
    distinct_domains = sorted(set(df["__domain"]))
    # "all" pools every grid dataset; per-domain options are added only when the run spans
    # more than one domain (otherwise they would just duplicate "all").
    domains = ["all"] + (distinct_domains if len(distinct_domains) > 1 else [])

    def _cell_elo_rows(cell: pd.DataFrame, metric: str) -> list[dict] | None:
        clf_like = cell.rename(columns={"dataset": "key", f"{metric}__mean": metric})[
            ["model", "key", metric]
        ]
        elo = compute_elo(score_table(clf_like, None, clf_metric=metric), n_boot=100)
        if elo.empty:
            return None
        elo = elo.sort_values("Elo", ascending=False)
        return [
            {
                "model_id": r["model_id"],
                "display": _display_of(r["model_id"]),
                "category": _category_of(r["model_id"]),
                "Elo": int(r["Elo"]),
                "Elo_lo": int(r["Elo_lo"]),
                "Elo_hi": int(r["Elo_hi"]),
                "n_targets": int(r["n_targets"]),
            }
            for _, r in elo.iterrows()
        ]

    # Per-cell pairwise Elo (RF = 1000), keyed "<metric>|<cap>|<n>|<domain>": one ranking per
    # recorded metric, across all grid datasets ("all") and per-domain when the run spans several.
    elo_cells: dict[str, list[dict]] = {}
    for metric in elo_metrics:
        mcol = f"{metric}__mean"
        for cap in caps:
            for n in samples:
                base_cell = df[(df["max_features"] == cap) & (df["n_train"] == n)].dropna(
                    subset=[mcol]
                )
                if base_cell.empty:
                    continue
                for domain in domains:
                    cell = (
                        base_cell if domain == "all" else base_cell[base_cell["__domain"] == domain]
                    )
                    if cell.empty:
                        continue
                    rows = _cell_elo_rows(cell, metric)
                    if rows:
                        elo_cells[f"{metric}|{cap}|{n}|{domain}"] = rows
    if not elo_cells:
        return None

    # Embed the per-dataset performance surfaces so the client draws the heatmaps /
    # learning curves itself: per dataset → metric → model a 2-D grid of the mean metric
    # indexed [sample][cap] (sample order = sample_sizes, cap order = feature_caps).
    surface: dict[str, dict] = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds].copy()
        per_metric: dict[str, dict] = {}
        for met in _GRID_FIG_METRICS:
            mcol = f"{met}__mean"
            if mcol not in sub.columns:
                continue
            lut = sub.set_index(["model", "n_train", "max_features"])[mcol].to_dict()
            grids: dict[str, dict] = {}
            for mdl in sorted(sub["model"].unique()):
                z = [[_clean(lut.get((mdl, n, cap))) for cap in caps] for n in samples]
                if any(v is not None for row in z for v in row):
                    grids[mdl] = {
                        "display": _display_of(mdl),
                        "category": _category_of(mdl),
                        "z": z,
                    }
            if grids:
                per_metric[met] = grids
        if per_metric:
            surface[ds] = per_metric
    fig_datasets = sorted(surface)
    fig_metrics = [m for m in _GRID_FIG_METRICS if any(m in surface[d] for d in surface)]

    # Defaults: primary metric on the 200-sample × 25000-feature "all"-domain cell, falling
    # back to whatever cell exists for that metric.
    default_elo_metric = elo_metrics[0]
    default_domain = "all"
    default_cap = "25000" if "25000" in caps else caps[-1]
    default_samples = "200" if "200" in samples else samples[-1]
    if f"{default_elo_metric}|{default_cap}|{default_samples}|{default_domain}" not in elo_cells:
        key = next(
            (k for k in elo_cells if k.startswith(f"{default_elo_metric}|") and k.endswith("|all")),
            next(iter(elo_cells)),
        )
        _, default_cap, default_samples, default_domain = key.split("|")

    return {
        "feature_caps": caps,
        "sample_sizes": samples,
        "domains": domains,
        "default_cap": default_cap,
        "default_samples": default_samples,
        "default_domain": default_domain,
        "elo_metrics": elo_metrics,
        "default_elo_metric": default_elo_metric,
        "elo": elo_cells,
        "surface": surface,
        "fig_datasets": fig_datasets,
        "display_names": {ds: _dataset_display(ds) for ds in fig_datasets},
        "fig_metrics": fig_metrics,
        "default_dataset": fig_datasets[0] if fig_datasets else None,
        "default_metric": fig_metrics[0] if fig_metrics else None,
    }


# ---------------------------------------------------------------------------
# Column definitions per leaderboard tab
# ---------------------------------------------------------------------------


def _col(key, label, type_="num", *, fixed=False, default=True, digits=3, better=None):
    return {
        "key": key,
        "label": label,
        "type": type_,
        "fixed": fixed,
        "default": default,
        "digits": digits,
        "better": better,
    }


_FIXED_COLS = [
    _col("Rank", "Rank", "int", fixed=True, digits=0),
    _col("Model", "Model", "model", fixed=True),
    _col("Category", "Category", "cat", fixed=True),
]

# Coverage columns are shown by default, not hidden behind the column picker: with
# absent-by-design pairings excluded from Score rather than zeroed, a Score is only
# interpretable next to the number of targets it was computed on.
_COVERAGE_COLS = [
    _col("# Targets", "# Targets", "int", digits=0),
    _col("# Failed", "# Failed", "int", digits=0, better="low"),
    _col("# Skipped", "# Skipped", "int", digits=0, better="low"),
]

_OVERALL_COLS = _FIXED_COLS + [
    _col("Elo", "Elo", "int", digits=0, better="high"),
    _col("Score", "Score", digits=3, better="high"),
    _col("Avg Rank", "Avg Rank", digits=1, better="low"),
    _col("Improvability", "Improvability %", digits=1, better="low"),
    *_COVERAGE_COLS,
    _col("Train Time s", "Train Time s", digits=1, better="low"),
    _col("Infer. s/1K", "Infer. s/1K", digits=2, better="low", default=False),
    _col("Peak Mem MB", "Peak Mem MB", digits=0, better="low", default=False),
]

_CLF_COLS = _FIXED_COLS + [
    _col("Elo", "Elo", "int", digits=0, better="high"),
    _col("Score", "Score", digits=3, better="high"),
    _col("Avg Rank", "Avg Rank", digits=1, better="low"),
    _col("Macro-F1", "Macro-F1", digits=3, better="high"),
    _col("Bal. Acc.", "Bal. Acc.", digits=3, better="high"),
    _col("MCC", "MCC", digits=3, better="high", default=False),
    _col("F1", "F1", digits=3, better="high", default=False),
    _col("ROC-AUC", "ROC-AUC", digits=3, better="high", default=False),
    _col("Improvability", "Improvability %", digits=1, better="low", default=False),
    *_COVERAGE_COLS,
]

_REG_COLS = _FIXED_COLS + [
    _col("Elo", "Elo", "int", digits=0, better="high"),
    _col("Score", "Score", digits=3, better="high"),
    _col("Avg Rank", "Avg Rank", digits=1, better="low"),
    _col("RMSE", "RMSE", digits=3, better="low"),
    _col("R²", "R²", digits=3, better="high"),
    _col("Improvability", "Improvability %", digits=1, better="low", default=False),
    *_COVERAGE_COLS,
]

_DATASET_COLS = [
    _col("Dataset", "Dataset", "str", fixed=True),
    _col("Type", "Type", "str", fixed=True),
    _col("Source", "Source", "str", fixed=True),
    _col("Task", "Task", "str", fixed=True),
    _col("Target", "Target", "str"),
    _col("# Samples", "# Samples", "int", digits=0),
    _col("# Classes", "# Classes", "int", digits=0),
    _col("# Features", "# Features", "int", digits=0),
    _col("License", "License", "str", default=False),
]


def _clean(v):
    """JSON-safe scalar: numpy → python, NaN/NaT → None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _records(df: pd.DataFrame, cols: list[dict]) -> list[dict]:
    """DataFrame → list of JSON-safe dicts, one per column key (missing → None)."""
    keys = [c["key"] for c in cols]
    out = []
    for _, row in df.iterrows():
        out.append({k: _clean(row[k]) if k in df.columns else None for k in keys})
    return out


# ---------------------------------------------------------------------------
# Figure data (client-rendered charts): raw arrays the page draws with Plotly
# ---------------------------------------------------------------------------


def _winrate_figdata(score_all, elo_all, clf_only):
    """Head-to-head win-count matrix, models ordered by Elo (best at top-left)."""
    if score_all is None or score_all.empty or score_all.shape[0] < 2:
        return None
    models, m = win_counts(score_all)
    elo_map = (
        dict(zip(elo_all["model_id"], elo_all["Elo"]))
        if (elo_all is not None and not elo_all.empty)
        else {}
    )
    order = sorted(range(len(models)), key=lambda i: -elo_map.get(models[i], 0))
    mo = m[np.ix_(order, order)]
    labels = [_display_of(models[i]) + (" *" if models[i] in clf_only else "") for i in order]
    return {
        "kind": "winrate",
        "title": "Pairwise Win Rates",
        "caption": (
            "Number of targets on which the row model outperforms the column model (ties count "
            "as 0.5). Green = high win rate for the row model, red = low. Models are sorted by Elo "
            "(best at top-left). Only targets where both models produced a prediction are counted."
        ),
        "labels": labels,
        "matrix": [[_clean(v) for v in row] for row in mo],
    }


def _perf_time_figdata(overall):
    """Overall Score vs. mean training-time scatter points, coloured by family."""
    if overall is None or overall.empty or "Train Time s" not in overall.columns:
        return None
    d = overall.dropna(subset=["Train Time s", "Score"])
    d = d[d["Train Time s"] > 0]
    if d.empty:
        return None
    points = [
        {
            "display": r["Model"],
            "category": r.get("Category", "Other"),
            "x": float(r["Train Time s"]),
            "y": float(r["Score"]),
        }
        for _, r in d.iterrows()
    ]
    return {
        "kind": "perf_time",
        "title": "Performance vs. Cost",
        "caption": (
            "Overall Score against mean training time (log scale). Upper-left is best "
            "(high score, low cost). Coloured by model family."
        ),
        "points": points,
    }


def _heatmap_figdata(score_all, elo_all):
    """Per-dataset normalized-score matrix (best = 1, worst = 0), rows ordered by Elo."""
    if score_all is None or score_all.empty:
        return None
    norm = _normalized_scores(score_all)
    elo_map = (
        dict(zip(elo_all["model_id"], elo_all["Elo"]))
        if (elo_all is not None and not elo_all.empty)
        else {}
    )
    order = sorted(norm.index, key=lambda mdl: -elo_map.get(mdl, 0))
    norm = norm.reindex(order)
    # Column labels: the dataset's display name; disambiguate with the target index only
    # when a dataset contributes more than one target column.
    bases = [_split_key(c)[0] for c in norm.columns]
    multi = {b for b in bases if bases.count(b) > 1}
    labels = [
        f"{_dataset_display(b)} (t{_split_key(c)[1]})" if b in multi else _dataset_display(b)
        for c, b in zip(norm.columns, bases)
    ]
    return {
        "kind": "score_heatmap",
        "title": "Per-dataset Scores",
        "caption": (
            "Each cell is a model's min-max normalized score on one dataset "
            "(best model = 1, worst = 0). Models are ordered by Elo (best at top)."
        ),
        "models": [_display_of(m) for m in norm.index],
        "datasets": labels,
        "z": [[_clean(v) for v in row] for row in norm.to_numpy()],
    }


def _composition_figdata(datasets):
    """Donut panels: datasets by source / task, and total samples by source."""
    if not datasets:
        return None
    df = pd.DataFrame(datasets)
    panels = []
    if "Source" in df.columns:
        vc = df["Source"].value_counts()
        panels.append(
            {
                "title": "Datasets by source",
                "labels": list(vc.index),
                "values": [int(v) for v in vc],
            }
        )
    if "Task" in df.columns:
        vc = df["Task"].value_counts()
        panels.append(
            {"title": "Datasets by task", "labels": list(vc.index), "values": [int(v) for v in vc]}
        )
    if "Source" in df.columns and "# Samples" in df.columns:
        s = df.copy()
        s["__n"] = pd.to_numeric(s["# Samples"], errors="coerce")
        s = s.dropna(subset=["__n"]).groupby("Source")["__n"].sum()
        if not s.empty:
            panels.append(
                {
                    "title": "Samples by source",
                    "labels": list(s.index),
                    "values": [int(v) for v in s],
                }
            )
    if not panels:
        return None
    return {
        "kind": "composition",
        "title": "Benchmark Composition",
        "caption": (
            "Dataset counts by data source and task type, plus total sample counts by source. "
            "Shows how the benchmark is distributed across origins and across classification vs. "
            "regression."
        ),
        "panels": panels,
    }


def _characteristics_figdata(datasets):
    """Per-dataset instance / feature / class counts, sorted by size."""
    if not datasets:
        return None
    df = pd.DataFrame(datasets)
    if "# Samples" not in df.columns:
        return None
    df = df.copy()
    df["__n"] = pd.to_numeric(df["# Samples"], errors="coerce")
    df = df.sort_values("__n", ascending=False)
    specs = [
        ("# Samples", "Instances", True),
        ("# Features", "Features", True),
        ("# Classes", "Classes", False),
    ]
    specs = [
        s
        for s in specs
        if s[0] in df.columns and pd.to_numeric(df[s[0]], errors="coerce").notna().any()
    ]
    if not specs:
        return None
    panels = [
        {
            "key": col,
            "title": title,
            "log": log,
            "values": [_clean(v) for v in pd.to_numeric(df[col], errors="coerce")],
        }
        for col, title, log in specs
    ]
    return {
        "kind": "characteristics",
        "title": "Dataset Characteristics",
        "caption": (
            "Per-dataset instance count, native feature count and class count, sorted by size. "
            "Bars are coloured by task (blue = classification, orange = regression). Instances and "
            "features use a log scale to span the HDLSS range."
        ),
        "datasets": [str(n) for n in df["Dataset"]],
        "tasks": [str(t) for t in df.get("Task", [""] * len(df))],
        "panels": panels,
    }


def _reference_collections() -> dict[str, list[list[int]]]:
    """Optional (samples, features) points for reference benchmark collections.

    Loaded from the JSON file named by ``$TABBENCH_BIO_REFERENCE_COLLECTIONS``, mapping a
    collection name to a list of ``[n_samples, n_features]`` pairs, e.g.::

        {"TabBench": [[1000, 20], ...], "TALENT": [...], "RamanBench": [...]}

    These are other benchmarks' dataset shapes, which we cannot derive from our own
    registry — the file has to be produced from their published metadata. When it is
    absent the positioning figure simply plots TabBench-Bio alone rather than inventing
    comparison points.
    """
    path = os.environ.get("TABBENCH_BIO_REFERENCE_COLLECTIONS")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        logger.warning("Could not read reference collections from %s — skipping.", path)
        return {}
    out: dict[str, list[list[int]]] = {}
    for name, pts in (raw or {}).items():
        cleaned = [
            [int(p[0]), int(p[1])]
            for p in pts
            if isinstance(p, (list, tuple)) and len(p) >= 2 and p[0] and p[1]
        ]
        if cleaned:
            out[str(name)] = cleaned
    return out


def _samples_features_figdata(datasets):
    """TabBench-Bio's datasets in the sample x feature plane, coloured by modality.

    The benchmark's positioning figure: it is what shows that these datasets occupy the
    HDLSS corner (features >> samples) that general tabular suites do not. Reference
    collections are overlaid when supplied (see :func:`_reference_collections`).
    """
    if not datasets:
        return None
    df = pd.DataFrame(datasets)
    if "# Samples" not in df.columns or "# Features" not in df.columns:
        return None
    df = df.copy()
    df["__n"] = pd.to_numeric(df["# Samples"], errors="coerce")
    df["__p"] = pd.to_numeric(df["# Features"], errors="coerce")
    df = df.dropna(subset=["__n", "__p"])
    df = df[(df["__n"] > 0) & (df["__p"] > 0)]
    if df.empty:
        return None

    modality_col = (
        "Type" if "Type" in df.columns else ("Source" if "Source" in df.columns else None)
    )
    modalities = sorted(set(df[modality_col].astype(str))) if modality_col else ["TabBench-Bio"]
    colors = {m: _MODALITY_PALETTE[i % len(_MODALITY_PALETTE)] for i, m in enumerate(modalities)}

    points = [
        {
            "display": str(r["Dataset"]),
            "modality": str(r[modality_col]) if modality_col else "TabBench-Bio",
            "x": int(r["__p"]),
            "y": int(r["__n"]),
        }
        for _, r in df.iterrows()
    ]
    return {
        "kind": "samples_features",
        "title": "Benchmark Positioning",
        "caption": (
            "Sample count against native feature count for every TabBench-Bio dataset "
            "(log-log), coloured by biological modality. The dashed diagonal marks "
            "samples = features: everything below it has more features than samples, the "
            "high-dimensional low-sample-size regime this benchmark targets. Reference "
            "benchmark collections are shown in grey where available."
        ),
        "points": points,
        "modality_colors": colors,
        "reference": _reference_collections(),
    }


def _build_figdata(overall, datasets, elo_all, score_all, clf_only):
    """Client-rendered figure payload: raw arrays + captions (no images).

    Each entry carries a ``kind`` the page dispatches on to draw the chart with Plotly,
    plus its title and caption. Builders with nothing to draw are skipped, so the gallery
    only shows what the run supports.
    """
    builders = [
        _samples_features_figdata(datasets),
        _winrate_figdata(score_all, elo_all, clf_only),
        _perf_time_figdata(overall),
        _heatmap_figdata(score_all, elo_all),
        _composition_figdata(datasets),
        _characteristics_figdata(datasets),
    ]
    return [f for f in builders if f]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_site(
    results_dir: str,
    out_dir: str = "docs",
    config_path: str | None = None,
    title: str = "TabBench-Bio Leaderboard",
    subtitle: str | None = None,
) -> str:
    """Render a static leaderboard site from a results directory.

    Parameters
    ----------
    results_dir : str
        Pipeline results directory (must contain ``metrics/*.csv``).
    out_dir : str
        Output directory for the site (default ``"docs"`` — GitHub Pages friendly).
    config_path : str | None
        Optional benchmark config; used to look up dataset feature counts from the
        processed-split cache for the Datasets tab.
    title, subtitle : str
        Page heading / sub-heading.

    Returns
    -------
    str
        Path to the written ``index.html``.
    """
    from tabbench_bio.config import load_config

    metrics_dir = os.path.join(results_dir, "metrics")
    clf_path = os.path.join(metrics_dir, "classification_metrics.csv")
    reg_path = os.path.join(metrics_dir, "regression_metrics.csv")
    clf_df = pd.read_csv(clf_path) if os.path.exists(clf_path) else pd.DataFrame()
    reg_df = pd.read_csv(reg_path) if os.path.exists(reg_path) else pd.DataFrame()

    if clf_df.empty and reg_df.empty:
        raise FileNotFoundError(
            f"No metrics found under {metrics_dir}. Run the pipeline (predictions + metrics) first."
        )

    config = None
    if config_path and os.path.exists(config_path):
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning("Could not load config %s for feature counts: %s", config_path, e)

    # One missingness convention for every table and figure on this page: a fit that failed
    # is scored at chance, a unit the design excluded is left out and shown as reduced
    # coverage, and a unit with neither a metric nor a status record blocks the build.
    status = load_status(results_dir)
    if config is not None:
        # Expected keys come from the *config*, not from what happened to land on disk: a
        # dataset that produced neither a metric nor a status record is absent from both
        # observed sets, which is precisely the case the gate exists to catch.
        expected = _expected_keys(config)
        assert_complete(
            pd.concat([clf_df, reg_df], ignore_index=True),
            status,
            keys=expected,
            models=config["models"],
            seeds=get_seeds(config),
        )
    clf_df = impute_failures(clf_df, status)
    reg_df = impute_failures(reg_df, status)

    model_ids = sorted(
        set(clf_df.get("model", pd.Series(dtype=str)))
        | set(reg_df.get("model", pd.Series(dtype=str)))
    )
    display_meta = _build_display_meta(results_dir, model_ids)
    lb = Leaderboard(reg_df, clf_df, display_meta)

    # Build the three leaderboard tabs (rank table + merged per-task mean metrics).
    overall = lb.rank("overall")
    clf = lb.rank("classification")
    reg = lb.rank("regression")

    clf_means = _mean_metrics(
        clf_df,
        {
            "balanced_accuracy": "Bal. Acc.",
            "matthews_corrcoef": "MCC",
            "f1_macro": "Macro-F1",
            "f1_score": "F1",
            "roc_auc": "ROC-AUC",
        },
    )
    reg_means = _mean_metrics(reg_df, {"rmse": "RMSE", "r2": "R²"})
    if not clf.empty and not clf_means.empty:
        clf = clf.merge(clf_means, on="model_id", how="left")
    if not reg.empty and not reg_means.empty:
        reg = reg.merge(reg_means, on="model_id", how="left")

    # Pairwise Elo (RF = 1000): overall pool + per-task pools, merged into each tab.
    score_all = score_table(clf_df, reg_df)
    elo_all = compute_elo(score_all)
    elo_clf = compute_elo(score_table(clf_df, None))
    elo_reg = compute_elo(score_table(None, reg_df))
    for tbl, elo in ((overall, elo_all), (clf, elo_clf), (reg, elo_reg)):
        if not tbl.empty and not elo.empty:
            tbl.drop(columns=[c for c in ("Elo",) if c in tbl.columns], inplace=True)
            tbl[["Elo"]] = tbl.merge(elo[["model_id", "Elo"]], on="model_id", how="left")[["Elo"]]

    # Mark models evaluated on classification targets only (no regression) with " *",
    # mirroring RamanBench. Only meaningful when the run actually has regression results.
    clf_models = set(clf_df.get("model", pd.Series(dtype=str)))
    reg_models = set(reg_df.get("model", pd.Series(dtype=str)))
    clf_only = (clf_models - reg_models) if not reg_df.empty else set()
    for tbl in (overall, clf):
        if not tbl.empty and clf_only:
            tbl["Model"] = [
                f"{m} *" if mid in clf_only else m for m, mid in zip(tbl["Model"], tbl["model_id"])
            ]

    keys = sorted(
        set(clf_df.get("key", pd.Series(dtype=str))) | set(reg_df.get("key", pd.Series(dtype=str)))
    )

    # Feature×sample grid: per-cell Elo + performance surfaces (when the grid sweep aggregates
    # have been committed under <out_dir>/data). Built before the dataset overview so its
    # datasets — which may include sets evaluated only in the sweep, e.g. the embeddings —
    # are folded into the Datasets tab automatically. Rendered as top-level page sections.
    grid = _build_grid(out_dir)

    # Datasets the site knows about: leaderboard-run targets ∪ grid-sweep datasets. The run
    # keys carry a "<dataset>_<target>" suffix; map each dataset to one representative key so
    # the overview can pull its run-effective sizes (grid-only datasets fall back to cache).
    name_to_key: dict[str, str] = {}
    for k in keys:
        name_to_key.setdefault(_split_key(k)[0], k)
    grid_names = set(grid["fig_datasets"]) if grid else set()
    dataset_names = sorted(set(name_to_key) | grid_names)
    datasets = _dataset_overview(results_dir, config, dataset_names, name_to_key)

    # Figure data: raw arrays the page draws client-side with Plotly (no images).
    figdata = _build_figdata(overall, datasets, elo_all, score_all, clf_only)

    # Assemble the embedded JSON payload.
    tabs: dict[str, dict] = {}
    if not overall.empty:
        tabs["overall"] = {
            "label": "Overall",
            "blurb": "Combined ranking across all tasks. Score is normalized performance "
            "averaged across classification (balanced accuracy) and regression (RMSE) datasets "
            "— best model per dataset = 1, worst = 0.",
            "columns": _OVERALL_COLS,
            "rows": _records(overall, _OVERALL_COLS),
            "filters": [{"column": "Category", "label": "Model family"}],
        }
    if not clf.empty:
        tabs["classification"] = {
            "label": "Classification",
            "blurb": "Per-model averages on classification datasets. Primary metric: balanced "
            "accuracy (imbalance-robust); MCC and macro-F1 also reported.",
            "columns": _CLF_COLS,
            "rows": _records(clf, _CLF_COLS),
            "filters": [{"column": "Category", "label": "Model family"}],
        }
    if not reg.empty:
        tabs["regression"] = {
            "label": "Regression",
            "blurb": "Per-model averages on regression datasets. Primary metric: RMSE.",
            "columns": _REG_COLS,
            "rows": _records(reg, _REG_COLS),
            "filters": [{"column": "Category", "label": "Model family"}],
        }
    tabs["datasets"] = {
        "label": "Datasets",
        "blurb": "The biological datasets covered by this benchmark — the leaderboard run plus "
        "the feature-grid sweep — with type, source, task, target and size.",
        "columns": _DATASET_COLS,
        "rows": [{c["key"]: _clean(r.get(c["key"])) for c in _DATASET_COLS} for r in datasets],
        "filters": [
            {"column": "Type", "label": "Dataset type"},
            {"column": "Source", "label": "Source"},
            {"column": "Task", "label": "Task"},
        ],
    }

    # Benchmark breakdown: how every model performs across dataset types / sizes / class counts.
    breakdown = _build_breakdown(score_all, results_dir, model_ids, config)
    if breakdown and breakdown["datasets"] and breakdown["axes"]:
        tabs["breakdown"] = {
            "label": "Breakdown",
            "blurb": "How each model performs across dataset profiles. Choose what to break the "
            "benchmark down by — dataset type, sample size, class count or missingness — to compare "
            "models within each category. Cells are the mean normalized score (best model per dataset "
            "= 1, worst = 0) over the datasets in that category; the best model per category is "
            "highlighted.",
            "kind": "breakdown",
            "axes": breakdown["axes"],
            "datasets": breakdown["datasets"],
            "models": breakdown["models"],
        }

    order = [
        t for t in ("overall", "breakdown", "classification", "regression", "datasets") if t in tabs
    ]
    n_models = len(model_ids)
    n_datasets = len(keys)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    run_name = os.path.basename(os.path.normpath(results_dir))
    subtitle = subtitle or (
        f"{n_models} models · {n_datasets} datasets · run {run_name} · generated {generated}"
    )

    payload = {
        "order": order,
        "tabs": tabs,
        "category_colors": CATEGORY_COLOR,
        "figdata": figdata,
        "grid": grid,
        "meta": {
            "title": title,
            "subtitle": subtitle,
            "generated": generated,
            "results_dir": run_name,
            "n_models": n_models,
            "n_datasets": n_datasets,
        },
    }

    # Write the data the page consumes (data/leaderboard.json) + supporting CSVs.
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "leaderboard.json"), "w") as f:
        json.dump(payload, f, indent=1)
    if not overall.empty:
        overall.to_csv(os.path.join(data_dir, "leaderboard_overall.csv"), index=False)
    if not clf.empty:
        clf.to_csv(os.path.join(data_dir, "leaderboard_clf.csv"), index=False)
    if not reg.empty:
        reg.to_csv(os.path.join(data_dir, "leaderboard_reg.csv"), index=False)
    pd.DataFrame(datasets).to_csv(os.path.join(data_dir, "datasets.csv"), index=False)

    # Add .nojekyll so GitHub Pages serves the files as-is (no Jekyll processing).
    open(os.path.join(out_dir, ".nojekyll"), "w").close()

    # The static shell (index.html / style.css / app.js) lives in the output dir itself
    # (committed under docs/, hand-maintained) — the builder only (re)writes the data and
    # figures above, it does not own the shell. Warn if the shell is missing so the page
    # won't silently fail to load.
    index_path = os.path.join(out_dir, "index.html")
    missing = [n for n in _SHELL_FILES if not os.path.exists(os.path.join(out_dir, n))]
    if missing:
        logger.warning(
            "Static shell %s missing from %s — the page won't render until they exist there. "
            "They are committed under docs/; copy them in to build a site in a new location.",
            missing,
            out_dir,
        )

    logger.info(
        "Wrote leaderboard data + %d figures to %s (%d models, %d datasets).",
        len(figdata),
        out_dir,
        n_models,
        n_datasets,
    )
    return index_path


#: The static page shell. It is committed and maintained directly under the output dir
#: (docs/); the builder regenerates only data/ and never overwrites these.
_SHELL_FILES = ("index.html", "style.css", "app.js", "assets/plotly.min.js")
