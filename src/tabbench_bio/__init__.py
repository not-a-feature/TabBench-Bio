"""TabBench Bio: a benchmark for machine learning on high-dimensional biological data.

TabBench Bio evaluates tabular models on high-dimensional, low-sample-size (HDLSS)
biological datasets — gene-expression, methylation, and other omics matrices — sourced
from GEO, TCGA, Kaggle, and OpenML.  It provides a reproducible fetch → split →
fit → score → rank pipeline built on AutoGluon.

Quick Start
-----------
::

    from tabbench_bio import TabBenchBio

    bench = TabBenchBio(
        dataset_names_classification=["OpenML-1138"],
        dataset_names_regression=[],
        cache_dir=".cache",
    )
    bench.init_datasets()
    for train_df, test_df, key, task_type in bench:
        print(key, len(train_df), len(test_df))

Or run the full feature × sample grid sweep and build the site::

    python scripts/feature_sweep.py --grid-config configs/grid_sweep.json
    tabbench-bio site --results-dir results/feature_sweep/cap_full --out docs
"""

import importlib

__version__ = "0.1.0"
__author__ = "Jules Kreuer (Uni Tübingen)"

_public_map = {
    # Core benchmark
    "TabBenchBio": ("tabbench_bio.benchmark", "TabBenchBio"),
    "configure_benchmark": ("tabbench_bio.benchmark", "configure_benchmark"),
    # Data layer
    "Dataset": ("tabbench_bio.dataset", "Dataset"),
    "DatasetInfo": ("tabbench_bio.dataset", "DatasetInfo"),
    "TaskType": ("tabbench_bio.dataset", "TaskType"),
    # Models
    "AutoGluonModel": ("tabbench_bio.model", "AutoGluonModel"),
    # Leaderboard — rank models from a results directory
    "Leaderboard": ("tabbench_bio.leaderboard", "Leaderboard"),
    # Static leaderboard site (GitHub Pages)
    "build_site": ("tabbench_bio.site", "build_site"),
    # Metrics
    "ClassificationMetrics": ("tabbench_bio.metrics", "ClassificationMetrics"),
    "RegressionMetrics": ("tabbench_bio.metrics", "RegressionMetrics"),
    "compute_metrics": ("tabbench_bio.metrics", "compute_metrics"),
    # Pipeline steps
    "compute_predictions": ("tabbench_bio.predictions", "compute_predictions"),
    "compute_metrics_from_predictions": (
        "tabbench_bio.evaluation",
        "compute_metrics_from_predictions",
    ),
    # Bio dataset registry / loaders
    "BIO_DATASETS": ("tabbench_bio.bio", "BIO_DATASETS"),
    "bio_dataset_names": ("tabbench_bio.bio", "bio_dataset_names"),
    "load_bio_dataset": ("tabbench_bio.bio", "load_bio_dataset"),
    "load_bio_as_dataset": ("tabbench_bio.bio", "load_bio_as_dataset"),
    # Config
    "load_config": ("tabbench_bio.config", "load_config"),
}

__all__: list[str] = sorted(_public_map.keys())


def __getattr__(name: str):
    if name in _public_map:
        module_name, attr = _public_map[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_public_map.keys()))
