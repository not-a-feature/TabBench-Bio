# TabBench-Bio

[Website](https://tabbench-bio.eu) · [GitHub](https://github.com/not-a-feature/TabBench-Bio) · [Codeberg](https://codeberg.org/not_a_feature/TabBench-Bio)

**A benchmark for machine learning on high-dimensional biological data.**

This is the public source repository for the benchmark package. The generated website is
published separately from the Codeberg deployment repository's `pages` branch.

TabBench-Bio evaluates tabular models on high-dimensional, low-sample-size (HDLSS)
biological datasets - gene-expression, methylation, and other omics matrices where the
number of features (genes/probes) vastly exceeds the number of samples. It provides a
reproducible **fetch → split → fit → score → rank** pipeline built on
[AutoGluon](https://auto.gluon.ai), with dataset loaders for public biological repositories
and a fully configurable registry containing 69 curated definitions (57 currently enabled).

---

## Installation

```bash
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
pip install -e .                 # core
pip install -e ".[bio]"          # + dataset loaders
```

**Model fitting requires the AutoGluon fork** that removes the 500-feature cap on
tabular foundation models. Install the fork before the local package with its full
set of optional dependencies:

```bash
pip install -r requirements-autogluon-fork.txt
pip install -e ".[full]"
```

---

## Quick start

```bash
# Run the feature × sample grid
python scripts/feature_sweep.py --grid-config configs/grid_sweep.json

# Or submit the complete workflow to SLURM
sbatch run_grid.sbatch

# Run or rank one grid cell
tabbench-bio run --config results/feature_sweep/cap_full/config.json --step predictions
tabbench-bio run --config results/feature_sweep/cap_full/config.json --step metrics
tabbench-bio leaderboard --results-dir results/feature_sweep/cap_full
```

Load a single dataset directly:

```python
from tabbench_bio import load_bio_as_dataset

ds = load_bio_as_dataset("TCGA-TCGA-BRCA_Gene-Expression-Quantification", cache_dir=".cache/bio")
df = ds.to_dataframe()        # features + "target" column
```

---

## Datasets

Datasets are defined in
[`src/tabbench_bio/bio/data/bio_datasets.json`](src/tabbench_bio/bio/data/bio_datasets.json),
where each entry has a stable `bio_id`. Without Python changes, you can:

- Add or enable datasets and set their target, problem type, or feature cap.
- Point `$TABBENCH_BIO_DATASETS` at your own JSON file to replace the registry entirely.
- Select datasets in a run config through `datasets_classification` and
  `datasets_regression`.

TCGA, GEO, and public OpenML datasets need no credentials. Kaggle downloads require a
`~/.kaggle/kaggle.json` API token.

---

## Models

Model names map to AutoGluon's registry:

- **Built-in tabular** - `LR`, `RF`, `XT`, `KNN`, `GBM`, `XGB`, `CAT`
- **Tabular foundation** - `TABPFN`/`REALTABPFN-V2`/`REALTABPFN-V2.5`, `TABPFN-V3`, `TABPFN-WIDE`, `TABFM`,
  `TABDPT`, `TABICL`, `TABM`, `MITRA`, `REALMLP`, `NN_TORCH`
- **`AUTOGLUON`** - AutoGluon's native `extreme` preset with a one-hour time limit at the
  two compute-intensive reference cells
- **Baseline** - `DUMMY`, a constant chance-level predictor (shown as **Random** on the site)

The bundled model roster lives in [`configs/models/all.json`](configs/models/all.json).

---

## Pipeline & outputs

`tabbench-bio run` runs two steps:

1. **predictions** - fit each (model, dataset, seed) and write prediction/probability CSVs,
   plus per-run stats (time, peak memory, GPU power, CPU energy).
2. **metrics** - compute per-(seed, dataset, model) classification/regression metrics into
   `results/<run>/metrics/`.

`Leaderboard.from_results_dir(...)` normalizes metrics produced by a local run. Published
TabBench Bio results use a single, content-addressed SQLite bundle: attempts point to
compressed ground-truth, prediction, and probability blobs, while metrics are recomputed
on read. The loader opens that file using SQLite `mode=ro` and never creates a writer,
metrics CSV, journal, or cache:

```python
from tabbench_bio import Leaderboard

sqlite_path = "tabbench-bio-results-v0.1.0.sqlite"
print(Leaderboard.sqlite_cells(sqlite_path))

leaderboard = Leaderboard.from_sqlite(sqlite_path, cell="cap_10000_n100")
print(leaderboard.rank())
```

The same operation is available from the CLI:

```bash
tabbench-bio leaderboard \
  --sqlite tabbench-bio-results-v0.1.0.sqlite \
  --cell cap_10000_n100
```

Use `leaderboard.evaluate_and_add(...)` with the matching cell config and any
scikit-learn-compatible estimator to compare a new model in memory:

```python
from sklearn.dummy import DummyClassifier

leaderboard.evaluate_and_add(
    "My model",
    DummyClassifier(strategy="most_frequent"),
    config_path="results/feature_sweep/cap_10000_n100/config.json",
)
print(leaderboard.rank())
```

The canonical database and its SHA-256 checksum are listed in the
[artifact browser](https://tabbench-bio.eu/artifacts.html).

The canonical built site is published from the orphan `pages` branch of the
[Codeberg deployment repository](https://codeberg.org/not_a_feature/TabBench-Bio) at
[tabbench-bio.eu](https://tabbench-bio.eu). GitHub `main` contains public code only; Codeberg
`main` is only a deployment-repository notice.

Maintainers synchronize the public code through an explicit allowlist and publish a complete
built site in one command:

```bash
python scripts/sync_from_tabarena.py --source /path/to/TabArena-Bio
python scripts/sync_from_tabarena.py --source /path/to/TabArena-Bio --check
python scripts/publish_codeberg_site.py --site-dir /path/to/built/site
```

Each site deployment replaces the Codeberg `pages` history with one commit. Generated site and
result files do not belong on GitHub `main`.

---

## Caching

Two layers keep re-runs cheap (everything under `<cache_dir>`):

- **Source caches** - TCGA matrices (`bio/tcga_raw/`), GEO SOFT files (`bio/geo_raw/`),
  and OpenML/Kaggle native caches.
- **Unified dataset cache** - the assembled dataset (`bio/datasets/<bio_id>.pkl`) and the
  prepared train/test splits (`datasets_processed/seed_N/`).

Override the cache root with `$TABBENCH_BIO_CACHE` or the `cache_dir` config key.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
