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
and a configurable registry of curated biomedical tasks.

---

## Installation

```bash
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e .                 # core
uv pip install -e ".[bio]"          # + dataset loaders
```

**The AutoGluon benchmark runner requires the AutoGluon fork** that removes the 500-feature cap on
tabular foundation models. Install the fork before the local package with its full
set of optional dependencies:

```bash
uv pip install -r requirements-autogluon-fork.txt
uv pip install -e ".[full]"
```

---

## Quick start

**Bring your own model:** start with [benchmark_my_model](benchmark_my_model/README.md).
It needs no AutoGluon fork, evaluates every configured fold, and produces a local HTML
report, Elo figure, leaderboard CSV, and fold metrics. The default example fits fresh
local baselines; `--baseline` compares against a published SQLite bundle on frozen splits.

Use `uv run --no-sync` to run commands in the environment created above:

```bash
# Run the feature × sample grid
uv run --no-sync python scripts/feature_sweep.py --grid-config configs/grid_sweep_all.json

# Select one compute lane when using a scheduler
uv run --no-sync python scripts/feature_sweep.py --grid-config configs/grid_sweep_all.json --include-device cpu

# Run or rank one grid cell
uv run --no-sync tabbench-bio run --config results/feature_sweep_all/cap_full/config.json --step predictions
uv run --no-sync tabbench-bio run --config results/feature_sweep_all/cap_full/config.json --step metrics
uv run --no-sync tabbench-bio leaderboard --results-dir results/feature_sweep_all/cap_full
```

The full configuration selects 30 classification and 13 regression datasets.
Use `python scripts/feature_sweep.py --help` for worker and resource settings.
Keep the configuration and `split_manifest.json` with the results when resuming;
changing either can invalidate comparisons.

The repository separates the package (`src/tabbench_bio/`), experiment definitions
(`configs/`), reusable commands (`scripts/`), and regression tests (`tests/`).

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
- **Tabular foundation** - `TABPFN`/`REALTABPFN-V2`/`REALTABPFN-V2.5`, `TABPFN-V3`,
  `TABPFN-WIDE`, `TABPFN-WIDE-5K-NE3`, `TABFM`,
  `TABDPT`, `TABICL`, `TABM`, `MITRA`, `REALMLP`, `NN_TORCH`
- **`AUTOGLUON`** - AutoGluon's native `extreme` preset with a one-hour time limit at the
  two compute-intensive reference cells
- **Baseline** - `DUMMY`, a constant predictor

The bundled model roster lives in [`configs/models/all.json`](configs/models/all.json).

---

## Pipeline & outputs

`tabbench-bio run` runs two steps:

1. **predictions** - fit each (model, dataset, seed) and store predictions, probabilities,
   logs, and run statistics in transactional SQLite writer bundles.
2. **metrics** - compute per-(seed, dataset, model) classification/regression metrics into
   `results/<run>/metrics/`.

`Leaderboard.from_results_dir(...)` loads fold metrics produced by a local run. Published
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

The site's Elo compares models on matching cross-validation folds, weighted by the reciprocal of
each target's fold count; confidence intervals resample whole targets. Incomplete
model–target pairs are omitted, failures use recorded DUMMY metrics, and pools without
Random Forest have no Elo. Fold means are descriptive summaries, not Elo inputs.

The same operation is available from the CLI:

```bash
uv run --no-sync tabbench-bio leaderboard \
  --sqlite tabbench-bio-results-v0.1.0.sqlite \
  --cell cap_10000_n100 --plot --csv leaderboard.csv
```

`rank()`, the terminal summary, and `--plot` all use fold-level Bradley–Terry Elo,
with Random Forest fixed at 1000 and target-bootstrap 95% intervals. Normalized `Score`
remains descriptive. Without RF and a comparable model, Elo/Rank are unavailable;
there is no implicit score-ranking fallback. `--task` selects both printed and plotted results.

Use `leaderboard.evaluate_and_add(...)` with the matching cell config and any
scikit-learn-compatible estimator to compare a new model in memory:

```python
from sklearn.dummy import DummyClassifier

leaderboard.evaluate_and_add(
    "My model",
    DummyClassifier(strategy="most_frequent"),
    config_path="results/feature_sweep/cap_10000_n100/config.json",
    task="classification",
)
print(leaderboard.rank())
```

The estimator is cloned for each configured fold. Model failures abort this convenience
API; incomplete custom results are not silently ranked. SQLite-backed comparisons check
the cell budget and held-out row IDs/labels. The example folder prepares frozen splits and
saves outputs; `evaluate_and_add` itself only updates the in-memory leaderboard.

The canonical database and its SHA-256 checksum are listed in the
[artifact browser](https://tabbench-bio.eu/artifacts.html).

The canonical built site is published from the orphan `pages` branch of the
[Codeberg deployment repository](https://codeberg.org/not_a_feature/TabBench-Bio) at
[tabbench-bio.eu](https://tabbench-bio.eu). GitHub `main` contains public code only; Codeberg
`main` is only a deployment-repository notice.

Maintainers publish a complete built site with:

```bash
uv run --no-sync python scripts/publish_codeberg_site.py --site-dir /path/to/built/site
```

The deployment helper appends a commit to Codeberg `pages`. Generated site and result files
do not belong on GitHub `main`.

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

EUPL-1.2 — see [LICENSE](LICENSE). The vendored TabArena Elo helper retains
Apache-2.0; see [NOTICE](NOTICE) and [its licence](licenses/Apache-2.0.txt).
Dataset and model licences remain those of their respective providers.

## Paper citation

Kreuer, J.; Ouaari, S.; Hellmig, J.; Braitinger, J.; Pfeifer, N. (2026). TabBench-Bio: A Living Benchmark for Machine Learning on High-Dimensional Biomedical Tables. arXiv:2609.07441. https://doi.org/10.48550/arXiv.2609.07441

[arXiv](https://arxiv.org/abs/2609.07441) · [DOI](https://doi.org/10.48550/arXiv.2609.07441)

## Agent skill for model selection

Use [biomedical-tabular-model-selection](skills/biomedical-tabular-model-selection/SKILL.md)
to compare methods for a biomedical dataset using the current published snapshot.
It accounts for modality, sample/feature budgets, uncertainty, failures, cost and
training-data overlap. It does not assume the reference leader is best for every task.

Install the skill in a compatible agent with:

```sh
npx skills add https://tabbench-bio.eu/skill.md
```

This downloads only the Markdown skill, without cloning the benchmark repository.
A browsing agent can read the [hosted skill](https://tabbench-bio.eu/skill.md)
and follow its linked data reference. Publication alone does not automatically
install or activate the skill in other agents.
