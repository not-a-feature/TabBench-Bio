# TabBench-Bio

[Website](https://tabbench-bio.eu) · [GitHub](https://github.com/not-a-feature/TabBench-Bio) · [Codeberg](https://codeberg.org/not_a_feature/TabBench-Bio)

TabBench-Bio benchmarks machine learning on biological tables with many features and few
samples, such as gene-expression and methylation data. It downloads datasets, fixes the
train/test splits, fits models and ranks their predictions using fold-level Elo.
The full configuration covers 30 classification and 13 regression datasets.

This repository contains the package and experiment definitions. The
[published website](https://tabbench-bio.eu) is built from Codeberg's `pages` branch.

## Install

Clone the repository and install the package with its dataset loaders:

```bash
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e ".[bio]"
```

The [scikit-learn example](benchmark_my_model/README.md) works with this installation.
Registered models run in separate [environment profiles](environments/README.md).
Install the profile declared by your model before running it. For the existing
models that use `standard`, on Linux:

```bash
uv venv .venvs/standard --python 3.12
uv pip install --python .venvs/standard/bin/python -r environments/standard.txt
```

The profile installs the required AutoGluon fork and model dependencies.
Compatible models can share a profile. The [model guide](benchmark_my_model/INTEGRATION.md)
explains how to give a new model its own environment.

## Run a benchmark

To try your own scikit-learn model, edit `benchmark_my_model/my_model.py` and run:

```bash
uv run --no-sync python benchmark_my_model/run.py --output my_model_results
```

This fits your model and local baselines, then saves an HTML report, Elo plot and CSV
results. Its `--baseline` option compares against published predictions on frozen splits.

For resumable runs and SQLite results, [register an adapter](benchmark_my_model/INTEGRATION.md).
The model command runs the reference cell by default. Add `--full-grid` for all 28 cells:

```bash
uv run --no-sync tabbench-bio MYMODEL
uv run --no-sync tabbench-bio MYMODEL --full-grid
```

The command selects `.venvs/<profile>/` automatically, including on Slurm.
The lower-level `run` command and `scripts/feature_sweep.py` use their current Python
environment, so run them in a profile that supports every selected model.
Keep each run's configuration and `split_manifest.json` with its results.
Resuming relies on those settings and row identities staying fixed.

## Datasets and models

The [dataset registry](src/tabbench_bio/bio/data/bio_datasets.json) gives each task a
stable `bio_id`. Edit it to add datasets from a supported source, or set
`TABBENCH_BIO_DATASETS` to your own registry file. A run config selects tasks through
`datasets_classification` and `datasets_regression`.

TCGA, GEO and public OpenML downloads need no credentials. Kaggle requires
`~/.kaggle/kaggle.json`. Some tasks also need local embedding files.
The [contributor guide](CONTRIBUTING.md#adding-a-dataset) explains how to add and check a dataset.

The [model roster](configs/models/all.json) includes linear models, trees, neural
networks and tabular foundation models. `DUMMY` provides a constant baseline, and
Random Forest anchors Elo at 1000. New adapters need an entry in
`src/tabbench_bio/models/custom.py`, including their environment profile.

## Merge results and build the website

Combine a new model run with an existing database:

```bash
uv run --no-sync tabbench-bio merge results.sqlite results/my_model/results.sqlite \
  --output results/combined/results.sqlite
```

The output path must be new. The merge checks cell settings and held-out targets,
removes duplicate attempts and leaves both inputs unchanged. It also accepts result
directories containing writer databases.

The `website/` submodule tracks Codeberg's `pages` branch. Initialise and update it
before building from the repository root:

```bash
git submodule update --init website
git -C website switch pages
git -C website pull --ff-only
uv run --no-sync tabbench-bio leaderboard results/combined/results.sqlite --workers 4
python -m http.server 8000 --directory website
```

Open `http://localhost:8000/`. The build writes the dashboard, model cards, dataset
pages and JSON data to `website/`. Per-cell PNG plots and CSV tables go in the
Git-ignored `website/local/` directory. It generates no per-cell HTML reports or SVGs.
Use `--out preview` to build elsewhere.

The input database stays read-only. Building needs no raw datasets, GPU or LaTeX,
and publishes nothing automatically. See the
[model guide](benchmark_my_model/INTEGRATION.md#merge-and-generate-plots) for publishing
and cache settings. The released database and SHA-256 checksum are listed in the
[artifact browser](https://tabbench-bio.eu/artifacts.html).

## Read a leaderboard

Load a published SQLite bundle in Python:

```python
from tabbench_bio import Leaderboard

sqlite_path = "results.sqlite"
print(Leaderboard.sqlite_cells(sqlite_path))
leaderboard = Leaderboard.from_sqlite(sqlite_path, cell="cap_10000_n100")
print(leaderboard.rank())
```

Or print one cell and export its plot and table:

```bash
uv run --no-sync tabbench-bio leaderboard --sqlite results.sqlite \
  --cell cap_10000_n100 --plot --csv leaderboard.csv
```

`rank()`, the terminal summary and plots use Bradley-Terry Elo from matching
cross-validation folds. Each fold has weight `1 / target_fold_count`, and the 95%
intervals resample whole targets. The default is 2,000 bootstrap rounds, matching the
paper. Normalised `Score` and fold means are descriptive summaries.

Strict results use recorded DUMMY metrics for failed fits and omit incomplete
model-target fold sets. Elo needs RF and at least one comparable model.
The website also provides adaptive and conditional views, explained in the model guide.
Use `--task` to select classification or regression.

## Caches

Dataset caches hold source downloads, assembled tables and prepared splits. Set
`cache_dir` in a run config or `TABBENCH_BIO_CACHE` for the biological dataset cache.

Website builds cache fold metrics in `<out>.cache/fold_metrics.sqlite` and Elo in
`data/dashboard.json`. Reuse the same output directory to avoid repeating work.
Changed inputs, calculation code or relevant settings invalidate cached results.
Completed metric batches survive an interrupted build. Elo is saved after a full build.
Keep the metric cache locally. Deleting it forces recalculation.

`--workers` controls both metric and Elo calculations. Progress bars show completed
folds and comparison pools, elapsed time and estimated time remaining.

## Licence

EUPL-1.2. See [LICENSE](LICENSE). The vendored TabArena Elo helper retains Apache-2.0,
as recorded in [NOTICE](NOTICE) and [its licence](licenses/Apache-2.0.txt).
Datasets and models retain their providers' licences.

## Paper citation

Kreuer, J., Ouaari, S., Hellmig, J., Braitinger, J., and Pfeifer, N. (2026).
TabBench-Bio: A Living Benchmark for Machine Learning on High-Dimensional Biomedical Tables.
[arXiv:2609.07441](https://arxiv.org/abs/2609.07441).
[DOI: 10.48550/arXiv.2609.07441](https://doi.org/10.48550/arXiv.2609.07441).

## Agent skill for model selection

The [model-selection skill](skills/biomedical-tabular-model-selection/SKILL.md) helps an
agent compare published results for a dataset's modality and feature/sample budgets,
including uncertainty, failures, cost and training-data overlap. Install it in a
compatible agent with:

```sh
npx skills add https://tabbench-bio.eu/skill.md
```

This downloads the Markdown skill. A browsing agent can also read the
[hosted version](https://tabbench-bio.eu/skill.md) directly.
