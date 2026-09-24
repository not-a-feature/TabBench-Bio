# Benchmark my model

Edit `my_model.py`, then run the helper to fit your model and get an Elo leaderboard,
figures, fold metrics and an HTML report. It uses scikit-learn directly.
AutoGluon, its fork and CUDA are optional.

For resumable cluster jobs, per-model environment profiles and SQLite results you
can merge with the benchmark, use the [model integration guide](INTEGRATION.md).

## Install and try it

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e ".[bio]"
uv run --no-sync python benchmark_my_model/run.py --output my_model_results
```

This installs the package from the clone. The first dataset download needs internet
access. Dataset caches go beneath the output directory, and OpenML also uses its user cache.

Open `my_model_results/report.html` directly in your browser.
The default example uses two public gene-expression datasets, OpenML-1083 and
OpenML-1088, with five outer folds, at most 10,000 retained features and 100 training
samples per fold. Your model, Random Forest and a constant baseline use the same splits.

This Random Forest is fitted locally with scikit-learn. Published benchmark results
use the AutoGluon adapter and a larger target pool, so the two comparisons differ.

Choose a fresh output directory each time. Existing output is protected.
A model error stops the run without producing a partial ranking.
The helper has no fit timeout or support for resuming interrupted fits.

## Add your model

Replace `create_model(task)` in `my_model.py`. Return a fresh, cloneable estimator
with `fit(X, y)` and `predict(X)`. For custom classes, inherit
`ClassifierMixin, BaseEstimator` or `RegressorMixin, BaseEstimator`, with the mixin
first, and expose all constructor parameters for cloning.

Features arrive as NumPy arrays. Handle missing values in a pipeline, fit preprocessing
on training data only and set random seeds explicitly. Each fold gets a fresh clone.

```sh
uv run --no-sync python benchmark_my_model/run.py \
  --name "My classifier" --dataset OpenML-1083 --output my_classifier_results
```

The template selects a classifier or regressor from `task`. To try regression:

```sh
uv run --no-sync python benchmark_my_model/run.py \
  --task regression --dataset OpenML-46983 --output my_regressor_results
```

The helper scores `predict` output with macro-F1 or RMSE. It does not collect
probability-based AUROC. This standalone helper uses the active environment and
does not select registry profiles. Install your model's dependencies there with
`uv pip install`, and set its thread and GPU parameters yourself. `--threads 2`
controls only the local RF baseline.

## Compare with published models

Download the [v0.1.0 SQLite bundle](https://github.com/not-a-feature/TabBench-Bio/releases/download/v0.1.0/results.sqlite)
and verify its SHA-256:
`98878cbc989c45a563f8b56ec3d3708eadf134fa7952fccddd04ecf6523d5b27`.

```sh
uv run --no-sync python benchmark_my_model/run.py \
  --baseline /path/to/results.sqlite --cell cap_10000_n100 \
  --dataset OpenML-1083 --dataset OpenML-1088 \
  --name "My classifier" --output published_comparison
```

The helper opens the database read-only, extracts the cell configuration and
reconstructs its frozen cross-validation splits from recorded held-out targets.
Before each fit it checks the held-out row IDs and labels. Only your model is fitted.
Published baseline predictions are reused across every configured fold.

Select a different published cell to change the feature or sample budget.
Omit `--dataset` to use all classification datasets in that cell, or add
`--task regression` for regression. Some tasks need local embedding files or
source credentials that the package does not include.

Start with the two public datasets above. A subset run covers fewer targets than
the full benchmark, and the report shows that coverage. Describe it as a subset
comparison when reporting the ranking. Nothing is uploaded.

## Read the outputs

| File | Contents |
|---|---|
| `report.html` | Report linking the figure, table and results |
| `leaderboard.png`, `leaderboard.svg` | Elo with 95% target-bootstrap intervals |
| `leaderboard.csv` | Elo, intervals, ranks, target counts and descriptive Score |
| `fold_metrics.csv` | New model/dataset/fold metrics and timings |
| `metrics/` | Full comparison metrics, including any imported baselines |
| `config.json` | Resolved configuration. Published comparisons also save frozen splits |

The helper writes CSV metrics rather than SQLite attempt bundles.
Use the [registry pipeline](INTEGRATION.md) if you need results for `tabbench-bio merge`.

To rank a published database without training:

```sh
uv run --no-sync tabbench-bio leaderboard \
  --sqlite /path/to/results.sqlite --cell cap_10000_n100 \
  --task classification --plot --plot-path elo.png --csv leaderboard.csv
```

The API and CLI use fold-level Bradley-Terry Elo with RF fixed at 1000.
Normalised Score is a separate descriptive statistic. Elo requires RF and at least
one comparable model. The default is 2,000 target-bootstrap rounds, matching the
website and paper. Small target pools still provide limited uncertainty information.
