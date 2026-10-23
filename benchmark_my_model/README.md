# Benchmark my model

Edit `my_model.py`, then run the helper to fit your model and get an Elo leaderboard,
figures, fold metrics and an HTML report. It uses scikit-learn directly.
AutoGluon, its fork and CUDA are optional.

For resumable cluster jobs, per-model environment profiles and SQLite results you
can merge with the benchmark, use the [model integration guide](INTEGRATION.md).

## Install and run

Use Python 3.13. Define `create_model(task)` in your model file,
returning a cloneable classifier or regressor; see [my_model.py](my_model.py).

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install "tabbench-bio[bio]"
tabbench-bio evaluate --model my_model.py --task overall
```

Before the first PyPI upload, install a built wheel with `[bio]` instead.
Open `my_model_results/report/report.html` after the run. The command downloads the
latest reference, verifies its checksum and evaluates your model on frozen folds.
Report generation shows confidence-interval progress. Choose a new output folder
for each run with `--output`; interrupted fits cannot resume.

## Add your model

Replace `create_model(task)` in `my_model.py`. Return a fresh, cloneable estimator
with `fit(X, y)` and `predict(X)`. For custom classes, inherit
`ClassifierMixin, BaseEstimator` or `RegressorMixin, BaseEstimator`, with the mixin
first, and expose all constructor parameters for cloning.

Features arrive as NumPy arrays. Handle missing values in a pipeline, fit preprocessing
on training data only and set random seeds explicitly. Each fold gets a fresh clone.

```sh
tabbench-bio evaluate --model my_model.py \
  --name "My classifier" --dataset OpenML-1083 --output my_classifier_results
```

The template selects a classifier or regressor from `task`. To try regression:

```sh
tabbench-bio evaluate --model my_model.py \
  --task regression --dataset OpenML-430 --output my_regressor_results
```

The helper scores `predict` output with macro-F1 or RMSE. It does not collect
probability-based AUROC. This standalone helper uses the active environment and
does not select registry profiles. Install your model's dependencies there with
`uv pip install`, and set its thread and GPU parameters yourself. `--threads 2`
limits CPU threads during fitting.

## Compare with published models

The default `--baseline latest` resolves the
[latest published release](https://github.com/not-a-feature/TabBench-Bio/releases/latest)
on each run. Pass a
local SQLite path to pin a particular reference or work without a release lookup.

```sh
tabbench-bio evaluate --model my_model.py \
  --baseline latest --cell cap_10000_n100 \
  --dataset OpenML-1083 --dataset OpenML-1088 \
  --name "My classifier" --output published_comparison
```

The helper opens the database read-only, extracts the cell configuration and
reconstructs its frozen cross-validation splits from recorded held-out targets.
Before each fit it checks the held-out row IDs and labels. Only your model is fitted.
Published baseline predictions are reused across every configured fold.

Select a different published cell to change the feature or sample budget.
Omit `--dataset` to use all classification datasets in that cell, or add
`--task regression` for regression. Missing datasets are downloaded and cached;
the curated embedding tables come from checksum-pinned GitHub Release assets.
Some sources require credentials that the package does not include.

Start with the two public datasets above. A subset run covers fewer targets than
the full benchmark, and the report shows that coverage. Describe it as a subset
comparison when reporting the ranking. Nothing is uploaded.

## Read the outputs

| File | Contents |
|---|---|
| `report/report.html` | Self-contained ranking, dataset scores, paired tests and coverage; sortable/filterable tables |
| `report/*.png`, `report/*.svg`, `report/*.pdf` | Elo, dataset heatmaps and paired-score plots |
| `report/leaderboard.csv` | Elo, intervals, ranks, target counts and descriptive Score |
| `fold_metrics.csv` | New model/dataset/fold metrics and timings |
| `metrics/` | Full comparison metrics, including any imported baselines |
| `config.json` | Resolved configuration. Published comparisons also save frozen splits |
| `baseline.json` | Automatically selected release, URL, local path and SHA-256 |
| `report/dataset_scores_*.csv`, `report/dataset_summary_*.csv` | Dataset means, fold SDs and fold counts |
| `report/paired_comparisons_*.csv`, `report/paired_successful_*.csv` | Paired effects, bootstrap intervals, Holm tests and successful-fit sensitivity |
| `report/coverage.csv`, `report/failure_reasons.csv` | Scored folds, imputations and failure summaries |
| `attempts.csv` | Recorded attempt outcomes |
| `report/report_config.json` | Report selection and score fingerprint for regeneration |
| `report/*.tex` | Paper tables; requires `booktabs` and `graphicx` |

Regenerate a report from saved metrics without fitting models or downloading data:

```sh
tabbench-bio evaluate --report-only --output my_classifier_results \
  --name "My classifier" --task classification
```

The helper writes CSV metrics rather than SQLite attempt bundles.
Use the [registry pipeline](INTEGRATION.md) if you need results for `tabbench-bio merge`.

To rank a published database without training:

```sh
tabbench-bio leaderboard \
  --sqlite /path/to/results.sqlite --cell cap_10000_n100 \
  --task classification --plot --plot-path elo.png --csv leaderboard.csv
```

The API and CLI use fold-level Bradley-Terry Elo with RF fixed at 1000.
Normalised Score is a separate descriptive statistic. Elo requires RF and at least
one comparable model. The default is 2,000 target-bootstrap rounds, matching the
website and paper. Small target pools still provide limited uncertainty information.
