# Benchmark my model

Edit `my_model.py`, then run one command to fit your model and produce an Elo
leaderboard, a PNG/SVG figure, a CSV of fold results, and an HTML report.
This uses scikit-learn directly; AutoGluon, CUDA, and the AutoGluon fork are not required.

## Install

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e ".[bio]"
```

The package is installed from this repository, not PyPI. The first dataset download
requires internet access. Benchmark caches are written beneath the output directory; OpenML also uses its user cache.

## Try the complete workflow

```sh
uv run --no-sync python benchmark_my_model/run.py --output my_model_results
```

Open `my_model_results/report.html` in your browser. No web server is necessary.
The default run uses two public gene-expression datasets (OpenML-1083 and
OpenML-1088), five outer folds, at most 10,000 retained features, and 100 training
samples per fold. It fits your model, a local Random Forest anchor, and a constant
baseline on the same splits. This is a small working example, **not a reproduction
of the published benchmark's AutoGluon Random Forest or its full target pool**.

Choose a fresh `--output` directory for every run. Existing output is never overwritten.
Model errors stop the run; the helper does not silently drop failed fits or publish
a partial leaderboard. It does not enforce a fit timeout or resume interrupted fits.

## Add your model

Replace `create_model(task)` in `my_model.py`. Return a fresh, cloneable
scikit-learn estimator with `fit(X, y)` and `predict(X)`. Custom classes should
inherit `ClassifierMixin, BaseEstimator` or `RegressorMixin, BaseEstimator`,
with the mixin first. All constructor parameters must be exposed for cloning.
Training and test features are NumPy arrays. Handle missing values in a pipeline,
fit all preprocessing only on training data, and set random seeds explicitly.

```sh
uv run --no-sync python benchmark_my_model/run.py \
  --name "My classifier" --dataset OpenML-1083 --output my_classifier_results
```

A fresh clone of the estimator is used for every fold. The default template returns
a classifier or regressor according to `task`. For a regression example:

```sh
uv run --no-sync python benchmark_my_model/run.py \
  --task regression --dataset OpenML-46983 --output my_regressor_results
```

Only the estimator's `predict` output is evaluated here (macro-F1 or RMSE); this
example does not collect probability-based AUROC. Add dependencies needed by your
estimator with `uv pip install ...`. Set its own thread/GPU parameters explicitly;
`--threads 2` controls the example's local RF baseline.

## Compare with the published models

This mode requires the canonical SQLite bundle. Its public download is currently
pending; the [artifact browser](https://tabbench-bio.eu/artifacts.html) shows its
availability and checksum. The local comparison above works without it.
Once you have the bundle, run:

```sh
uv run --no-sync python benchmark_my_model/run.py \
  --baseline /path/to/results.sqlite --cell cap_10000_n100 \
  --dataset OpenML-1083 --dataset OpenML-1088 \
  --name "My classifier" --output published_comparison
```

The database is opened read-only. The helper extracts the cell's configuration and
reconstructs its frozen CV row identities from the recorded held-out targets. It
checks the held-out row IDs and labels before each fit. It then evaluates only your
model; published baselines are not retrained. It uses every configured fold, rather
than assuming three seeds. Changing the feature/sample budget requires selecting
the corresponding published cell.

Omit `--dataset` to select the cell's classification datasets (or use
`--task regression` for regression). Some datasets need external local embedding
files or source credentials; those are not bundled in the package. Start with the
two public datasets above. In a subset run your model has lower target coverage
than the published baselines; the report shows coverage, and its ranking must not
be presented as a full benchmark result. No results are uploaded automatically.

## Outputs and command-line ranking

- `report.html`: local report linking the figure, table, and result files.
- `leaderboard.png` / `leaderboard.svg`: Elo and 95% target-bootstrap intervals.
- `leaderboard.csv`: Elo, intervals, ranks, and target counts. Score is descriptive.
- `fold_metrics.csv`: newly evaluated model/dataset/fold metrics and timings.
- `metrics/`: full comparison metrics, including imported baselines when selected.
- `config.json`: resolved configuration; published comparisons also save frozen splits.

To rank an existing published database without training anything:

```sh
uv run --no-sync tabbench-bio leaderboard \
  --sqlite /path/to/results.sqlite --cell cap_10000_n100 \
  --task classification --plot --plot-path elo.png --csv leaderboard.csv
```

Both the API and CLI rank **fold-level Bradley–Terry Elo**, anchored at RF = 1000.
They retain normalized Score as an additional descriptive statistic. Without RF
and at least one comparable model, Elo is unavailable; the command never silently
substitutes a normalized-score ranking. The API uses 100 target-bootstrap rounds; the published website uses 2,000, so
bootstrap summaries can differ slightly. Very small target pools give limited
uncertainty information even when all folds finish.
