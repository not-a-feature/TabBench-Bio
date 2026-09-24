# Contributing to TabBench-Bio

You can contribute datasets, model adapters or fixes to the benchmark. For bugs and
feature requests, open a [GitHub issue](https://github.com/not-a-feature/TabBench-Bio/issues).

## Development environment

```bash
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e ".[dev,bio]"
```

This is enough for core development and the
[scikit-learn example](benchmark_my_model/README.md). Registered models use separate
[environment profiles](environments/README.md). Install the selected profile to test
its adapter. Keep new model dependencies in that profile, so adding a model does not
change another model's environment.

## Adding a dataset

For GEO, TCGA, Kaggle or OpenML data, start with an entry in the
[dataset registry](src/tabbench_bio/bio/data/bio_datasets.json). Existing loaders handle
these sources. Here is the shape of a TCGA entry:

```json
{
  "bio_id": "TCGA-TCGA-LUAD_Gene-Expression-Quantification",
  "source": "tcga",
  "fetch_id": "TCGA-LUAD",
  "target": "sample_type",
  "problem_type": "binary",
  "enabled": true,
  "redistributable": false,
  "license": "NIH GDC open access",
  "source_url": "https://portal.gdc.cancer.gov/projects/TCGA-LUAD",
  "citation": "...",
  "max_features": null
}
```

Choose a stable `bio_id`. Set `fetch_id` to the source accession or dataset ID and
`target` to the label column or characteristic. `problem_type` accepts `binary`,
`multiclass` or `regression`. `max_features` sets an optional dataset-specific cap.
Record the actual citation before submitting the entry.

To try a separate registry, set `TABBENCH_BIO_DATASETS` to its JSON file. Check that
the dataset loads and that its dimensions and task type are correct:

```python
from tabbench_bio import load_bio_as_dataset

ds = load_bio_as_dataset("TCGA-TCGA-LUAD_Gene-Expression-Quantification", cache_dir=".cache/bio")
print(ds.features.shape, ds.info.task_type)
```

A new source needs a loader in `src/tabbench_bio/bio/loaders/` that returns a
`BioRawDataset`. Register it in `loaders/__init__.py` and add tests under `tests/bio/`.
Keep heavy optional imports inside `fetch()` so loading the core package does not
require every source's dependencies.

Datasets must be publicly accessible under an open licence and have a classification
or regression target. Record the licence, source URL and citation in the registry.
For classification, the full benchmark keeps classes with at least 10 labelled samples,
filters rare classes before splitting and excludes tasks with fewer than two retained classes.

## Adding a model

For a quick comparison, use the [scikit-learn example](benchmark_my_model/README.md).
It fits a cloneable estimator with `fit(X, y)` and `predict(X)` and saves an HTML report,
Elo plot and fold metrics. You can also add a model to a leaderboard in memory:

```python
from tabbench_bio import Leaderboard
from mypackage import MyModel

lb = Leaderboard.from_results_dir("results/feature_sweep/cap_full")
lb.evaluate_and_add("My Model", MyModel(), config_path="results/feature_sweep/cap_full/config.json")
print(lb.rank())
```

For resumable jobs and mergeable SQLite results, follow the
[end-to-end model guide](benchmark_my_model/INTEGRATION.md). Models already registered
in AutoGluon's `ag_model_registry` need no adapter. Other models need an adapter and
an entry in `src/tabbench_bio/models/custom.py` with an `environment` field.
Choose an existing compatible profile or add `environments/<profile>.txt` with the
backend requirements. Install it in `.venvs/<profile>/` and test every supported task
type before submitting a long run. Built-in model entries in `configs/models/all.json`
also declare their profile.

Run the new model in its own result directory, then merge its database with the
benchmark. Keep existing frozen cell configurations unchanged.

## Checks

Run these before submitting a change:

```bash
uv run --no-sync ruff check src/ tests/ scripts/ benchmark_my_model/
uv run --no-sync ruff format --check src/ tests/ benchmark_my_model/
uv run --no-sync pytest
```

CI runs the same checks.

## Licence

By contributing, you agree that your contributions will be licensed under the
[European Union Public Licence 1.2](LICENSE).
