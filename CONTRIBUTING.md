# Contributing to TabBench-Bio

Thank you for your interest in contributing! This guide covers the two main paths:

1. **[Adding a dataset](#adding-a-dataset)** — register a biological dataset from GEO,
   TCGA, Kaggle, or OpenML.
2. **[Adding a model](#adding-a-model)** — evaluate a new model against the benchmark.

For bug reports and feature requests, please open an issue on GitHub
(`https://github.com/not-a-feature/TabBench-Bio/issues`).

---

## Setting up the development environment

```bash
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e ".[dev]"          # core + lint/test tooling
uv pip install -e ".[bio]"          # + dataset loaders, if you'll (re)fetch data
```

For the AutoGluon benchmark runner, install the fork (see the README).
The [custom-model example](benchmark_my_model/README.md) needs no AutoGluon:

```bash
uv pip install -r requirements-autogluon-fork.txt
uv pip install -e ".[autogluon,models]"
```

---

## Adding a dataset

Datasets live in a single JSON registry,
[`src/tabbench_bio/bio/data/bio_datasets.json`](src/tabbench_bio/bio/data/bio_datasets.json).
Adding one from a supported source (GEO, TCGA, Kaggle, OpenML) is usually a one-line entry —
**no Python required**.

### Step 1: Add a registry entry

```jsonc
{
  "bio_id": "TCGA-TCGA-LUAD_Gene-Expression-Quantification",
  "source": "tcga",
  "fetch_id": "TCGA-LUAD",        // accession / dataset id / OpenML id
  "target": "sample_type",        // column or characteristic to predict
  "problem_type": "binary",       // binary | multiclass | regression
  "enabled": true,
  "redistributable": false,
  "license": "NIH GDC open access",
  "source_url": "https://portal.gdc.cancer.gov/projects/TCGA-LUAD",
  "citation": "...",
  "max_features": null            // optional per-dataset feature cap override
}
```

You can also point `$TABBENCH_BIO_DATASETS` at your own JSON file to replace the registry
entirely without editing the bundled one.

### Step 2: Verify it fetches

```python
from tabbench_bio import load_bio_as_dataset

ds = load_bio_as_dataset("TCGA-TCGA-LUAD_Gene-Expression-Quantification", cache_dir=".cache/bio")
print(ds.features.shape, ds.info.task_type)
```

### Step 3 (new source only): Add a loader

If your data comes from a source not yet supported, add a loader in
[`src/tabbench_bio/bio/loaders/`](src/tabbench_bio/bio/loaders/) that returns a
`BioRawDataset` and register it in `loaders/__init__.py`. Keep heavy/optional imports
**inside** `fetch()` so the core package stays dependency-light. Add tests under
`tests/bio/`.

### Dataset inclusion criteria

| Criterion | Details |
|---|---|
| **Freely accessible** | Public source under an open license |
| **Supervised labels** | At least one classification or regression target |
| **Minimum size** | The full benchmark requires at least 10 labeled samples per retained class; rare classes are filtered before splitting, and tasks with fewer than two retained classes are excluded |
| **Provenance** | License, source URL, and citation recorded in the registry entry |

---

## Adding a model

### Option A: scikit-learn-compatible model (simplest)

Start with the [custom-model example](benchmark_my_model/README.md) to generate
an HTML report, Elo plot and fold metrics from the command line.

If your model has `.fit(X, y)` / `.predict(X)`, evaluate it against a results directory
without touching the package source:

```python
from tabbench_bio import Leaderboard
from mypackage import MyModel

lb = Leaderboard.from_results_dir("results/feature_sweep/cap_full")
lb.evaluate_and_add("My Model", MyModel(), config_path="results/feature_sweep/cap_full/config.json")
print(lb.rank())
```

### Option B: AutoGluon model key

Any key registered in AutoGluon's `ag_model_registry` (built-in or foundation) works as a
benchmark model — just add it to a config model list (e.g.
[`configs/models/all.json`](configs/models/all.json)) and run:

```bash
uv run --no-sync tabbench-bio run --config results/feature_sweep/cap_full/config.json --model MYMODEL
```

`tabbench_bio.model.AutoGluonModel` resolves model-name strings against the registry, so
no wrapper class is needed.

---

## Code style

```bash
uv run --no-sync ruff check src/ tests/ scripts/ benchmark_my_model/   # linting
uv run --no-sync ruff format --check src/ tests/ benchmark_my_model/  # formatting
uv run --no-sync pytest                            # tests
```

All of these run automatically in CI.

---

## License

By contributing, you agree that your contributions will be licensed under the
[European Union Public Licence 1.2](LICENSE).
