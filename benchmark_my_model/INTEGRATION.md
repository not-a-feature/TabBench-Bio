# Register a model for resumable runs

Use this workflow to run a model through the benchmark grid, resume interrupted
jobs and merge its SQLite results with a reference. For a direct scikit-learn
comparison and a local report, follow the [quickstart](README.md#quickstart).

Work from a source checkout with the Python 3.13 core environment active.

## Add an adapter

Create an AutoGluon adapter under [src/tabbench_bio/models](../src/tabbench_bio/models/).
Use an existing adapter, such as [GeneICL](../src/tabbench_bio/models/geneicl.py),
as a guide to the interface:

- Implement fitting and prediction for the tasks your model supports.
- Fit preprocessing on training rows only and set random seeds.
- Respect the supplied CPU and GPU allocation.
- Return classification probabilities in AutoGluon's class order, or numerical
  predictions for regression.

Register the adapter in [model_registry.py](../src/tabbench_bio/model_registry.py):

```python
"MYMODEL": ModelSpec(
    display="My Model",
    category="Tabular Foundation",
    device="gpu",  # or "cpu"
    environment="my-model",
    adapter="tabbench_bio.models.my_model:MyModel",
),
```

Use `supported_tasks=("classification",)` for a classification-only model, or
`max_classes` for a class limit. Use a new key for a different checkpoint or method.
Add the key to `configs/models/all.json` only if it should join the default roster;
no CLI or grid edits are needed to run a registered key directly.

## Install its environment

Reuse a compatible [environment profile](../environments/README.md), or create
`environments/my-model.txt`:

```text
-r ../requirements-autogluon-fork.txt
-e .[autogluon]
# Add your backend and its version constraints here.
```

The AutoGluon fork supports the wide feature tables used by this benchmark.
Install the profile from the repository root:

```sh
uv venv .venvs/my-model --python 3.13
uv pip install --python .venvs/my-model/bin/python -r environments/my-model.txt
```

On Windows, use `.venvs/my-model/Scripts/python.exe`. Keep the core environment
active when launching commands; the runner selects the model's environment.
`--python /path/to/python` overrides that selection. Dependencies are not installed
during a run.

Check that the adapter imports before starting experiments:

```sh
tabbench-bio doctor --model MYMODEL
```

This checks installation and importability, not predictive performance. Test every
supported task type before committing to the full grid.

## Run and resume

Start with the reference cell, then expand to the full grid:

```sh
tabbench-bio MYMODEL --threads 8
tabbench-bio MYMODEL --threads 8 --full-grid
```

The reference cell uses up to 10,000 features and 100 training rows, with five
folds. DUMMY runs first on each split and gates the other model's fit. Results go
to `results/mymodel/results.sqlite`. Repeat the command to resume completed work;
adding `--full-grid` fills the remaining cells.

Use `--output results/my-experiment` for a different result directory. Keep the
model, checkpoint, environment and settings unchanged when resuming. Changed
settings require new output. See [TUNING.md](TUNING.md) for tuned variants.

Prepare dataset credentials and checkpoint access before running. Keep
`split_manifest.json` with the results. To reuse a reference run's frozen folds,
copy its manifest into the new result root before starting and change only
`experiment_id` to the new root's directory name.

## Merge and generate plots

A model-only run needs reference predictions containing RF to compute anchored Elo:

```sh
tabbench-bio merge reference.sqlite results/mymodel/results.sqlite \
  --output results/merged/results.sqlite

tabbench-bio leaderboard --sqlite results/merged/results.sqlite \
  --cell cap_10000_n100 --plot --csv leaderboard.csv
```

The output database must be new. The merge checks settings and held-out targets,
removes duplicate attempts and leaves both inputs unchanged. Conflicting results
stop the merge. RF anchors Elo at 1000.

For an interactive dashboard, build a local preview:

```sh
tabbench-bio leaderboard results/merged/results.sqlite --out preview --workers 4
python -m http.server 8000 --directory preview
```

Open `http://localhost:8000/`. Building does not publish anything. See the
[main README](../README.md#merge-results-and-build-the-website) for the website workflow.

## Run on a cluster

Make the core environment, model environment, data and output paths available on
the compute node. Within a Slurm allocation:

```sh
srun /path/to/core/.venv/bin/tabbench-bio MYMODEL \
  --threads "$SLURM_CPUS_PER_TASK" --output results/my-model
```

Request the memory, wall time and GPU your model needs. GPU adapters require CUDA
unless they support `--device cpu`. Run one registered-model job per output
directory, and resubmit the same command to resume. Keep `--threads` consistent
with the experiment's resource settings.
