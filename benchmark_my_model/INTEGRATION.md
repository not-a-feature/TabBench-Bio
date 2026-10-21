# Integrate a model with TabBench-Bio

## Use a Python model without a clone

Most models only need a `create_model(task)` factory returning a cloneable
scikit-learn classifier or regressor. Start from [my_model.py](my_model.py),
which provides SVC and SVR with training-only imputation and scaling.

With Python 3.11 or 3.12:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "tabbench-bio[bio]"
tabbench-bio evaluate --model my_model.py --task overall
```

Before the first PyPI upload, install a built wheel with `[bio]` instead.
The command downloads missing data and the latest reference database, checks
frozen test rows and labels, and fits only your model. Open
`my_model_results/report.html` for the combined leaderboard and plots.
Use `--task classification` or `--task regression` to select one task type.

This workflow writes CSV metrics and release provenance. It requires a fresh
output directory and cannot resume or produce SQLite attempt bundles for merging.
See [the model guide](README.md) for dataset selection and other options.

## Register an adapter for resumable runs

Adapter development requires a source checkout. Install it into your core environment:

```bash
uv pip install -e '.[bio]'
```

Add an AutoGluon adapter at `src/tabbench_bio/models/my_model.py`, using an existing
adapter as a starting point. Fit preprocessing on training rows only, fix random
seeds, and honour the assigned CPU and device budget. Classification probabilities
must follow AutoGluon's class order; regression returns numerical predictions.

Add its key to `MODEL_REGISTRY` in `src/tabbench_bio/model_registry.py`:

```python
"MYMODEL": ModelSpec(
    display="My Model",
    category="Tabular Foundation",
    device="gpu",  # or "cpu"
    environment="my-model",
    adapter="tabbench_bio.models.my_model:MyModel",
),
```

Use `supported_tasks=("classification",)` or `max_classes` for unsupported tasks.
`max_features` records the regular feature limit; enforce hard limits in the adapter.
No CLI or grid edits are needed. Add the key to `configs/models/all.json` only if
it should join the default roster. Use a new key for a different checkpoint or method.

### Install the backend

Reuse a compatible [environment profile](../environments/README.md), or create
`environments/my-model.txt`:

```text
-r ../requirements-autogluon-fork.txt
-e .[bio,autogluon]
# Your backend and version constraints
```

Keep the AutoGluon fork, which removes the 500-feature cap for foundation models.
From the checkout root:

```bash
uv venv .venvs/my-model --python 3.12
uv pip install --python .venvs/my-model/bin/python -r environments/my-model.txt
```

In a checkout or exported workspace, registered commands select `.venvs/<profile>/`.
An installed package outside a workspace uses the current interpreter.
`--python /path/to/python` overrides either choice. To export the bundled profiles
and configurations from an installed package, use `tabbench-bio init my-benchmark`.
The command does not install backend dependencies during a run.

### Run and resume

Keep the core environment active:

```bash
tabbench-bio MYMODEL --threads 8              # reference cell: up to 10,000 features and 100 training rows, five folds
tabbench-bio MYMODEL --threads 8 --full-grid  # all 28 bundled cells
```

Results go to `results/mymodel/results.sqlite`. Repeat the command to resume;
adding `--full-grid` after the reference run fills the remaining cells.
Use `--output` for another result directory. Keep model, checkpoint, environment
and frozen settings unchanged when resuming; changed settings require new output.
Tuned variants use `--model-config`; see [the tuning guide](TUNING.md).

One fit runs at a time on at most one GPU. A GPU model requires CUDA unless you
explicitly select `--device cpu` and the adapter supports it. Set `--threads` to
match your allocation: the registered command's default is 32, independent of
scheduler allocation. Hardware and interpreter details are recorded with results.

Prepare dataset credentials and checkpoint access before running. Keep the resolved
configuration and `split_manifest.json` with the results. To reuse a reference
manifest, copy it to the new result root and change only `experiment_id` to that
root's name; preserve all row identities and hashes.

## Compare registered results

Registered runs need reference predictions containing RF to compute anchored Elo:

```bash
tabbench-bio merge /path/to/reference.sqlite results/mymodel/results.sqlite \
  --output results/merged/results.sqlite
tabbench-bio leaderboard --sqlite results/merged/results.sqlite \
  --cell cap_10000_n100 --plot --csv leaderboard.csv
```

The merge destination must be new. Inputs stay unchanged; incompatible settings,
targets or conflicting successful predictions stop the merge. RF anchors Elo at
1000. Coverage reflects completed comparable folds.

To build the full dashboard instead:

```bash
tabbench-bio leaderboard results/merged/results.sqlite --out preview --workers 4
python -m http.server 8000 --directory preview
```

Open `http://localhost:8000/`. Building does not publish anything. The default
confidence intervals use 2,000 target-bootstrap rounds; `--bootstrap-rounds`
changes that count. Keep released databases unchanged and publish later results
as a new release. See the [main README](../README.md) for website details.

## Run on a cluster

Use the same command within your scheduler allocation. The core environment,
backend profile, data and output paths must be accessible on compute nodes.
For example, within a Slurm allocation:

```bash
srun /path/to/core/.venv/bin/tabbench-bio MYMODEL \
  --threads "$SLURM_CPUS_PER_TASK" --output results/my-model
```

Request the memory, wall time and GPU required by your adapter. Run only one
registered-model job per result directory. After a job stops, resubmit the same
command to resume completed units. The `evaluate` workflow above does not resume.
