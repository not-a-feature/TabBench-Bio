# Add a model and run TabBench-Bio

Add an adapter, register its key and choose its environment profile. Then run the
benchmark, merge its results and build a leaderboard. The same model command works
locally and in a cluster job.

## Install once

```bash
git clone https://github.com/not-a-feature/TabBench-Bio.git
cd TabBench-Bio
uv venv --python 3.12
uv pip install -e '.[bio]'
source .venv/bin/activate
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1`. This is the core environment
for launching runs, merging databases and generating leaderboards. Model backends
live in separate environments. Keep the editable clone: the model command reads its
grid, dataset lists and profiles.

## Add the adapter and registry entry

Add `src/tabbench_bio/models/my_model.py` with an AutoGluon adapter.
Start from an existing adapter with a similar fit/predict interface.
Implement fitting and prediction, select checkpoints explicitly, fix random seeds
and honour the assigned device and CPU budget. Fit preprocessing on training data only.
Classification probabilities must follow AutoGluon's class order.
Regression returns numerical predictions.

Then add your key to `CUSTOM_MODELS` in `src/tabbench_bio/models/custom.py`:

```python
"MYMODEL": {
    "adapter": "tabbench_bio.models.my_model:MyModel",
    "environment": "my-model",
    "device": "gpu",  # or "cpu"
    "max_features": None,
    "classification_only": False,
},
```

The entry selects the adapter, environment and device, and whether to skip regression.
`max_features` records the regular feature limit for reporting.
Enforce any hard input limit in the adapter.

No CLI or grid edits are needed. Keys already listed in `configs/models/all.json`
also work. Use a new key for a different method or checkpoint version.
Before a long run, test a small fit for each supported task type in the selected
environment on the intended hardware. Check probability order, saving and reloading,
and unseen categories. Check the backend's preprocessing too: some fit transforms
on training and test rows together.

## Choose an environment profile

Compatible models can share a profile. For a new dependency set, add
`environments/my-model.txt`:

```text
-r ../requirements-autogluon-fork.txt
-e .[bio,autogluon]
# Add your model backend and its version constraints here.
```

Keep the AutoGluon fork: it removes the 500-feature cap on tabular foundation models.
Put model-specific packages in this file. You do not need to add a package extra or
change the CLI. Pin Git dependencies to a commit and downloaded weights to a revision.
Profile names use lower-case letters, digits, hyphens and underscores,
starting with a letter or digit.

From the repository root, install the profile once on each machine:

```bash
uv venv .venvs/my-model --python 3.12
uv pip install --python .venvs/my-model/bin/python -r environments/my-model.txt
```

On Windows, use `.venvs/my-model/Scripts/python.exe` in the install command.
Keep the core `.venv` active. `tabbench-bio MYMODEL` selects `.venvs/my-model/`
for the hardware check, adapter check and every cell's execution. It prints the
profile and interpreter path, and stops before fitting if the environment is missing.
It does not install or update packages during a run.

Existing models declare profiles in `configs/models/all.json`. Custom entries take
precedence. See [available profiles](../environments/README.md) for the bundled choices.
These files declare requirements, not complete dependency locks. Keep an environment
unchanged while running or resuming a benchmark. The profile name and interpreter
path are recorded with the hardware details, but package changes are not checked.

## Run

```bash
tabbench-bio MYMODEL              # reference: 10,000 features, 100 samples, 5 folds
tabbench-bio MYMODEL --full-grid  # all 28 feature/sample cells
```

The terminal shows cell and fold progress, followed by pass/fail/skip counts and
the database path. Results go to `results/mymodel/results.sqlite`, with transactional
writer files saved during the run.

Repeat the command to resume. You can finish the reference cell first, then add
`--full-grid` to run the remaining cells. Changed frozen cell settings need a new
`--output` directory.

One fit runs at a time, using at most one GPU. The command selects the first GPU in
`CUDA_VISIBLE_DEVICES`. Set `--threads` explicitly to match your CPU allocation;
the command's default thread count is independent of that allocation.
Hardware details are saved in the run records and `hardware.jsonl`. The runner may
warn when hardware differs from the benchmark reference, because resource changes
can affect timings and time-limited results.

A GPU model stops if CUDA is unavailable. If the adapter supports CPU execution,
select it explicitly:

```bash
tabbench-bio MYMODEL --threads 16 --device cpu
```

You can also choose the result and cache directories:

```bash
tabbench-bio MYMODEL --output results/my_experiment --cache-dir /path/to/cache
```

Prepare checkpoint access, dataset credentials and a writable dataset cache before
starting. `TABBENCH_BIO_LOCAL_DIR` selects the embedding directory, and
`TABBENCH_CACHE_DIR` selects the shared cache. Raw data, split settings and
preprocessing must match the reference benchmark.

To reuse a `split_manifest.json`, place it in the new result root and change only
its `experiment_id` to that directory's name. Preserve every row identity and hash.

## Merge and generate plots

```bash
tabbench-bio merge /path/to/reference.sqlite results/mymodel/results.sqlite
tabbench-bio leaderboard
```

The merge creates `results/merged/results.sqlite`. That path must be new.
It combines model and dataset rosters, removes duplicate attempts and preserves
source metadata. Incompatible settings, corrupt artefacts, mismatched targets or
conflicting successful predictions stop the merge. Inputs stay unchanged.
You can also pass result directories containing writer databases.

The leaderboard command reads the predictions and builds a static website in
`website/`, relative to the current directory. In a repository clone, that directory
is the Codeberg `pages` submodule. Initialise it before building:

```bash
git submodule update --init website
```

The build includes the dashboard, dataset explorer, model cards, three analysis views
and fitting/prediction cost plots. It prints fold-level Elo and saves per-cell PNG
plots and CSV tables in `website/local/<cell>/overall/`. Git ignores that directory.
It generates no per-cell HTML reports or SVG plots. Raw datasets, a GPU and LaTeX
are unnecessary for this step.

```bash
tabbench-bio leaderboard results/merged/results.sqlite --workers 4
python -m http.server 8000 --directory website
```

Open `http://localhost:8000/`. Use `--out preview` to build in a separate directory.

The main JSON exports are `data/dashboard.json`, `data/leaderboard.json` and
`data/datasets/index.json`, alongside the per-dataset score files. Upload the site
to a static webserver, or just `data/` to an existing compatible TabBench-Bio site.
Building does not publish anything.

To publish through Codeberg, switch the submodule to `pages` and update it before
building:

```bash
git -C website switch pages
git -C website pull --ff-only
```

After building and reviewing the changes, commit and push inside the submodule:

```bash
git -C website add --all
git -C website commit -m "Update leaderboard"
git -C website push origin pages
```

### Build settings and caches

The default is 2,000 bootstrap rounds, matching the paper.
Change it with `--bootstrap-rounds`. `--workers` controls both fold metrics and Elo,
with one numerical-library thread per worker. Progress bars show completed folds,
Elo pools and estimated time remaining.

Reuse the same output directory to keep cached results.
Elo is saved in `data/dashboard.json` after a completed build. Changes to fold scores,
bootstrap settings or the Elo implementation invalidate the affected entries.

Fold metrics are saved separately in `<out>.cache/fold_metrics.sqlite`.
Entries match prediction, probability and target hashes, metric code and numerical-library
versions. Completed batches survive an interrupted build. Keep this cache locally for
later merges and builds. Deleting it forces recalculation.
The original result database and its release checksum stay unchanged.

The default reference cell is `cap_10000_n100`, or the first available cell.
Choose another with `--reference-cell`. Custom model keys get a neutral display style.
`--results-url URL` adds a download link for the exact input SQLite file.
The database itself is not copied into the website.

### Reading and releasing results

RF anchors Elo at 1000. Rankings need RF and comparable completed folds.
Coverage remains visible for partial runs.

Strict results score failed fits using recorded DUMMY metrics and omit incomplete
model-target fold sets. Adaptive results reuse compatible lower-sample runs after
memory failures. Conditional results omit failures, so their target pools can differ.
Cost plots omit missing timings.

To choose explicit files, one cell or a task type:

```bash
tabbench-bio merge old.sqlite new.sqlite --output results/updated/results.sqlite
tabbench-bio leaderboard results/updated/results.sqlite --cell cap_10000_n100
tabbench-bio leaderboard results/updated/results.sqlite --task classification
```

Use a frozen snapshot for a reproducible build, and wait for jobs to finish before
a final release. Snapshots of active writer files are individually consistent,
but can represent different points in the run. Allow disk space for the combined
database and the largest input snapshot.

Continue training in the per-model directories. Keep released databases unchanged
and publish later results as a new release.

## Run on a cluster

Run the same model command inside your scheduler's job allocation. Install the core
environment and the model profile where compute nodes can access them. Make the
checkout, data cache and result directory available to the job, and configure any
dataset credentials or checkpoint access before submitting it.

Choose CPU count, memory, GPU requirements and wall time for your model. Keep model
threads within the allocation and use one benchmark worker per GPU. The scheduler's
wall-time limit covers the entire job; it is separate from the per-fit benchmark
budget. Submit only one model-command job per result directory at a time.

For example, save this as `run_benchmark.sbatch` on a Slurm cluster. The resource
values are illustrative: adapt them and add your site's account or partition
directives. For a GPU model, uncomment the GPU request and adjust its syntax if
your cluster requires a GPU type.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=tabbench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/benchmark.%j.out
#SBATCH --error=logs/benchmark.%j.err
##SBATCH --gres=gpu:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository checkout}"
srun .venv/bin/tabbench-bio "$@" --threads "${SLURM_CPUS_PER_TASK:?}"
```

Submit from the repository root after creating the log directory:

```bash
mkdir -p logs
sbatch run_benchmark.sbatch MYMODEL --output results/my-model
```

For a tuned model, pass the JSON definition from the [tuning guide](TUNING.md):

```bash
sbatch run_benchmark.sbatch MY-MODEL-TUNED --model-config my_grid.json --output results/my-tuned-model
```

Add `--full-grid` to either submission to run every configured feature and sample
budget. Use `--device cpu` if you want CPU execution and the adapter supports it;
otherwise request the GPU required by the model's registry entry. The model command
selects its installed environment profile automatically.

If a job stops at its wall-time limit, submit the same command again after it exits.
Completed benchmark units are resumed from the same result directory. Keep the
model definition, split settings and environment unchanged between submissions.
For another scheduler, replace the allocation directives and launch command while
keeping the TabBench command and paths the same.

The lower-level `run` command and `scripts/feature_sweep.py` use the interpreter
that launches them; activate a compatible model profile when using those directly.

For a small comparison using scikit-learn alone, see [Benchmark my model](README.md).
That helper exports metrics but does not write SQLite attempt bundles for merging.
