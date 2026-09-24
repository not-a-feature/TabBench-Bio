# Add a model and run TabBench-Bio

Add an adapter, register its key and choose its environment profile. Then run the
benchmark, merge its results and build a leaderboard. The same model command works
locally and on TCML.

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
environment on the intended hardware.

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
change the CLI. Profile names use lower-case letters, digits, hyphens and underscores,
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
`CUDA_VISIBLE_DEVICES` and respects Slurm's allocation. It defaults to 32 model/library
threads. The TCML reference reserves 18 Slurm CPUs and uses 32 model threads.

The command warns if the allocation differs from 18 Slurm CPUs, the model thread
count differs from 32 or the GPU is not an NVIDIA L40S. Outside Slurm, it also warns
when fewer than 32 CPUs are available. These differences can affect timings and
time-limited results. Hardware details are saved in the run records and `hardware.jsonl`.

A GPU model stops if CUDA is unavailable. If the adapter supports CPU execution,
select it explicitly:

```bash
tabbench-bio MYMODEL --threads 16 --device cpu
```

You can also choose the result and cache directories:

```bash
tabbench-bio MYMODEL --output results/my_experiment --cache-dir /path/to/cache
```

Prepare checkpoint access, dataset credentials and local embedding files before
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

## Run on TCML

On TCML, install the core environment and the model's profile from the checkout.
Export credentials and data paths, then submit the reference run:

```bash
mkdir -p logs
sbatch scripts/run_model_tcml.sbatch MYMODEL
```

For the full grid, submit this instead:

```bash
sbatch scripts/run_model_tcml.sbatch MYMODEL --full-grid
```

Submit one job per run directory. The launcher requests one L40S, 18 Slurm CPUs and
90 GiB RAM, with 32 model threads by default. It runs the same resumable command
used locally, selecting the environment from the model registry. Export any access
tokens or checkpoint paths required by the backend. For TabPFN, the launcher can
source a token file through `TABPFN_CREDENTIAL_FILE`.

GPU runs use one benchmark worker per GPU. The multi-model feature sweep follows
the same policy: model entries need `"device": "gpu"`, with no `solo` flag.
The lower-level `run` command and `scripts/feature_sweep.py` use their current Python
interpreter. Launch them from a profile that supports all selected models.
Installing or registering a model submits no jobs.

For a small comparison using scikit-learn alone, see [Benchmark my model](README.md).
That helper exports metrics but does not write SQLite attempt bundles for merging.
