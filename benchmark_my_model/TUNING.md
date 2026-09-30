# Add a tuned model

You can compare a model's default settings with a small grid of alternatives.
Give the tuned version its own name, choose the settings to try, and run it on the
same benchmark folds. TabBench selects settings using training data only, then
evaluates the chosen model on the held-out test data.

Choose an existing parent adapter supported by `TUNABLE_MODELS` in
[tuning.py](../src/tabbench_bio/tuning.py). You only need a JSON definition to add
a tuned variant of a supported parent. For an unsupported parent, first add and
test its adapter using the [integration guide](INTEGRATION.md), then add support
for it in the tuning wrapper.

## Choose your grid

Save a JSON list like the one below as `my_grid.json`. This is a template: replace
`BASE_MODEL`, `parameter_name` and the example values with settings your adapter
accepts. Set `device` to `cpu` or `gpu` and `environment` to an installed
[environment profile](../environments/README.md) containing that adapter.

```json
[
  {
    "key": "MY-MODEL-TUNED",
    "base_model": "BASE_MODEL",
    "device": "cpu",
    "environment": "my-profile",
    "tuning": {
      "protocol": "my-grid-v1",
      "selection": "inner_holdout",
      "validation_fraction": 0.2,
      "seed": 42,
      "search_fraction": 0.8,
      "grid": [
        {},
        {"parameter_name": [0.1, 1.0, 10.0]}
      ]
    }
  }
]
```

`key` is the name saved in your results. `base_model` chooses the adapter; the
package does not infer it from the name. The empty `{}` includes the installed
library's default settings and is required. This example tries four candidates
in total.

If you list several parameters in one grid, TabBench tries every combination.
Use separate grid entries for combinations you want to keep separate. For
different classification and regression settings, replace `grid` with an object
containing a `classification` grid and a `regression` grid. Parameter names must
match the AutoGluon adapter, which may use different names from the underlying
library.

Choose the grid before looking at test scores. Give a revised experiment a new
protocol version, model key and output directory. Resume and merge checks reject
different tuning definitions under the same key.

## Run the benchmark

Install the core environment and your chosen model profile, then run from the
editable checkout:

```sh
tabbench-bio MY-MODEL-TUNED --model-config my_grid.json --output results/my-tuned-model
```

Start with this reference-cell run to check the configuration and resource use.
Add `--full-grid` to evaluate all configured feature and sample budgets. Use
`--threads` and `--device` to match your available resources. For batch execution,
pass the same model key and `--model-config` to your
[cluster job](INTEGRATION.md#run-on-a-cluster).

For `tabbench-bio run --config ...`, set `models` to your roster file, or use model
keys with an explicit `model_tuning` map. Keep `autogluon_presets` set to
`medium_quality`, with `ensemble: false` and `optimize: false`.

## How settings are selected

For each outer fold, TabBench sets aside part of the available training data for
validation. Classification splits are stratified; regression splits are random.
If biological groups repeat, whole groups stay together. Classification requires
every class in both parts. Very small sample budgets may not allow this and will
produce a failed evaluation.

All candidates use the same validation rows. Within the benchmark's existing
feature cap and outer-training feature filtering, imputation, feature generation
and regression target scaling are fitted on the inner training rows only.
Selection uses macro-F1 for classification and RMSE for regression. Exact ties
go to the first candidate in grid order.

The chosen settings are fitted again using all permitted outer-training rows,
including fresh preprocessing. Only the resulting test predictions contribute
to the reported metrics and fold-level Elo. Selection uses one inner holdout,
not an inner cross-validation loop.

## Keep an eye on the budget

Search and refitting share the per-model time budget configured for the run.
`search_fraction` sets the maximum fraction available to search; the rest is
reserved for refitting. For example, `0.8` allows search to use 80% of the budget.
Unused search time is also available for refitting. The standard benchmark
configuration uses one hour per dataset, outer fold and model variant.

Candidates run sequentially. An expensive candidate can use the remaining search
allocation, so a larger grid does not guarantee that every candidate is tried.
If the budget ends between candidates, TabBench chooses among those completed
and records `search_truncated`. A candidate error, no completed candidate, or a
failed refit makes the evaluation fail. Reported runtime includes the search and
refit.

Each saved SQLite attempt includes a `tuning` record with the grid, library
versions, validation row identifiers, candidate scores and timings, and the
selected settings. These scores describe selection, not test performance.
AutoGluon stores losses with a negative sign; regression selection scores use
the inner-training target scale. Reported test metrics use the original units.

### Resume and merge results

An unchanged run resumes completed benchmark units from SQLite. Candidate scores
and timings are retained for auditing; fitted candidate models are deleted after
scoring. Changing a grid requires a fresh output directory and model key.
The frozen configuration attached to each result defines its protocol. Merging
different tuning specifications under the same model key is rejected.

## Compare your results

Merge your results with compatible reference results, then create a leaderboard:

```sh
tabbench-bio merge results/reference/results.sqlite results/my-tuned-model/results.sqlite --output results/comparison/results.sqlite
tabbench-bio leaderboard --sqlite results/comparison/results.sqlite --cell cap_10000_n100 --plot
```

Both inputs must use the same cells, frozen folds and compatible settings. The
reference results supply the Elo anchor and comparison models. Adding models can
change Elo ratings, so describe which models are included when sharing the plot.
