---
name: biomedical-tabular-model-selection
description: Select and compare methods for high-dimensional biomedical tables using current TabBench-Bio evidence. Match task, modality, data budgets and compute; interpret uncertainty, failed fits and training-data overlap.
license: EUPL-1.2
---

# Select models with TabBench-Bio

Identify the task, modality, labelled training samples, usable features, metric,
grouping and compute constraints. Ask only for details that change the shortlist;
otherwise state assumptions. The user need not upload their dataset.

## Retrieve current evidence

Start with [llms.txt](https://tabbench-bio.eu/llms.txt) for benchmark context.
Retrieve complete JSON with an HTTP client when a browser snapshot is incomplete.
If retrieval fails, report the missing evidence; do not substitute remembered or
paper results and present them as current.

| Export | Purpose |
| --- | --- |
| [Dashboard](https://tabbench-bio.eu/data/dashboard.json) | Modality/metric Elo, costs, coverage and model metadata |
| [Leaderboard](https://tabbench-bio.eu/data/leaderboard.json) | Aggregate Elo; does not replace modality-specific evidence |
| [Dataset index](https://tabbench-bio.eu/data/datasets/index.json) | Dataset metadata, metric directions and score-file links |

Use `analysis_views.strict` by default. Read `cell_options`, `domain_elo`,
`cost_grid` and `model_card_coverage` from that view, not the dashboard's top level.
Other views may be inline or linked by `analysis_view_files`; resolve linked paths
against `https://tabbench-bio.eu/`. Join row `model_id` to top-level `models` for
names, `tuned_from`, `regular_max_features` and overlap flags.
Record `meta.snapshot_utc` and check the schema before using an export.

Match cells by `feature_cap` and `n_train`, not guessed IDs. A null budget means
full/uncapped, not zero. Filter `domain_elo` by `cell`, `domain` and `metric`.
If the requested budget or metric is absent, inspect relevant neighbouring cells
or dataset scores and disclose the mismatch; do not extrapolate a measured score.

For regression or individual datasets, follow the index entry's `scores_file`
relative to `https://tabbench-bio.eu/data/datasets/`. Read `scores[].values` and
`metrics[].better` (`high` or `low`). These are Strict dataset scores, not per-fold
observations or paired significance tests. Do not guess hashed filenames.

### Example lookup

Replace these example budgets, modality and metric with the user's setting:

```python
import json
from urllib.request import urlopen

with urlopen("https://tabbench-bio.eu/data/dashboard.json", timeout=30) as response:
    data = json.load(response)
view = data["analysis_views"]["strict"]
cell = next(c for c in view["cell_options"]
            if c["feature_cap"] == 10000 and c["n_train"] == 100)
rows = [r for r in view["domain_elo"]
        if r["cell"] == cell["id"] and r["domain"] == "Gene expression"
        and r["metric"] == "f1_macro"]
assert rows, "No matching published comparison"
print(data["meta"]["snapshot_utc"], "strict", cell["label"])
for row in sorted(rows, key=lambda r: r["Elo"], reverse=True):
    model = data["models"][row["model_id"]]
    print(model["display"], row["Elo"], row["Elo_lo"], row["Elo_hi"],
          row["n_targets"], "training-data overlap:", model["training_data_overlap"])
```

## Interpret comparisons

- Keep cell, domain, metric and analysis view consistent. Strict scores failures
  at a constant/chance predictor. Adaptive may reuse a smaller training cell after
  OOM; report it as sensitivity evidence. Conditional (Failures-omitted) drops
  failures and compares different target pools, giving an optimistic bound.
- Report Elo, 95% intervals and `n_targets`. Random Forest anchors Elo at 1,000;
  Elo is not accuracy. Individual intervals do not establish pairwise significance,
  and a reference-cell leader is not a universal winner.
- For `training_data_overlap: true`, disclose: "Part of the benchmark training data
  was used in the training process of this model." Prefer unflagged evidence for
  the main shortlist; a false flag does not establish absence of overlap.
- Match costs on `cell`, `domain`, `model_id`. Per-fold timings depend on benchmark
  hardware and are not per-patient latency. Coverage uses `successful_fits` and
  `failed_fits`; inspect modality rows if an all-domain row is absent. Missing
  results are not zero. Check feature limits and whether adapters were modified.
- Use `tuned_from` for parent/tuned comparisons. Read the frozen result configuration
  for the actual one-hour tuning grid; current registry edits do not change past
  experiments. Rating intervals are not intervals on the tuning gain. Treat
  AutoGluon as a separate one-hour AutoML reference.
- `meta.plot_excluded_models` controls website display, not candidate validity.
  Apply it when reproducing the displayed ranking, not as a blanket exclusion.

## Recommend and validate

Suggest a simple baseline and a few complementary candidates. State the snapshot,
view, cell and metric; compare scores, uncertainty, coverage, cost and overlap when
available. Cite the exact exports and separate measurements from inference. If
relevant evidence is missing, propose a baseline experiment instead of a ranking.

For local validation, use common splits and metrics, fit preprocessing and feature
selection inside training folds, preserve patient/batch/time grouping, and keep the
final test set separate from tuning. A shortlist does not authorize expensive runs.

Cite [the benchmark paper](https://doi.org/10.48550/arXiv.2609.07441), using
[CITATION.cff](https://tabbench-bio.eu/CITATION.cff) for citation metadata.
