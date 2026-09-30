---
name: biomedical-tabular-model-selection
description: Shortlist machine learning methods for high-dimensional biomedical tabular data using TabBench-Bio. Match modality, sample/feature budgets, metrics and compute constraints; account for uncertainty, failed fits and training-data overlap.
license: Apache-2.0
---

# Biomedical Tabular Model Selection — TabBench-Bio

Use current evidence from https://tabbench-bio.eu to shortlist machine learning methods
and recommend a local validation plan. Do not hard-code winners, scores or overlap flags.

## 1. Match the user's problem

Identify task, modality, labelled training samples, usable features, target metric,
grouping/imbalance, and compute constraints. Ask only for details that change the
shortlist; otherwise state assumptions. A task description and counts suffice.

Start with [llms.txt](https://tabbench-bio.eu/llms.txt). Use
[data reference](https://tabbench-bio.eu/skills/biomedical-tabular-model-selection/references/data.md)
for JSON endpoints and joins.
Record the snapshot date; disclose unavailable evidence rather than inventing it.

Choose matching `cell_options` and filter `domain_elo` by cell, domain and metric.
When an exact budget is unavailable, inspect neighbouring grid points and state
the mismatch. `null` means full/uncapped, never zero. Aggregate reference rankings
are not modality-specific evidence; use task-specific scores and metric directions
for regression or individual datasets. Report limited target coverage.

### Minimal lookup example

This standard-library Python example queries an *example* operating point. Replace
its budgets, modality and metric with those appropriate to the user's problem.
It fetches metadata only and does not fit models or download biomedical datasets.

```python
import json
from urllib.request import urlopen

url = "https://tabbench-bio.eu/data/dashboard.json"
with urlopen(url, timeout=30) as response:
    data = json.load(response)
view = data["analysis_views"]["strict"]
cell = next(c for c in data["cell_options"]
            if c["feature_cap"] == 10000 and c["n_train"] == 100)
rows = [r for r in view["domain_elo"]
        if r["cell"] == cell["id"] and r["domain"] == "Gene expression"
        and r["metric"] == "f1_macro"]
print("View: strict", "Snapshot:", data["meta"]["snapshot_utc"], "Cell:", cell["label"])
for row in sorted(rows, key=lambda r: r["Elo"], reverse=True):
    model = data["models"][row["model_id"]]
    if row["model_id"] in data["meta"]["plot_excluded_models"]:
        continue
    print(model["display"], row["Elo"], row["Elo_lo"], row["Elo_hi"],
          row["n_targets"], "training-data overlap:", model["training_data_overlap"])
```

## 2. Compare candidates fairly

- The website defaults to **strict** results, matching its default rankings,
  nominal-budget comparisons and the paper.
  Always name the view. Adaptive results may reuse a smaller
  sample budget after training OOM; label them as sensitivity results, not nominal
  measurements. Never mix views.
- Compare Elo, 95% intervals and `n_targets` at the same operating point. Elo is not
  accuracy, and individual intervals do not establish pairwise significance.
- Identify tuned families using `models[model_id].tuned_from`, which is exported from
  the JSON model registry. Untuned Random Forest remains the Elo reference at 1,000.
  A paired bar shows untuned Elo plus the signed tuning change; its diamond is tuned
  Elo. Only the tuned interval is drawn for a pair; the hidden untuned interval is
  still available in the data. The interval is not uncertainty on the tuning change.
  Plotted error bars stop at zero; exports retain the original statistical interval,
  including a negative lower bound.
  Budget-response line plots omit an untuned parent when its tuned version is shown;
  underlying exports retain both versions. Unpaired models remain visible.
- The completed tuned snapshot uses a small predefined grid under a one-hour budget.
  Read the frozen run configuration for its actual candidate set. A changed registry
  does not retroactively change previously measured scores.
- System configuration: NVIDIA L40S, 48 GB VRAM, 16 CPU cores and 80 GB RAM.
- Include a competitive simple baseline and a few complementary candidates.
  Compare `cost_grid` at the same cell/domain; its per-fold timings are not hardware
  guarantees or per-patient latency.
- Join `models[model_id]` for current flags. For each `training_data_overlap: true`,
  disclose: **Part of the benchmark training data was used in the training process
  of this model.** Prefer unflagged evidence for the main shortlist; discuss flagged
  models separately when relevant. A false flag does not prove absence of overlap.
- Check model version, `regular_max_features` and failed-fit coverage. Above-limit
  results may rely on modified adapters. Distinguish failed fits scored at chance
  from successful fits; missing results are not zero.
- Treat AutoGluon as the benchmark's separate one-hour AutoML reference. Respect
  `plot_excluded_models` when interpreting displayed rankings.

## 3. Recommend and validate

Return the task/modality, budgets, metric, analysis view and snapshot date, plus a
compact table of candidates, scores/uncertainty, cost and rationale. Cite the exact
artifacts used and separate measurements from inference. If relevant evidence is
missing, propose a baseline experiment instead of forcing a ranking.

Validate candidates on the user's data with common splits and metrics. Fit
preprocessing and feature selection within training folds; preserve patient,
batch or temporal grouping and keep final test data separate from tuning.
Benchmark evidence guides a shortlist, not a universal winner or permission to
start expensive experiments.

Cite [CITATION.cff](https://tabbench-bio.eu/CITATION.cff) or
[CITATION.bib](https://tabbench-bio.eu/CITATION.bib):
[arXiv:2609.07441](https://arxiv.org/abs/2609.07441),
[DOI](https://doi.org/10.48550/arXiv.2609.07441).
