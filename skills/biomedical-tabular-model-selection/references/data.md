# Published data for model selection

All endpoints are public, read-only HTTPS files; no account or API key is needed.
Use the current responses and check their schema before relying on these fields.

| URL | Use |
| --- | --- |
| https://tabbench-bio.eu/data/dashboard.json | Modality/metric rankings, costs, fit coverage, model flags, strict and adaptive views |
| https://tabbench-bio.eu/data/leaderboard.json | Smaller aggregate ranking export; not a substitute for modality-specific rankings |
| https://tabbench-bio.eu/data/datasets/index.json | Dataset task, source, metric direction, available cell descriptions and score-file names |
| https://tabbench-bio.eu/datasets.html | Human-readable dataset explorer |
| https://tabbench-bio.eu/artifacts.html | Available exports, including whether the SQLite download is published |

## Dashboard joins

- `models` is keyed by model ID. Read display name, family, regular feature limit,
  and `training_data_overlap` here; copies of metadata embedded in older rows may
  lack the latest flags.
- `cell_options`: `id`, `feature_cap`, `n_train`, `label`. Match cells by the two
  budgets rather than constructing IDs. A null budget means full/uncapped.
- `analysis_views.strict` and `.adaptive` contain the respective `domain_elo`,
  `cost_grid`, `model_card_coverage`, `reference` and other view-specific arrays.
  Top-level model metadata is shared. Check view availability explicitly.
- `domain_elo`: filter `cell`, `domain`, `metric`; join on `model_id`.
  Read `Elo`, `Elo_lo`, `Elo_hi`, `n_targets`.
- `cost_grid`: join `cell`, `domain`, `model_id`; read `train_time_s`,
  `inference_time_s` and relevant metric values. Do not join costs from another
  cell or treat per-fold inference time as per-patient latency.
- `model_card_coverage`: counts of `successful_fits` and `failed_fits` for a
  model/cell/domain. The all-domain aggregate may not have a matching coverage row;
  inspect the relevant modality rows instead of inventing an all-domain count.
- `meta.snapshot_utc` identifies the exported snapshot. Monitoring progress may
  include runs newer than the completed aggregation used for the displayed scores.

## Minimal lookup example

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
print("Snapshot:", data["meta"]["snapshot_utc"], "Cell:", cell["label"])
for row in sorted(rows, key=lambda r: r["Elo"], reverse=True):
    model = data["models"][row["model_id"]]
    if row["model_id"] in data["meta"]["plot_excluded_models"]:
        continue
    print(model["display"], row["Elo"], row["Elo_lo"], row["Elo_hi"],
          row["n_targets"], "training-data overlap:", model["training_data_overlap"])
```

## Dataset-specific scores

Find the dataset in `datasets/index.json` using its task, modality and source.
Each `datasets` entry provides `scores_file`; fetch that filename relative to
`https://tabbench-bio.eu/data/datasets/`. Do not guess hashed filenames.
The response has `dataset_id` and `scores`; each score has `cell`, `model_id`, and
`values` keyed by metric. Use the index entry's `metrics[].better` direction
(`high` or `low`), especially for regression errors. These are exported dataset
scores, not per-fold observations or automatically paired significance tests.
Do not label them adaptive unless their provenance explicitly says so.

If deeper analysis requires SQLite, read `raw_exports` in the dashboard and check
`available`, URL, size and checksum first. Do not infer a download URL from a
filename or silently fall back to unpublished local results.
