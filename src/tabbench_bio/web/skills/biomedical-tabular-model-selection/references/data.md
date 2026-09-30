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
  Top-level rankings follow the Strict website default, declared by
  `meta.primary_analysis_view`, matching the paper analysis. Top-level model
  metadata is shared. Check view availability explicitly.
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

## Tuned models and timing

Use `models[model_id].tuned_from` to join a tuned configuration to its parent;
do not infer this relation from names. For a paired bar, untuned Elo is the base,
`tuned.Elo - parent.Elo` is the signed extension, and the diamond marks tuned Elo.
Untuned Random Forest remains anchored at 1,000. Only the tuned configuration's
95% interval is drawn for a pair; it is not an interval for the difference.
Plotted error bars are clipped at zero. The original statistical intervals remain
unchanged in the exports, including any negative lower bound.
The parent remains available in the JSON even when omitted from budget-response
line plots. Only parent/tuned rows at the same cell, modality, metric and analysis
view are comparable.

The completed tuning snapshot uses a small predefined grid under a one-hour budget.
Read the frozen configuration attached to a result before reporting its search size.
System configuration: NVIDIA L40S with 48 GB VRAM, 16 CPU cores and 80 GB RAM.

## Minimal lookup example

See the [minimal lookup in the main skill](../SKILL.md#minimal-lookup-example).

## Dataset-specific scores

Find the dataset in `datasets/index.json` using its task, modality and source.
Each `datasets` entry provides `scores_file`; fetch that filename relative to
`https://tabbench-bio.eu/data/datasets/`. Do not guess hashed filenames.
The response has `dataset_id` and `scores`; each score has `cell`, `model_id`, and
`values` keyed by metric. Use the index entry's `metrics[].better` direction
(`high` or `low`), especially for regression errors. These are exported dataset
scores, not per-fold observations or automatically paired significance tests.
These dataset-specific exports use Strict nominal-cell scores, matching the homepage default.

If deeper analysis requires SQLite, read `raw_exports` in the dashboard and check
`available`, URL, size and checksum first. Do not infer a download URL from a
filename or silently fall back to unpublished local results.
