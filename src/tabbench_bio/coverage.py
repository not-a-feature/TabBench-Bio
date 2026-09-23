"""Per-unit run status: what passed, what the design excluded, and what failed.

The predictions step commits one record per ``(key, model)`` attempt. That record is the
only place the pipeline distinguishes a unit the
benchmark excluded **by design** — a declared per-model input-size limit, a degenerate
split, a grid cell whose sample budget exceeds the data — from a unit the model was asked
to fit and **failed** on. Neither appears in the metrics CSVs, so any ranking that reads
only those CSVs cannot tell the two apart and ends up measuring the harness.

One convention, applied by every consumer (leaderboard ``Score``, the per-dataset score
heatmap, and Elo alike):

* ``fail``   — imputed at the run's chance-level baseline (:data:`BASELINE_MODEL`) on that
  ``(seed, key)``. A model that cannot fit the data delivered chance performance; that is a
  real outcome, and imputing it stops a model from improving its standing by crashing on
  the targets it finds hard. Without a baseline result on that fold, the failure stays
  missing.
* ``skip``   — dropped from that model's target pool, and reported as reduced coverage
  (:func:`coverage_counts`). The design excluded the unit, so scoring it as a loss would
  rank the benchmark's own limits rather than the model.
* no record — the unit has not run yet and is missing. :func:`complete_folds` removes a model
  from a target until every scheduled, non-skipped fold is present, so every comparison on
  that target uses the same folds.
"""

from __future__ import annotations

import pandas as pd

from tabbench_bio.result_store import ResultRepository, result_location

#: Model whose result defines chance level on a target; failures are imputed to it.
BASELINE_MODEL = "DUMMY"

#: ``reason`` codes that mark a unit the benchmark excluded by design rather than a model
#: outcome. Kept as an explicit set so a new skip path must declare which side it is on.
DESIGN_SKIPS = frozenset(
    {
        "model_limit",
        "empty_split",
        "empty_split_after_filtering",
        "constant_target",
        "classification_only",
        "duplicate_cell",
        "benchmark_exclusion",
    }
)

_STATUS_COLUMNS = ["seed", "key", "model", "status", "reason"]


def load_status(results_dir: str) -> pd.DataFrame:
    """Load current unit states from the result database."""
    root, cell = result_location(results_dir)
    repository = ResultRepository.from_root(root, cell=cell)
    assert repository.bundle_paths(), f"No result database found under {root}"
    frame = repository.current_frame(cell=cell)
    if frame.empty:
        return pd.DataFrame(columns=_STATUS_COLUMNS)
    return frame.rename(columns={"dataset": "key"})[_STATUS_COLUMNS]


def coverage_counts(status: pd.DataFrame) -> pd.DataFrame:
    """Per-model unit counts: ``# Passed`` / ``# Failed`` / ``# Skipped``.

    Counted over ``(seed, key, model)`` units, so a model skipped on one dataset across all
    five folds shows five skips — the same unit granularity the pipeline schedules on.
    """
    cols = ["model_id", "# Passed", "# Failed", "# Skipped"]
    if status.empty:
        return pd.DataFrame(columns=cols)
    counts = (
        status.assign(n=1)
        .pivot_table(index="model", columns="status", values="n", aggfunc="sum", fill_value=0)
        .reset_index()
        .rename(columns={"model": "model_id"})
    )
    for src, dst in (("pass", "# Passed"), ("fail", "# Failed"), ("skip", "# Skipped")):
        counts[dst] = counts[src].astype(int) if src in counts.columns else 0
    return counts[cols]


def impute_failures(
    metrics_df: pd.DataFrame, status: pd.DataFrame, baseline_model: str = BASELINE_MODEL
) -> pd.DataFrame:
    """Append a chance-level row for every failed unit that produced no metrics.

    The appended row copies :data:`BASELINE_MODEL`'s metrics on the same ``(seed, key)`` and
    is flagged ``imputed``, so downstream aggregation scores the failure at chance instead
    of silently omitting it. Design skips (:data:`DESIGN_SKIPS`) are left absent, as is a
    failure whose ``(seed, key)`` has no baseline result. Returns *metrics_df* unchanged
    when there is nothing to impute.
    """
    if metrics_df.empty or status.empty:
        return metrics_df

    tasks = set(zip(metrics_df["seed"], metrics_df["key"]))
    have = set(zip(metrics_df["seed"], metrics_df["key"], metrics_df["model"]))
    baseline = metrics_df[metrics_df["model"] == baseline_model].set_index(["seed", "key"])

    failures = status[status["status"] == "fail"]
    rows = []
    for unit in failures.itertuples(index=False):
        task = (unit.seed, unit.key)
        if task not in tasks or task not in baseline.index:
            continue
        if (unit.seed, unit.key, unit.model) in have:
            continue
        row = baseline.loc[task].to_dict()
        row.update({"seed": unit.seed, "key": unit.key, "model": unit.model, "imputed": True})
        rows.append(row)

    if not rows:
        return metrics_df
    out = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
    out["imputed"] = out["imputed"].astype("boolean").fillna(False).astype(bool)
    return out


def complete_folds(
    metrics_df: pd.DataFrame, status: pd.DataFrame, seeds: list[int]
) -> pd.DataFrame:
    """Keep a ``(model, key)`` only when every scheduled, non-skipped fold has a row.

    *metrics_df* holds passes and imputed failures. A model with a fold still missing on a
    target is removed from that target, so every comparison on it uses the same folds.
    """
    if metrics_df.empty:
        return metrics_df
    scheduled = set(seeds)
    observed = set(metrics_df["seed"])
    assert observed <= scheduled, f"Unscheduled folds in metrics: {sorted(observed - scheduled)}"
    skips = status[status["status"] == "skip"]
    skipped = skips.groupby(["model", "key"])["seed"].agg(set)
    present = metrics_df.groupby(["model", "key"])["seed"].agg(set)
    keep = [
        pair
        for pair, folds in present.items()
        if scheduled - (skipped[pair] if pair in skipped.index else set()) <= folds
    ]
    return metrics_df[pd.MultiIndex.from_frame(metrics_df[["model", "key"]]).isin(keep)]
