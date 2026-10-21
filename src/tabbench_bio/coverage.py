"""Per-unit run status: what passed, what the design excluded, and what failed.

The predictions step commits one record per ``(key, model)`` attempt. That record is the
only place the pipeline distinguishes a unit the
benchmark excluded **by design** — a declared per-model input-size limit, a degenerate
split, a grid cell whose sample budget exceeds the data — from a unit the model was asked
to fit and **failed** on. Neither appears in the metrics CSVs, so any ranking that reads
only those CSVs cannot tell the two apart and ends up measuring the harness.

One convention, applied by every consumer (leaderboard ``Score``, the per-dataset score
heatmap, and Elo alike):

* ``fail``   — a tuned variant uses its declared untuned parent's metrics on the same
  ``(seed, key)``; other models use the run's chance-level baseline
  (:data:`BASELINE_MODEL`). Without the required source result on that fold, the failure
  stays missing. The original attempt remains a failure in coverage counts.
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

#: Model whose result defines chance level for failures of untuned models.
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
        "class_limit",
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
    metrics_df: pd.DataFrame,
    status: pd.DataFrame,
    baseline_model: str = BASELINE_MODEL,
    *,
    model_tuning: dict | None = None,
) -> pd.DataFrame:
    """Score failed tuning at its untuned parent and other failures at chance.

    ``model_tuning`` comes from the cell's frozen configuration. Each appended row copies
    the source model's metrics on the same ``(seed, key)`` and is flagged ``imputed``.
    Design skips and failures without their required source result stay absent. Only
    existing metrics supply imputations; a parent's failure cannot cascade to its child.
    """
    if metrics_df.empty or status.empty:
        return metrics_df

    specifications = model_tuning if model_tuning is not None else {}
    have = set(zip(metrics_df["seed"], metrics_df["key"], metrics_df["model"]))
    sources = (
        metrics_df[~metrics_df["imputed"].eq(True).fillna(False)]
        if "imputed" in metrics_df
        else metrics_df
    ).set_index(["seed", "key", "model"])

    failures = status[status["status"] == "fail"]
    rows = []
    for unit in failures.itertuples(index=False):
        if (unit.seed, unit.key, unit.model) in have:
            continue
        source_model = (
            specifications[unit.model]["base_model"]
            if unit.model in specifications
            else baseline_model
        )
        source = (unit.seed, unit.key, source_model)
        if source not in sources.index:
            continue
        row = sources.loc[source].to_dict()
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
