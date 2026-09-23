"""Pairwise Elo ratings and head-to-head win counts for the leaderboard.

Elo is derived from fold-level pairwise comparisons: within every cross-validation fold of
every dataset target, each model is compared against every other model that also has a
result on that fold. Each battle is weighted by ``1 / n_folds`` of its target, so a target
contributes a total weight of one while a model that wins three of five folds is
distinguished from one that wins all five. The comparison score is "higher is better" for
both task types — the primary classification metric
(:data:`~tabbench_bio.metrics.PRIMARY_CLF_METRIC`, macro-F1) for classification targets,
**negated** RMSE for regression. Ratings are calibrated so a chosen anchor model (Random
Forest) sits at exactly 1000, and confidence intervals come from bootstrap resampling of
whole targets, carrying their folds with them.

Absent ``(model, target, fold)`` results are not imputed here: a fit that was attempted and
failed has already been imputed at chance level upstream
(:func:`tabbench_bio.coverage.impute_failures`), and a model without every scheduled fold on
a target has been removed from that target (:func:`tabbench_bio.coverage.complete_folds`).
What reaches this module as missing is therefore simply not played.

The rating itself is a **Bradley-Terry (BT) maximum-likelihood fit**, computed by the
vendored :mod:`tabbench_bio._vendor.tabarena_elo_utils` — TabArena's own ``EloHelper``,
which in turn follows Chatbot Arena's ``compute_mle_elo``. Using their code verbatim
rather than a reimplementation means our leaderboard is the same estimator as TabArena's,
which is the point of a benchmark named TabBench-Bio.

"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from tabbench_bio._vendor.tabarena_elo_utils import EloHelper
from tabbench_bio.metrics import PRIMARY_CLF_METRIC, PRIMARY_REG_METRIC

logger = logging.getLogger(__name__)

DEFAULT_BASE = 1000.0
DEFAULT_ANCHOR = "RF"
DEFAULT_SCALE = 400.0

#: Bootstrap rounds for the Elo confidence intervals. Matches TabArena's default.
DEFAULT_N_BOOT = 100


def fold_scores(
    clf_df: pd.DataFrame | None,
    reg_df: pd.DataFrame | None,
    clf_metric: str = PRIMARY_CLF_METRIC,
    reg_metric: str = PRIMARY_REG_METRIC,
) -> pd.DataFrame:
    """Fold-level comparison scores: columns ``model``, ``key``, ``seed``, ``score``.

    Classification rows use ``clf_metric`` (higher is better); regression rows use the
    **negated** ``reg_metric`` (so higher is better there too). The two task pools share one
    target space (dataset ``key``s). Pass ``None`` / empty for one task to build a
    task-specific pool. Folds whose metric is undefined are dropped.
    """
    frames = []
    if clf_df is not None and not clf_df.empty and clf_metric in clf_df.columns:
        frames.append(clf_df[["model", "key", "seed"]].assign(score=clf_df[clf_metric].to_numpy()))
    if reg_df is not None and not reg_df.empty and reg_metric in reg_df.columns:
        frames.append(reg_df[["model", "key", "seed"]].assign(score=-reg_df[reg_metric].to_numpy()))
    if not frames:
        return pd.DataFrame(columns=["model", "key", "seed", "score"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["score"])


def score_table(
    clf_df: pd.DataFrame | None,
    reg_df: pd.DataFrame | None,
    clf_metric: str = PRIMARY_CLF_METRIC,
    reg_metric: str = PRIMARY_REG_METRIC,
) -> pd.DataFrame:
    """Models x targets table of fold-mean scores (higher = better; NaN = absent).

    Descriptive only (win counts, paired effects); ratings use :func:`fold_scores`.
    """
    scores = fold_scores(clf_df, reg_df, clf_metric, reg_metric)
    if scores.empty:
        return pd.DataFrame()
    return scores.groupby(["model", "key"])["score"].mean().unstack("key")


def _fold_battles(scores: pd.DataFrame) -> tuple[pd.DataFrame, EloHelper]:
    """Fold scores → TabArena ``battles`` frame, via its own ``convert_results_to_battles``.

    Battles pair models within each ``(target, fold)``; TabArena's helper weights each by
    ``1 / n_folds`` of its target. The helper is written against a *metric error* (lower is
    better), so the score is negated back into an error here.
    """
    helper = EloHelper(
        method_col="method", task_col="task", error_col="metric_error", split_col="split"
    )
    frame = pd.DataFrame(
        {
            "method": scores["model"].to_numpy(),
            "task": scores["key"].to_numpy(),
            "metric_error": -scores["score"].astype(float).to_numpy(),
            "split": scores["seed"].to_numpy(),
        }
    )
    return helper.convert_results_to_battles(frame), helper


def compute_elo(
    scores: pd.DataFrame,
    anchor: str = DEFAULT_ANCHOR,
    anchor_value: float = DEFAULT_BASE,
    n_boot: int = DEFAULT_N_BOOT,
    scale: float = DEFAULT_SCALE,
    random_state: int = 0,
    show_process: bool = False,
) -> pd.DataFrame:
    """Bradley-Terry Elo per model with bootstrap 95% CIs (TabArena's estimator).

    Parameters
    ----------
    scores : pd.DataFrame
        Fold-level scores from :func:`fold_scores` (higher = better).
    anchor, anchor_value : str, float
        Model whose rating is fixed (Random Forest → 1000). Without the anchor no rating is
        published: an empty table is returned, because an uncalibrated origin would not be
        comparable with any other ranking.
    n_boot : int
        Bootstrap resamples of the *target* pool for the confidence interval. As in
        TabArena, the reported rating is the bootstrap **median**, not the point fit.
    scale : float
        Elo scale; 400 gives the conventional "400 points = 10:1 odds" reading.

    Returns
    -------
    pd.DataFrame
        Columns ``model_id``, ``Elo``, ``Elo_lo``, ``Elo_hi``, ``n_targets`` (rounded ints).
    """
    cols = ["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"]
    if scores.empty or scores["model"].nunique() < 2:
        return pd.DataFrame(columns=cols)
    if anchor not in set(scores["model"]):
        logger.warning("Elo anchor %r absent; no ratings for this pool.", anchor)
        return pd.DataFrame(columns=cols)

    battles, helper = _fold_battles(scores)
    if battles.empty:
        return pd.DataFrame(columns=cols)

    draws = helper.compute_elo_ratings(
        battles=battles,
        seed=random_state,
        calibration_framework=anchor,
        calibration_elo=anchor_value,
        INIT_RATING=anchor_value,
        BOOTSTRAP_ROUNDS=n_boot,
        SCALE=scale,
        show_process=show_process,
    )

    counts = scores.groupby("model")["key"].nunique()
    models = [m for m in draws.columns]
    out = pd.DataFrame(
        {
            "model_id": models,
            "Elo": np.round(draws.quantile(0.5)[models].to_numpy()).astype(int),
            "Elo_lo": np.round(draws.quantile(0.025)[models].to_numpy()).astype(int),
            "Elo_hi": np.round(draws.quantile(0.975)[models].to_numpy()).astype(int),
            "n_targets": counts.reindex(models).fillna(0).to_numpy().astype(int),
        }
    )
    return out.sort_values("Elo", ascending=False).reset_index(drop=True)


def paired_elo_difference(
    scores: pd.DataFrame,
    model_a: str,
    model_b: str,
    anchor: str = DEFAULT_ANCHOR,
    anchor_value: float = DEFAULT_BASE,
    n_boot: int = DEFAULT_N_BOOT,
    scale: float = DEFAULT_SCALE,
    random_state: int = 0,
) -> dict[str, float]:
    """Bootstrap interval for the Elo *difference* between two models.

    Bradley-Terry identifies only differences of latent abilities, so the per-model
    intervals from :func:`compute_elo` are intervals for ``model - anchor``: the anchor
    shift is applied inside every bootstrap draw, which is why the anchor's own interval
    has zero width. Two such intervals share the ``-theta_anchor`` term and are therefore
    positively correlated, so their overlap is not a test of ``model_a`` against
    ``model_b``. This resamples the same target pool and reads off the difference within
    each draw, where the anchor cancels.

    Returns ``diff`` (bootstrap median), ``diff_lo`` / ``diff_hi`` (percentile interval)
    and ``prob_a_gt_b`` (share of draws in which *model_a* rates above *model_b*).
    """
    models = set(scores["model"])
    assert model_a in models, f"{model_a} absent from the scores"
    assert model_b in models, f"{model_b} absent from the scores"

    battles, helper = _fold_battles(scores)
    assert not battles.empty, "No battles to compare"

    calibration = anchor if anchor in models else None
    draws = helper.compute_elo_ratings(
        battles=battles,
        seed=random_state,
        calibration_framework=calibration,
        calibration_elo=anchor_value if calibration else None,
        INIT_RATING=anchor_value,
        BOOTSTRAP_ROUNDS=n_boot,
        SCALE=scale,
        show_process=False,
    )
    delta = draws[model_a] - draws[model_b]
    return {
        "diff": float(delta.quantile(0.5)),
        "diff_lo": float(delta.quantile(0.025)),
        "diff_hi": float(delta.quantile(0.975)),
        "prob_a_gt_b": float((delta > 0).mean()),
    }


def win_counts(table: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Head-to-head win counts: ``M[i, j]`` = targets where model *i* beats *j* (ties 0.5).

    Only targets on which both models have a result are counted.
    """
    models = list(table.index)
    arr = table.to_numpy()
    n = len(models)
    m = np.zeros((n, n), dtype=float)
    for t in range(arr.shape[1]):
        col = arr[:, t]
        present = [i for i in range(n) if not np.isnan(col[i])]
        for a, b in combinations(present, 2):
            va, vb = col[a], col[b]
            if va > vb:
                m[a, b] += 1.0
            elif va < vb:
                m[b, a] += 1.0
            else:
                m[a, b] += 0.5
                m[b, a] += 0.5
    return models, m
