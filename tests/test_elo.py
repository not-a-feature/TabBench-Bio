"""Tests for the Bradley-Terry Elo leaderboard.

The properties pinned here are the ones a silent regression would quietly corrupt: that a
strict dominance order comes back in the right order, that the anchor lands exactly on its
target, that unplayed (model, target) pairings are never imputed into a rating, that fold
wins rather than fold means decide a target, and that a rating difference means what the
paper says it means (a 400-point gap is 10:1 odds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.elo import (
    _fold_battles,
    compute_elo,
    fold_scores,
    score_table,
    win_counts,
)
from tabbench_bio.metrics import PRIMARY_CLF_METRIC

# Keep the bootstrap small: these tests assert on the point rating and its ordering, not
# on interval width, and TabArena's default of 100 rounds makes the suite crawl.
N_BOOT = 20


def _clf(rows: list[tuple]) -> pd.DataFrame:
    """(model, key, primary-metric[, seed]) tuples -> a fold-level metrics frame.

    Named from PRIMARY_CLF_METRIC rather than hard-coded, so re-ranking the benchmark on a
    different recorded metric does not silently turn these tests into no-ops.
    """
    rows = [row if len(row) == 4 else (*row, 0) for row in rows]
    return pd.DataFrame(rows, columns=["model", "key", PRIMARY_CLF_METRIC, "seed"])


def _dominance_rows(n_models: int = 4, n_targets: int = 6) -> list[tuple]:
    """Model i beats model j on every target iff i < j, by construction."""
    return [(f"M{i}", f"task{t}", 1.0 - 0.1 * i) for t in range(n_targets) for i in range(n_models)]


def _dominance_scores(n_models: int = 4, n_targets: int = 6) -> pd.DataFrame:
    return fold_scores(_clf(_dominance_rows(n_models, n_targets)), None)


class TestScoreTable:
    def test_regression_metric_is_negated_so_higher_is_better(self):
        reg = pd.DataFrame(
            [("A", "t1", 1.0, 0), ("B", "t1", 5.0, 0)], columns=["model", "key", "rmse", "seed"]
        )
        table = score_table(None, reg)
        # A has the lower RMSE, so after negation it must hold the larger score.
        assert table.loc["A", "t1"] > table.loc["B", "t1"]

    def test_absent_pairings_stay_nan(self):
        table = score_table(_clf([("A", "t1", 0.9), ("B", "t2", 0.8)]), None)
        assert np.isnan(table.loc["A", "t2"])
        assert np.isnan(table.loc["B", "t1"])

    def test_empty_input_returns_empty(self):
        assert score_table(None, None).empty


class TestFoldBattles:
    def test_each_target_contributes_total_weight_one(self):
        rows = [("A", "t5", 0.9, s) for s in range(5)] + [("B", "t5", 0.1, s) for s in range(5)]
        rows += [("A", "t1", 0.9, 0), ("B", "t1", 0.1, 0)]
        battles, _ = _fold_battles(fold_scores(_clf(rows), None))
        assert battles.groupby("task")["weight"].sum().to_dict() == pytest.approx(
            {"t1": 1.0, "t5": 1.0}
        )

    def test_fold_wins_not_fold_means_decide_a_target(self):
        """B has the higher fold mean on every target but wins only three of five folds."""
        rows = []
        for t in range(20):
            for s in range(5):
                rows += [("A", f"t{t}", 0.7, s), ("B", f"t{t}", 0.95 if s < 3 else 0.5, s)]
        means = score_table(_clf(rows), None)
        assert (means.loc["B"] > means.loc["A"]).all()
        elo = compute_elo(fold_scores(_clf(rows), None), anchor="A", n_boot=N_BOOT)
        gap = elo.loc[elo.model_id == "B", "Elo"].iloc[0] - 1000
        assert gap == pytest.approx(400 * np.log10(3 / 2), abs=15)


class TestComputeElo:
    def test_strict_dominance_recovers_the_true_order(self):
        elo = compute_elo(_dominance_scores(), anchor="M0", n_boot=N_BOOT)
        assert list(elo["model_id"]) == ["M0", "M1", "M2", "M3"]

    def test_anchor_sits_exactly_on_its_target(self):
        elo = compute_elo(_dominance_scores(), anchor="M2", anchor_value=1000, n_boot=N_BOOT)
        assert elo.loc[elo.model_id == "M2", "Elo"].iloc[0] == 1000

        shifted = compute_elo(_dominance_scores(), anchor="M2", anchor_value=1500, n_boot=N_BOOT)
        assert shifted.loc[shifted.model_id == "M2", "Elo"].iloc[0] == 1500

    def test_anchor_shift_preserves_rating_differences(self):
        a = compute_elo(_dominance_scores(), anchor="M0", anchor_value=1000, n_boot=N_BOOT)
        b = compute_elo(_dominance_scores(), anchor="M0", anchor_value=1700, n_boot=N_BOOT)
        merged = a.merge(b, on="model_id", suffixes=("_a", "_b"))
        # Calibration is affine, so every rating must move by the same constant.
        deltas = (merged["Elo_b"] - merged["Elo_a"]).unique()
        assert len(deltas) == 1

    def test_missing_anchor_yields_no_ratings(self):
        elo = compute_elo(_dominance_scores(), anchor="NOPE", n_boot=N_BOOT)
        assert elo.empty
        assert list(elo.columns) == ["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"]

    def test_equal_models_get_equal_ratings(self):
        rows = [(m, f"task{t}", 0.5) for m in ("A", "B", "C") for t in range(4)]
        elo = compute_elo(fold_scores(_clf(rows), None), anchor="A", n_boot=N_BOOT)
        assert elo["Elo"].nunique() == 1

    def test_n_targets_counts_only_played_pairings(self):
        rows = [("A", f"t{i}", 0.9) for i in range(5)]
        rows += [("B", f"t{i}", 0.8) for i in range(5)]
        rows += [("C", "t0", 0.7)]  # C ran on a single target
        elo = compute_elo(fold_scores(_clf(rows), None), anchor="A", n_boot=N_BOOT)
        n = dict(zip(elo["model_id"], elo["n_targets"]))
        assert n == {"A": 5, "B": 5, "C": 1}

    def test_rating_difference_is_a_calibrated_log_odds(self):
        """A 400-point gap must imply a 10:1 (90.9%) expected win rate.

        This is the claim the paper makes about the scale; it holds for the BT fit and is
        what the online variant cannot support at an arbitrary K.

        Uses 100 targets rather than 10 for a reason: TabArena's bootstrap falls back to
        an iterative Elo (K=1, a wholly different scale) for any draw in which no pair has
        a win on one side, and with a 9:1 record over 10 targets ~35% of draws miss the
        lone loss and trip that fallback, dragging the median far off the MLE. At 90:10
        the probability is 0.9**100 ~ 3e-5. See test_bootstrap_fallback_is_documented.
        """
        # 90 wins to 10 losses = 9:1 odds => a gap of 400*log10(9) points.
        rows = []
        for t in range(100):
            a, b = (0.9, 0.1) if t < 90 else (0.1, 0.9)
            rows += [("A", f"t{t}", a), ("B", f"t{t}", b)]
        elo = compute_elo(
            fold_scores(_clf(rows), None), anchor="B", anchor_value=1000, n_boot=N_BOOT
        )
        gap = elo.loc[elo.model_id == "A", "Elo"].iloc[0] - 1000
        expected = 400 * np.log10(9)  # ~382
        assert gap == pytest.approx(expected, abs=25)

    def test_bootstrap_fallback_is_documented(self):
        """Pin the upstream quirk that the calibration test has to route around.

        When a bootstrap draw leaves one side of every pair without a win, TabArena's
        helper abandons the MLE for that draw and returns an iterative Elo at K=1. The
        reported median then mixes two scales. This is upstream behaviour we inherit
        deliberately (we run their estimator verbatim); it only bites when the target pool
        is tiny and one model is near-perfectly dominant, so it is asserted here rather
        than patched, and it is why leaderboard cells over few targets need reading with
        their intervals rather than their point rating.
        """
        rows = []
        for t in range(10):
            a, b = (0.9, 0.1) if t < 9 else (0.1, 0.9)
            rows += [("A", f"t{t}", a), ("B", f"t{t}", b)]
        scores = fold_scores(_clf(rows), None)

        battles, helper = _fold_battles(scores)
        point = helper.compute_mle_elo(battles, calibration_framework="B", calibration_elo=1000)
        # The point fit itself is exact...
        assert (point["A"] - point["B"]) == pytest.approx(400 * np.log10(9), abs=1)

        # ...but the bootstrap median is pulled below it by the fallback draws.
        elo = compute_elo(scores, anchor="B", anchor_value=1000, n_boot=100)
        gap = elo.loc[elo.model_id == "A", "Elo"].iloc[0] - 1000
        assert gap < 400 * np.log10(9)

    def test_empty_input_returns_empty_frame(self):
        out = compute_elo(fold_scores(None, None), n_boot=N_BOOT)
        assert out.empty
        assert list(out.columns) == ["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"]

    def test_single_model_returns_empty(self):
        out = compute_elo(fold_scores(_clf([("A", "t1", 0.9)]), None), anchor="A", n_boot=N_BOOT)
        assert out.empty

    def test_confidence_interval_brackets_the_rating(self):
        elo = compute_elo(_dominance_scores(n_targets=8), anchor="M1", n_boot=N_BOOT)
        assert (elo["Elo_lo"] <= elo["Elo"]).all()
        assert (elo["Elo"] <= elo["Elo_hi"]).all()


class TestWinCounts:
    def test_counts_wins_and_splits_ties(self):
        rows = [("A", "t1", 0.9), ("B", "t1", 0.1), ("A", "t2", 0.5), ("B", "t2", 0.5)]
        models, m = win_counts(score_table(_clf(rows), None))
        i, j = models.index("A"), models.index("B")
        assert m[i][j] == 1.5  # one outright win + half a tie
        assert m[j][i] == 0.5  # half a tie

    def test_ignores_targets_only_one_model_ran(self):
        rows = [("A", "t1", 0.9), ("B", "t1", 0.1), ("A", "t2", 0.9)]
        models, m = win_counts(score_table(_clf(rows), None))
        i, j = models.index("A"), models.index("B")
        assert m[i][j] == 1.0  # t2 contributes nothing, having no opponent
