"""Regression tests for leakage prevention and score-definition consistency."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabbench_bio.benchmark import TabBenchBio
from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.loaders.local import LocalLoader
from tabbench_bio.config import model_limits, model_overrides
from tabbench_bio.coverage import (
    DESIGN_SKIPS,
    assert_complete,
    coverage_counts,
    impute_failures,
    load_status,
)
from tabbench_bio.dataset import Dataset, DatasetInfo, TaskType
from tabbench_bio.leaderboard import Leaderboard
from tabbench_bio.metrics import PRIMARY_CLF_METRIC
from tabbench_bio.site import _normalized_scores


def test_local_group_column_is_preserved_but_not_a_feature(tmp_path):
    path = tmp_path / "grouped.csv"
    pd.DataFrame(
        {
            "embedding": ["1,2", "3,4", "5,6"],
            "group_id": ["locus-a", "locus-a", "locus-b"],
            "y": ["FUNC", "LOF", "FUNC"],
        }
    ).to_csv(path, index=False)
    spec = BioDatasetSpec(
        bio_id="local-test",
        source="local",
        fetch_id=str(path),
        target="y",
        problem_type="binary",
        embedding_column="embedding",
        group_column="group_id",
    )

    raw = LocalLoader(embedding_column=spec.embedding_column, group_column=spec.group_column).fetch(
        spec
    )

    assert list(raw.X.columns) == ["embedding_0", "embedding_1"]
    assert raw.groups.tolist() == ["locus-a", "locus-a", "locus-b"]


def test_grouped_five_fold_has_no_independence_unit_leakage(tmp_path):
    # Five pure groups per class, two replicate rows per group.
    labels = np.repeat(["A", "B", "C"], 10)
    groups = np.repeat([f"{label}-{i}" for label in "ABC" for i in range(5)], 2)
    dataset = Dataset(
        features=np.arange(60, dtype=np.float32).reshape(30, 2),
        targets=labels,
        feature_names=["f0", "f1"],
        target_names=["label"],
        info=DatasetInfo("grouped", "grouped", TaskType.Classification),
        groups=groups,
    )
    data = dataset.to_dataframe()
    held_out: list[int] = []

    for fold in range(5):
        bench = TabBenchBio(
            dataset_names_classification=["grouped"],
            dataset_names_regression=[],
            random_state=fold,
            cache_dir=str(tmp_path / f"fold-{fold}"),
            cv_folds=5,
        )
        train, test = bench._split(data, "grouped", dataset, num_targets=1)
        assert set(groups[train.index]).isdisjoint(groups[test.index])
        held_out.extend(test.index.tolist())

    assert sorted(held_out) == list(range(len(data)))


def test_site_normalization_matches_leaderboard_minmax_definition():
    scores = pd.DataFrame(
        {
            "dataset_0": [0.2, 0.5, 0.8, np.nan],
            "tied_0": [1.0, 1.0, 1.0, np.nan],
        },
        index=["worst", "middle", "best", "not_run"],
    )

    normalized = _normalized_scores(scores)

    assert normalized["dataset_0"].iloc[:3].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert normalized["tied_0"].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert normalized.loc["not_run"].isna().all()


# ---------------------------------------------------------------------------
# Absent (model, target) pairings: design exclusion vs. attempted-and-failed
# ---------------------------------------------------------------------------


def _results_dir(tmp_path, records, metric_rows):
    """A minimal results dir: one stats record per unit plus the metrics CSV."""
    stats_dir = tmp_path / "seed_0" / "stats"
    stats_dir.mkdir(parents=True)
    for key, model, status, reason in records:
        (stats_dir / f"{key}_{model}.json").write_text(
            json.dumps({"dataset": key, "model": model, "status": status, "reason": reason})
        )
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(metric_rows, columns=["seed", "key", "model", PRIMARY_CLF_METRIC]).to_csv(
        metrics_dir / "classification_metrics.csv", index=False
    )
    return str(tmp_path)


#: Two targets. LIMITED is excluded by design on the second, BROKEN failed on it.
_RECORDS = [
    ("ds1_0", "DUMMY", "pass", ""),
    ("ds1_0", "STRONG", "pass", ""),
    ("ds1_0", "LIMITED", "pass", ""),
    ("ds1_0", "BROKEN", "pass", ""),
    ("ds2_0", "DUMMY", "pass", ""),
    ("ds2_0", "STRONG", "pass", ""),
    ("ds2_0", "LIMITED", "skip", "model_limit"),
    ("ds2_0", "BROKEN", "fail", "fit_error"),
]
_METRICS = [
    (0, "ds1_0", "DUMMY", 0.20),
    (0, "ds1_0", "STRONG", 0.90),
    (0, "ds1_0", "LIMITED", 0.80),
    (0, "ds1_0", "BROKEN", 0.85),
    (0, "ds2_0", "DUMMY", 0.20),
    (0, "ds2_0", "STRONG", 0.70),
]


def test_design_skip_is_excluded_from_the_score_not_zeroed(tmp_path):
    # A model kept off an input by the benchmark's own size limit must not be scored as
    # though it had lost there: that ranks the harness, not the model.
    lb = Leaderboard.from_results_dir(_results_dir(tmp_path, _RECORDS, _METRICS)).rank(
        "classification"
    )
    row = lb.set_index("model_id").loc["LIMITED"]
    assert row["# Targets"] == 1
    assert row["Score"] == pytest.approx((0.80 - 0.20) / (0.90 - 0.20), abs=1e-4)


def test_failed_fit_is_scored_at_the_chance_baseline(tmp_path):
    # A fit that was attempted and crashed is a real outcome. Imputing it at the chance
    # baseline stops a model from improving its standing by failing where it struggles.
    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    lb = Leaderboard.from_results_dir(results_dir).rank("classification")
    row = lb.set_index("model_id").loc["BROKEN"]
    assert row["# Targets"] == 2
    # ds1: (0.85-0.20)/(0.90-0.20); ds2: imputed to DUMMY, i.e. the per-target minimum -> 0.
    assert row["Score"] == pytest.approx(((0.85 - 0.20) / (0.90 - 0.20)) / 2, abs=1e-4)

    counts = coverage_counts(load_status(results_dir)).set_index("model_id")
    assert counts.loc["BROKEN", "# Failed"] == 1
    assert counts.loc["BROKEN", "# Skipped"] == 0
    assert counts.loc["LIMITED", "# Skipped"] == 1
    assert counts.loc["LIMITED", "# Failed"] == 0


def test_failure_imputation_makes_elo_and_score_agree_on_direction(tmp_path):
    # Both aggregates must see the failure. Omitting it would let BROKEN out-rank STRONG on
    # Elo (one unbeaten target) while scoring below it on the mean.
    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    status = load_status(results_dir)
    clf = impute_failures(pd.read_csv(f"{results_dir}/metrics/classification_metrics.csv"), status)
    played = clf[clf["model"] == "BROKEN"]
    assert set(played["key"]) == {"ds1_0", "ds2_0"}
    assert played.set_index("key").loc["ds2_0", PRIMARY_CLF_METRIC] == pytest.approx(0.20)
    # The design exclusion is still absent, so that pairing is simply not played.
    assert set(clf[clf["model"] == "LIMITED"]["key"]) == {"ds1_0"}


def test_unrecorded_unit_blocks_publication(tmp_path):
    # A dataset that silently produced nothing would otherwise renormalise every aggregate
    # over the survivors and report a dataset count nobody chose.
    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    status = load_status(results_dir)
    clf = pd.read_csv(f"{results_dir}/metrics/classification_metrics.csv")
    models = ["DUMMY", "STRONG", "LIMITED", "BROKEN"]

    assert_complete(clf, status, keys=["ds1_0", "ds2_0"], models=models, seeds=[0])
    with pytest.raises(AssertionError, match="incomplete"):
        assert_complete(clf, status, keys=["ds1_0", "ds2_0", "ds3_0"], models=models, seeds=[0])


def test_duplicate_grid_cell_is_a_declared_design_skip():
    # The grid's sample axis runs past the size of the smaller datasets; those cells repeat
    # the full-sample fit and must be recognisable as exclusions rather than results.
    assert "duplicate_cell" in DESIGN_SKIPS


def test_only_autogluon_opts_out_of_the_run_wide_fitting_regime():
    # The leaderboard compares models under one budget; an entry that overrides the regime
    # is a reference line, so the roster must not quietly grow more of them.
    with open("configs/models/all.json") as f:
        roster = json.load(f)
    overrides = model_overrides(roster)
    assert set(overrides) == {"AUTOGLUON"}
    assert overrides["AUTOGLUON"]["presets"] == "extreme"


def test_no_model_enforces_a_legacy_memory_prior():
    with open("configs/models/all.json") as f:
        roster = json.load(f)
    limits = model_limits(roster)
    assert not {model for model, prior in limits.items() if prior["enforce_memory_prior"]}


def test_a_wholly_absent_dataset_is_caught_from_the_config(tmp_path):
    # The failure this gate exists for: a configured dataset that produced neither a metric
    # row nor a status record is absent from everything on disk, so the expected key set has
    # to come from the config rather than from what happened to land.
    from tabbench_bio.site import _expected_keys

    keys = _expected_keys(
        {
            "dataset_names_classification": ["ds1", "ds2", "vanished"],
            "dataset_names_regression": [],
        }
    )
    assert keys == ["ds1_0", "ds2_0", "vanished_0"]

    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    clf = pd.read_csv(f"{results_dir}/metrics/classification_metrics.csv")
    with pytest.raises(AssertionError, match="vanished_0"):
        assert_complete(
            clf,
            load_status(results_dir),
            keys=keys,
            models=["DUMMY", "STRONG", "LIMITED", "BROKEN"],
            seeds=[0],
        )
