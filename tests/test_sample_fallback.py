import json
from pathlib import Path

import pandas as pd
import pytest

from tabbench_bio.sample_fallback import resolve_sample_fallbacks, sample_fallback_chain

KEY = "toy__target"
MODEL = "TABPFN-WIDE"


def _make_cell(root: Path, name: str, train_subsample: int | None) -> Path:
    cell = root / name
    (cell / "seed_0" / "stats").mkdir(parents=True)
    (cell / "seed_0" / "logs").mkdir(parents=True)
    (cell / "seed_0" / "predictions").mkdir(parents=True)
    (cell / "metrics").mkdir()
    config = {
        "models": [MODEL],
        "output_dir": str(cell),
        "train_subsample": train_subsample,
        "bio_max_features": 2000,
        "cv_folds": 5,
        "random_state": 42,
    }
    (cell / "config.json").write_text(json.dumps(config), encoding="utf-8")
    truth = pd.DataFrame({"target": [0, 1]}, index=[10, 11])
    truth.to_csv(cell / "seed_0" / "predictions" / f"{KEY}_ground_truth.csv")
    return cell


def _write_status(
    cell: Path,
    *,
    status: str,
    reason: str,
    n_train: int,
    error: str = "",
    **details,
) -> None:
    record = {
        "dataset": KEY,
        "model": MODEL,
        "status": status,
        "reason": reason,
        "error": error,
        "n_train_samples": n_train,
        **details,
    }
    path = cell / "seed_0" / "stats" / f"{KEY}_{MODEL}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def _write_metric(cell: Path, score: float) -> Path:
    path = cell / "metrics" / "classification_metrics.csv"
    pd.DataFrame(
        [
            {
                "seed": 0,
                "key": KEY,
                "dataset": "toy",
                "task_type": "binary",
                "target_idx": 0,
                "model": MODEL,
                "f1_macro": score,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_sample_fallback_chain_uses_configured_cells_nearest_each_half():
    budgets = [20, 50, 100, 200, 500, 1000, 5000]
    assert sample_fallback_chain(985, budgets) == [500, 200, 100, 50, 20]
    assert sample_fallback_chain(8000, budgets) == [5000, 1000, 500, 200, 100, 50, 20]
    assert sample_fallback_chain(20, budgets) == []


def test_sample_fallback_chain_jumps_to_calibrated_start_then_keeps_halving():
    budgets = [20, 50, 100, 200, 500, 1000]
    assert sample_fallback_chain(2156, budgets, max_start_n=425) == [200, 100, 50, 20]


def test_resolver_reuses_full_cell_for_duplicate_sample_cell(tmp_path):
    full = _make_cell(tmp_path, "cap_2000", None)
    duplicate = _make_cell(tmp_path, "cap_2000_n100", 100)
    _make_cell(tmp_path, "cap_2000_n20", 20)
    for cell, datasets in ((full, ["toy"]), (duplicate, ["toy", "new-dataset"])):
        config_path = cell / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["datasets_classification"] = datasets
        config["datasets_regression"] = []
        config_path.write_text(json.dumps(config), encoding="utf-8")
    _write_status(full, status="pass", reason="", n_train=70)
    _write_status(duplicate, status="skip", reason="duplicate_cell", n_train=70)
    (duplicate / "seed_0" / "predictions" / f"{KEY}_ground_truth.csv").unlink()
    metric_path = _write_metric(full, 0.81)
    original = metric_path.read_bytes()

    strict, adaptive, manifest = resolve_sample_fallbacks(
        tmp_path, ["cap_2000", "cap_2000_n100", "cap_2000_n20"]
    )

    assert len(strict["classification"]) == 1
    row = adaptive["classification"].query("cell == 'cap_2000_n100'").iloc[0]
    assert row["f1_macro"] == pytest.approx(0.81)
    assert bool(row["fallback"])
    assert row["nominal_n_train"] == row["effective_n_train"] == 70
    assert row["fallback_reason"] == "duplicate_cell"
    assert row["reused_from_cell"] == "cap_2000"
    assert manifest.iloc[0]["status"] == "resolved"
    assert metric_path.read_bytes() == original


def test_resolver_reuses_passing_lower_cell_without_modifying_source(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    lower = _make_cell(tmp_path, "cap_2000_n500", 500)
    # Historical full cells may contain an override for a full-cell-only model. It is
    # irrelevant to this target model and must not invalidate otherwise identical cells.
    target_config_path = target / "config.json"
    target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    target_config["model_overrides"] = {"AUTOGLUON": {"presets": "extreme"}}
    target_config_path.write_text(json.dumps(target_config), encoding="utf-8")
    lower_config_path = lower / "config.json"
    lower_config = json.loads(lower_config_path.read_text(encoding="utf-8"))
    lower_config["model_overrides"] = {}
    lower_config_path.write_text(json.dumps(lower_config), encoding="utf-8")
    _write_status(
        target,
        status="fail",
        reason="fit_error",
        n_train=985,
        error="CUDA out of memory",
    )
    _write_status(lower, status="pass", reason="", n_train=500)
    metric_path = _write_metric(lower, 0.73)
    original = metric_path.read_bytes()

    strict, adaptive, manifest = resolve_sample_fallbacks(
        tmp_path, ["cap_2000", "cap_2000_n500"]
    )

    assert len(strict["classification"]) == 1
    target_row = adaptive["classification"].query("cell == 'cap_2000'").iloc[0]
    assert target_row["f1_macro"] == pytest.approx(0.73)
    assert bool(target_row["fallback"])
    assert target_row["nominal_n_train"] == 985
    assert target_row["effective_n_train"] == 500
    assert target_row["reused_from_cell"] == "cap_2000_n500"
    assert json.loads(target_row["fallback_path"]) == [985, 500]
    assert manifest.iloc[0]["status"] == "resolved"
    assert metric_path.read_bytes() == original


def test_resolver_continues_past_lower_memory_failures(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    lower_500 = _make_cell(tmp_path, "cap_2000_n500", 500)
    lower_200 = _make_cell(tmp_path, "cap_2000_n200", 200)
    _write_status(target, status="fail", reason="fit_oom", n_train=985)
    _write_status(lower_500, status="fail", reason="fit_oom", n_train=500)
    _write_status(lower_200, status="pass", reason="", n_train=200)
    _write_metric(lower_200, 0.61)

    _, adaptive, manifest = resolve_sample_fallbacks(
        tmp_path, ["cap_2000", "cap_2000_n500", "cap_2000_n200"]
    )

    target_row = adaptive["classification"].query("cell == 'cap_2000'").iloc[0]
    assert target_row["effective_n_train"] == 200
    assert json.loads(target_row["fallback_path"]) == [985, 500, 200]
    assert manifest.query("cell == 'cap_2000'").iloc[0]["status"] == "resolved"


def test_resolver_uses_recorded_memory_prior_start(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    too_large = _make_cell(tmp_path, "cap_2000_n500", 500)
    start = _make_cell(tmp_path, "cap_2000_n200", 200)
    _write_status(
        target,
        status="skip",
        reason="model_limit",
        n_train=985,
        recommended_max_n_train=425,
    )
    _write_status(too_large, status="fail", reason="fit_oom", n_train=500)
    _write_status(start, status="pass", reason="", n_train=200)
    _write_metric(start, 0.62)

    _, adaptive, manifest = resolve_sample_fallbacks(
        tmp_path, ["cap_2000", "cap_2000_n500", "cap_2000_n200"]
    )

    target_row = adaptive["classification"].query("cell == 'cap_2000'").iloc[0]
    assert target_row["effective_n_train"] == 200
    assert json.loads(target_row["fallback_path"]) == [985, 200]
    assert json.loads(manifest.query("cell == 'cap_2000'").iloc[0]["fallback_path"]) == [
        985,
        200,
    ]


def test_resolver_does_not_hide_non_memory_failures(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    lower = _make_cell(tmp_path, "cap_2000_n500", 500)
    _write_status(target, status="fail", reason="time_limit", n_train=985)
    _write_status(lower, status="pass", reason="", n_train=500)
    _write_metric(lower, 0.73)

    _, adaptive, manifest = resolve_sample_fallbacks(
        tmp_path, ["cap_2000", "cap_2000_n500"]
    )

    assert not (adaptive["classification"]["cell"] == "cap_2000").any()
    assert manifest.empty


def test_legacy_generic_fit_error_requires_explicit_oom_evidence(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    lower = _make_cell(tmp_path, "cap_2000_n500", 500)
    _write_status(
        target,
        status="fail",
        reason="fit_error",
        n_train=985,
        error="No models were trained successfully",
    )
    (target / "seed_0" / "logs" / f"{KEY}_{MODEL}.log").write_text(
        "RuntimeError: CUDA out of memory\n", encoding="utf-8"
    )
    _write_status(lower, status="pass", reason="", n_train=500)
    _write_metric(lower, 0.73)

    _, adaptive, manifest = resolve_sample_fallbacks(
        tmp_path, ["cap_2000", "cap_2000_n500"]
    )

    assert (adaptive["classification"]["cell"] == "cap_2000").any()
    assert manifest.iloc[0]["status"] == "resolved"


def test_resolver_refuses_reuse_when_test_targets_differ(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    lower = _make_cell(tmp_path, "cap_2000_n500", 500)
    _write_status(target, status="fail", reason="fit_oom", n_train=985)
    _write_status(lower, status="pass", reason="", n_train=500)
    _write_metric(lower, 0.73)
    pd.DataFrame({"target": [1, 1]}, index=[10, 11]).to_csv(
        lower / "seed_0" / "predictions" / f"{KEY}_ground_truth.csv"
    )

    with pytest.raises(AssertionError):
        resolve_sample_fallbacks(tmp_path, ["cap_2000", "cap_2000_n500"])


def test_resolver_refuses_reuse_when_target_model_config_differs(tmp_path):
    target = _make_cell(tmp_path, "cap_2000", None)
    lower = _make_cell(tmp_path, "cap_2000_n500", 500)
    _write_status(target, status="fail", reason="fit_oom", n_train=985)
    _write_status(lower, status="pass", reason="", n_train=500)
    _write_metric(lower, 0.73)
    lower_config_path = lower / "config.json"
    lower_config = json.loads(lower_config_path.read_text(encoding="utf-8"))
    lower_config["model_overrides"] = {MODEL: {"presets": "different"}}
    lower_config_path.write_text(json.dumps(lower_config), encoding="utf-8")

    with pytest.raises(AssertionError):
        resolve_sample_fallbacks(tmp_path, ["cap_2000", "cap_2000_n500"])
