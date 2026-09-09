import hashlib
import json

import pandas as pd
import pytest

from scripts import build_site


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_metadata_must_match_current_configuration(tmp_path, monkeypatch):
    classification = tmp_path / "classification.json"
    regression = tmp_path / "regression.json"
    classification.write_text(json.dumps(["old", "new"]), encoding="utf-8")
    regression.write_text(json.dumps(["regression"]), encoding="utf-8")
    monkeypatch.setattr(build_site, "CLASSIFICATION_DATASETS", classification)
    monkeypatch.setattr(build_site, "REGRESSION_DATASETS", regression)

    with pytest.raises(AssertionError, match="missing=\\['new'\\]"):
        build_site.assert_dataset_metadata_current(
            [{"dataset_id": "old"}, {"dataset_id": "regression"}]
        )

    build_site.assert_dataset_metadata_current(
        [
            {"dataset_id": "old"},
            {"dataset_id": "new"},
            {"dataset_id": "regression"},
        ]
    )


def test_generated_input_manifest_must_match_files(tmp_path, monkeypatch):
    current_input = tmp_path / "input.json"
    current_input.write_text("current", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "path": "input.json",
                        "sha256": sha256(current_input),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_site, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(build_site, "GENERATED_MANIFEST", manifest)

    build_site.assert_generated_inputs_current()
    current_input.write_text("stale", encoding="utf-8")

    with pytest.raises(AssertionError, match="input.json"):
        build_site.assert_generated_inputs_current()


def test_site_build_writes_current_latex_leaderboard(tmp_path):
    output = tmp_path / "leaderboard_table.tex"
    reference = [
        {
            "model_id": "AUTOGLUON",
            "display": "AutoGluon",
            "Elo": 1200,
            "Elo_lo": 1100,
            "Elo_hi": 1300,
            "n_targets": 57,
            "f1_macro": 0.8,
        },
        {
            "model_id": "TABPFN-WIDE",
            "display": "TabPFN Wide (8k)",
            "Elo": 1100,
            "Elo_lo": 1000,
            "Elo_hi": 1200,
            "n_targets": 44,
            "f1_macro": 0.75,
        },
        {
            "model_id": "DUMMY",
            "display": "Constant",
            "Elo": 500,
            "Elo_lo": 400,
            "Elo_hi": 600,
            "n_targets": 57,
            "f1_macro": 0.2,
        },
    ]

    build_site.write_latex_leaderboard(reference, output)

    table = output.read_text(encoding="utf-8")
    assert "1 & TabPFN Wide (8k) & 1100 & [1000, 1200] & 44 & 0.750" in table
    assert "AutoGluon" not in table
    assert "Constant" not in table


def test_sitemap_keeps_citation_and_machine_readable_routes():
    sitemap = build_site.build_sitemap("https://tabbench-bio.eu", "2026-08-30")

    assert "https://tabbench-bio.eu/citation.html" in sitemap
    assert "https://tabbench-bio.eu/CITATION.cff" in sitemap
    assert "https://tabbench-bio.eu/data/raw/index.json" not in sitemap


def test_llms_metadata_describes_strict_primary_and_adaptive_sensitivity():
    dashboard = {
        "meta": {
            "snapshot_utc": "2026-08-30T18:30:00Z",
            "configured_model_count": 1,
            "evaluation_points_per_model": 10,
            "paper_url": "https://arxiv.org/abs/XXXX.XXXXX",
        },
        "progress": {
            "recorded": 8,
            "expected": 10,
            "fraction": 0.8,
            "status": {"pass": 7, "skip": 1, "fail": 0},
        },
        "datasets": [
            {"task": "Classification", "modality": "Gene expression"},
            {"task": "Regression", "modality": "Molecular properties"},
        ],
        "models": {"LR": {"display": "Logistic Regression", "training_data_overlap": False}},
        "cell_options": [{"id": "cap_10000_n100"}],
        "reference": [
            {
                "display": "Logistic Regression",
                "cell_label": "p=10,000, n=100",
                "Elo": 1000,
                "Elo_lo": 900,
                "Elo_hi": 1100,
                "n_targets": 2,
            }
        ],
        "raw_exports": [
            {
                "format": "sqlite3",
                "available": False,
                "path": "",
                "records": 123,
                "bytes": 456,
                "sha256": "abc123",
            }
        ],
    }

    text = build_site.build_llms_text(dashboard, "https://tabbench-bio.eu")

    assert "2 registered datasets: 1 classification, 1 regression" in text
    assert "primary charts and rankings use strict nominal-cell results" in text
    assert "data/raw/" not in text
    assert "Canonical results SQLite: release upload pending" in text
    assert "CITATION.cff" in text
    assert "## Training-data overlap" not in text
    dashboard["models"]["TABDPT"] = {"display": "TabDPT", "training_data_overlap": True}
    dashboard["models"]["NEW"] = {"display": "New model", "training_data_overlap": True}
    annotated = build_site.build_llms_text(dashboard, "https://tabbench-bio.eu")
    for name in ("TabDPT", "New model"):
        assert f"† {name}: Part of the benchmark training data was used in the training process of this model." in annotated
    assert "† Logistic Regression" not in annotated
    dashboard["models"]["NEW"]["training_data_overlap"] = False
    assert "† New model" not in build_site.build_llms_text(dashboard, "https://tabbench-bio.eu")



def test_artifact_index_contains_only_canonical_sqlite(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "results.sqlite"
    sqlite_path.write_bytes(b"database")
    monkeypatch.setattr(build_site, "_sqlite_attempt_count", lambda path: 123)

    exports = build_site.build_artifact_index(sqlite_path, "https://example.test/results.sqlite")

    assert len(exports) == 1
    assert exports[0]["format"] == "sqlite3"
    assert exports[0]["available"] is True


def test_leaderboard_export_contains_both_analysis_views(tmp_path):
    output = tmp_path / "leaderboard.json"
    dashboard = {
        "meta": {"snapshot_utc": "2026-09-02T20:30:05Z", "reference_cell": "cap_10000_n100"},
        "models": {"LR": {"display": "Logistic Regression"}},
        "cell_options": [{"id": "cap_10000_n100"}],
        "reference": [{"model_id": "LR", "Elo": 1000}],
        "cell_elo": [{"cell": "cap_10000_n100", "model_id": "LR", "Elo": 1000}],
        "analysis_views": {
            "strict": {"reference": [{"model_id": "LR"}], "cell_elo": []},
            "adaptive": {"reference": [{"model_id": "LR"}], "cell_elo": []},
        },
    }

    entry = build_site.write_leaderboard_export(dashboard, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert set(payload["analysis_views"]) == {"strict", "adaptive"}
    assert payload["primary_analysis_view"] == "strict"
    assert entry["path"] == "data/leaderboard.json"


def test_adaptive_cost_uses_the_reused_source_cell_timing(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    generated.mkdir()
    pd.DataFrame(
        [
            {
                "cell": "cap_100_n100",
                "dataset": "dataset-id",
                "key": "target-id",
                "seed": 0,
                "model": "MODEL",
                "f1_macro": 0.8,
                "fallback": True,
                "reused_from_cell": "cap_100_n50",
            }
        ]
    ).to_csv(generated / "sweep_metrics_classification_adaptive.csv", index=False)
    run_stats = tmp_path / "run_stats.csv"
    pd.DataFrame(
        [
            {
                "cell": "cap_100_n50",
                "seed": 0,
                "dataset": "target-id",
                "model": "MODEL",
                "status": "pass",
                "train_time_s": 3.0,
                "inference_time_s": 0.5,
            }
        ]
    ).to_csv(run_stats, index=False)
    monkeypatch.setattr(build_site, "GENERATED_DATA", generated)

    rows = build_site.build_cost_grid(
        {"MODEL": {"display": "Model", "category": "Classical"}},
        [{"dataset_id": "dataset-id", "modality": "Modality"}],
        ["cap_100_n100"],
        ["all", "Modality"],
        ["f1_macro"],
        {"run_stats": run_stats},
        "_adaptive",
    )

    assert {row["cell"] for row in rows} == {"cap_100_n100"}
    assert {row["train_time_s"] for row in rows} == {3.0}
