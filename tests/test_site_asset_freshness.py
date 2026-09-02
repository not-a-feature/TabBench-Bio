import hashlib
import json

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


def test_write_latex_leaderboard_uses_current_reference_rows(tmp_path):
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
            "display": "TabPFN-Wide",
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
    assert "1 & TabPFN-Wide & 1100 & [1000, 1200] & 44 & 0.750" in table
    assert "AutoGluon" not in table
    assert "Constant" not in table


@pytest.mark.parametrize("manifest_in_parent", [False, True])
def test_resolve_aggregation_manifest_supports_both_export_layouts(
    tmp_path, manifest_in_parent
):
    aggregated = tmp_path / "generated" / "data"
    aggregated.mkdir(parents=True)
    manifest_dir = aggregated.parent if manifest_in_parent else aggregated
    manifest = manifest_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    assert build_site.resolve_aggregation_manifest(aggregated, None) == manifest.resolve()


def test_sitemap_keeps_citation_and_machine_readable_routes():
    sitemap = build_site.build_sitemap("https://tabbench-bio.eu", "2026-08-30")

    assert "https://tabbench-bio.eu/citation.html" in sitemap
    assert "https://tabbench-bio.eu/CITATION.cff" in sitemap
    assert "https://tabbench-bio.eu/data/raw/index.json" in sitemap


def test_llms_metadata_counts_classification_tasks_and_keeps_strict_export():
    dashboard = {
        "meta": {
            "snapshot_utc": "2026-08-30T18:30:00Z",
            "configured_model_count": 1,
            "evaluation_points_per_model": 10,
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
        "models": {"LR": {"display": "Logistic Reg."}},
        "cell_options": [{"id": "cap_10000_n100"}],
        "reference": [
            {
                "display": "Logistic Reg.",
                "cell_label": "p=10,000, n=100",
                "Elo": 1000,
                "Elo_lo": 900,
                "Elo_hi": 1100,
                "n_targets": 2,
            }
        ],
    }

    text = build_site.build_llms_text(dashboard, "https://tabbench-bio.eu")

    assert "2 registered datasets: 1 classification, 1 regression" in text
    assert "sweep_summary_strict.json" in text
    assert "CITATION.cff" in text
