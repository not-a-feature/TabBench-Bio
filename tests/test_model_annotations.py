import json

from scripts import build_site, generate_social_preview
from tabbench_bio.config import parse_models


def test_existing_model_roster_contains_annotation():
    models = json.loads(build_site.MODEL_CONFIG.read_text(encoding="utf-8"))
    tabdpt = next(model for model in models if model["key"] == "TABDPT")
    assert tabdpt["training_data_overlap"] is True
    assert build_site.model_meta("TABDPT")["training_data_overlap"] is True
    assert ("TABDPT", "gpu", True) in parse_models(models)


def test_annotation_applies_to_new_model(monkeypatch):
    monkeypatch.setattr(build_site, "TRAINING_DATA_OVERLAP", {"TABDPT", "RF"})
    assert build_site.model_meta("TABDPT")["training_data_overlap"]
    assert build_site.model_meta("RF")["training_data_overlap"]
    assert not build_site.model_meta("LR")["training_data_overlap"]


def test_social_podium_excludes_all_flagged_models(tmp_path, monkeypatch):
    model_ids = ["DUMMY", "AUTOGLUON", "TABDPT", "NEW_MODEL", "RF", "LR", "GBM"]
    dashboard = {
        "meta": {"plot_excluded_models": ["DUMMY"], "configured_model_count": 7},
        "datasets": [],
        "models": {
            model_id: {"training_data_overlap": model_id in {"TABDPT", "NEW_MODEL"}}
            for model_id in model_ids
        },
        "reference": [
            {
                "model_id": model_id,
                "display": model_id,
                "Elo": 2000 - index * 100,
                "cell_label": "p=10000, n=100",
                "n_targets": 39,
            }
            for index, model_id in enumerate(model_ids)
        ],
    }
    path = tmp_path / "dashboard.json"
    path.write_text(json.dumps(dashboard), encoding="utf-8")
    selected = []
    monkeypatch.setattr(
        generate_social_preview,
        "draw_podium",
        lambda draw, leaders, *args: selected.extend(leaders),
    )
    generate_social_preview.generate_social_preview(path, tmp_path / "og.png")
    assert [row["model_id"] for row in selected] == ["RF", "LR", "GBM"]
