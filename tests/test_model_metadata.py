"""Public metadata is authored once in JSON; Python supplies execution defaults."""

import json
import shutil
from pathlib import Path

import pytest

from tabbench_bio import dashboard, model_metadata
from tabbench_bio.model_metadata import (
    PUBLIC_METADATA_FIELDS,
    hidden_model_ids,
    load_model_catalog,
    load_model_metadata,
)
from tabbench_bio.model_registry import EXECUTION_REGISTRY, load_registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "model_metadata.json"
    shutil.copyfile(ROOT / "configs/model_metadata.json", path)
    return path


def edit_catalog(path, change):
    rows = json.loads(path.read_text())
    change(rows)
    path.write_text(json.dumps(rows))


def test_json_edits_control_public_capabilities_and_website(catalog):
    def change(rows):
        entry = next(row for row in rows if row["key"] == "FTFM")
        entry.update(
            display="Edited FTFM",
            category="Edited category",
            description="Edited public model description",
            source_url="https://example.com/model",
            hidden=True,
            training_data_overlap=True,
            supported_tasks=["classification", "regression"],
            max_features=123,
            max_classes=7,
        )

    edit_catalog(catalog, change)
    spec = load_registry(catalog)["FTFM"]
    assert spec.display == "Edited FTFM" and spec.category == "Edited category"
    assert spec.max_features == 123 and spec.max_classes == 7
    assert spec.supported_tasks == ("classification", "regression")
    assert spec.hidden and spec.training_data_overlap
    assert spec.device == EXECUTION_REGISTRY["FTFM"].device
    assert spec.adapter == EXECUTION_REGISTRY["FTFM"].adapter
    metadata = load_model_metadata(catalog)
    public = dashboard.model_meta("FTFM", metadata)
    assert public["display"] == "Edited FTFM"
    assert public["source_url"] == "https://example.com/model"
    assert public["regular_max_features"] == 123
    assert public["training_data_overlap"] and not public["classification_only"]
    assert "FTFM" in hidden_model_ids(metadata)


def test_variants_inherit_flags_limits_and_tasks(catalog):
    def change(rows):
        next(row for row in rows if row["key"] == "RF").update(
            hidden=True,
            training_data_overlap=True,
            max_features=99,
            supported_tasks=["classification"],
        )

    edit_catalog(catalog, change)
    metadata = load_model_metadata(catalog)
    variant = metadata["RF-TUNED"]
    assert variant["hidden"] and variant["training_data_overlap"]
    assert variant["max_features"] == 99
    assert variant["supported_tasks"] == ["classification"]
    assert "RF-TUNED" in hidden_model_ids(metadata)
    assert dashboard.model_meta("RF-TUNED", metadata)["classification_only"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden", "true"),
        ("training_data_overlap", 1),
        ("display", ""),
        ("max_features", 0),
        ("max_classes", True),
        ("max_classes", 1),
        ("supported_tasks", []),
        ("supported_tasks", ["classification", "classification"]),
        ("supported_tasks", ["unknown"]),
        ("environment", "gpu"),
    ],
)
def test_invalid_metadata_fails_with_model_and_field(catalog, field, value):
    edit_catalog(
        catalog,
        lambda rows: next(row for row in rows if row["key"] == "RF").update({field: value}),
    )
    with pytest.raises(ValueError, match=rf"RF.*{field}"):
        load_model_metadata(catalog)


def test_missing_fields_and_duplicate_ids_are_rejected(catalog):
    edit_catalog(catalog, lambda rows: rows[0].pop("category"))
    with pytest.raises(ValueError, match="missing public metadata.*category"):
        load_model_metadata(catalog)
    shutil.copyfile(ROOT / "configs/model_metadata.json", catalog)
    edit_catalog(catalog, lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(ValueError, match="duplicate model key"):
        load_model_metadata(catalog)


@pytest.mark.parametrize("parent", ["RF-TUNED", "UNKNOWN"])
def test_invalid_parent_inheritance_is_rejected(catalog, parent):
    edit_catalog(
        catalog,
        lambda rows: next(row for row in rows if row["key"] == "RF").update(base_model=parent),
    )
    with pytest.raises(ValueError, match="cyclic|unknown metadata parent"):
        load_model_metadata(catalog)


def test_all_execution_models_require_metadata(catalog):
    edit_catalog(
        catalog, lambda rows: rows.remove(next(row for row in rows if row["key"] == "FASTAI"))
    )
    with pytest.raises(ValueError, match="Missing public metadata.*FASTAI"):
        load_registry(catalog)


def test_packaged_catalog_and_rosters_have_separate_roles(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.setattr(model_metadata, "resource_root", lambda: tmp_path)
    metadata = load_model_metadata()
    assert load_registry()["SELDON"].supported_tasks == ("classification", "regression")
    assert metadata["SELDON-MINI"]["hidden"]
    assert metadata["KUMO-TABULAR-MEDIUM"]["hidden"]
    assert not metadata["FTFM"]["training_data_overlap"]
    assert metadata["TABDPT"]["training_data_overlap"]
    roster = json.loads((tmp_path / "configs/models/all.json").read_text())
    assert {"FASTAI", "TABPFN"}.isdisjoint(row["key"] for row in roster)
    for path in (tmp_path / "configs/models").glob("*.json"):
        for entry in json.loads(path.read_text()):
            assert PUBLIC_METADATA_FIELDS.isdisjoint(entry)
            assert "device" not in entry and "environment" not in entry


def test_roster_cannot_silently_override_public_metadata(tmp_path):
    from tabbench_bio.doctor import check_metadata

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    path = tmp_path / "configs/models/all.json"
    edit_catalog(path, lambda rows: rows[0].update(hidden=False))
    with pytest.raises(ValueError, match="move DUMMY metadata.*hidden"):
        load_model_catalog(tmp_path)
    problems = check_metadata(tmp_path)
    assert any("move DUMMY metadata" in problem for problem in problems)
