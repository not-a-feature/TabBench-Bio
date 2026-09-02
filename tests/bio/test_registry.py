"""Unit tests for the configurable bio dataset registry (network-free)."""

from __future__ import annotations

import json

import pytest

from tabbench_bio.bio import datasets as ds
from tabbench_bio.bio.datasets import BioDatasetSpec, load_specs


def test_bundled_registry_loads_and_validates():
    specs = load_specs()  # the bundled data/bio_datasets.json
    assert specs, "registry is empty"
    # bio_id uniqueness is enforced by load_specs; spot-check a known entry.
    assert "OpenML-1138" in specs
    assert specs["OpenML-1138"].source == "openml"


def test_curated_datasets_enabled_and_safe_sources_present():
    runnable = ds.runnable_specs()
    by_source = {s.source for s in runnable}
    assert {
        "openml",
        "tcga",
        "geo",
        "geo_matrix",
        "fusionai",
        "local",
        "mgnify",
        "metagenomics",
        "tdc",
        "gp",
        "chembl",
    } <= by_source
    assert all(s.is_curated for s in runnable)
    assert all(s.bio_id != "kaggle-drsaeedmohsen-eeghumanemotiondataset2021" for s in runnable)
    assert all(s.bio_id != "local-GABBA_Evo2-VEP" for s in runnable)


def test_problem_type_filtering():
    assert "OpenML-1138" in ds.bio_dataset_names("binary")
    assert "GEO-GSE10893" in ds.bio_dataset_names("multiclass")
    assert "OpenML-1138" not in ds.bio_dataset_names("multiclass")


def test_disabled_stub_is_listed_but_not_runnable():
    dataset = "kaggle-drsaeedmohsen-eeghumanemotiondataset2021"
    assert ds.is_bio_dataset(dataset)
    assert dataset not in ds.bio_dataset_names()


def test_brca_embeddings_require_group_identifiers():
    specs = load_specs()
    for name in ("local-BRCA_Evo2-VEP", "local-BRCA_DNABERT2-VEP", "local-BRCA_NTv2-VEP"):
        assert specs[name].group_column == "group_id"


def test_protein_embeddings_are_enabled_grouped_and_not_rehosted():
    specs = load_specs()
    expected = {
        "local-DeepLoc2-Fungi-ESM2": ("binary", "homology_partition"),
        "local-PEER-BetaLactamase-ESM2": ("regression", "mutation_position"),
    }
    for name, (problem_type, group_column) in expected.items():
        spec = specs[name]
        assert spec.enabled
        assert spec.data_type == "Protein Embedding"
        assert spec.problem_type == problem_type
        assert spec.group_column == group_column
        assert spec.embedding_column is None
        assert not spec.redistributable


def test_new_embedding_arcene_and_methylation_panels_are_curated():
    specs = load_specs()
    fusion = specs["FusionAI-NTv2-GeneFusion"]
    assert fusion.source == "fusionai"
    assert fusion.fetch_id == "18713246/nt-middle"
    assert fusion.data_type == "DNA Embedding"
    assert fusion.problem_type == "binary"
    assert fusion.redistributable

    arcene = specs["OpenML-1458"]
    assert arcene.fetch_id == "1458"
    assert arcene.target == "Class"
    assert arcene.data_type == "Mass Spectrometry"

    smoking = specs["GEO-GSE50660-Methylation-Smoking"]
    schizophrenia = specs["GEO-GSE147221-Methylation-Schizophrenia"]
    assert smoking.problem_type == "multiclass"
    assert schizophrenia.problem_type == "binary"
    assert schizophrenia.group_column == "geo_title_plate"
    assert all(spec.max_features == 30_000 for spec in (smoking, schizophrenia))


def test_tdc_endpoints_are_grouped_and_not_marked_for_rehosting():
    specs = load_specs()
    tdc = [spec for spec in specs.values() if spec.source == "tdc" and spec.enabled]
    assert len(tdc) == 6
    assert all(spec.target == "Y" for spec in tdc)
    assert all(not spec.redistributable for spec in tdc)


def test_genomic_prediction_panels_are_the_four_curated_regressions():
    specs = load_specs()
    gp = {spec.bio_id: spec for spec in specs.values() if spec.source == "gp" and spec.enabled}
    assert set(gp) == {
        "gp-maize-FT",
        "gp-rice-FT",
        "gp-soy-YLD",
        "gp-switchgrass-HT",
    }
    assert all(spec.problem_type == "regression" for spec in gp.values())
    assert all(spec.redistributable for spec in gp.values())


def test_chembl_endpoints_are_curated_regressions_not_marked_for_rehosting():
    specs = load_specs()
    chembl = [spec for spec in specs.values() if spec.source == "chembl" and spec.enabled]
    assert {spec.bio_id for spec in chembl} == {"CHEMBL206", "CHEMBL2034"}
    assert all(spec.target == "pchembl_value" for spec in chembl)
    assert all(spec.problem_type == "regression" for spec in chembl)
    assert all(not spec.redistributable for spec in chembl)


def test_unknown_field_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"bio_id": "x", "source": "openml", "fetch_id": "1", "bogus": 1}]))
    with pytest.raises(ValueError, match="unknown field"):
        load_specs(bad)


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        BioDatasetSpec(bio_id="x", source="nope", fetch_id="1")


def test_resolved_eval_metric_defaults_by_problem_type():
    spec = BioDatasetSpec(
        bio_id="x", source="openml", fetch_id="1", target="t", problem_type="binary"
    )
    assert spec.resolved_eval_metric() == "roc_auc"
