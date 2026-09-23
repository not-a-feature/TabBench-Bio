"""Network-free tests for streamed GEO series-matrix parsing."""

from __future__ import annotations

import io

import pandas as pd

from tabbench_bio.bio.loaders.geo_matrix import (
    GEO_TITLE_PLATE,
    RECORDS,
    _curate_target_labels,
    _parse_series_metadata,
    parse_processed_signals,
    parse_series_matrix,
)


def _matrix(characteristic: str) -> io.StringIO:
    return io.StringIO(
        "\n".join(
            [
                '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"',
                '!Sample_title\t"plate-a_R01C01"\t"plate-a_R02C01"\t"plate-b_R01C01"',
                f'!Sample_characteristics_ch1\t"{characteristic}: 1"\t"{characteristic}: 2"\t"{characteristic}: 1"',
                '!Sample_characteristics_ch1\t"batch: a"\t"batch: b"\t"batch: b"',
                "!series_matrix_table_begin",
                '"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"',
                '"cg1"\t"0.1"\t"0.2"\t"0.3"',
                '"cg2"\t"0.4"\t"0.5"\t"0.6"',
                '"cg3"\t"0.7"\t"0.8"\t"0.9"',
                "!series_matrix_table_end",
            ]
        )
    )


def test_parse_series_matrix_aligns_samples_and_caps_without_values():
    X, y, groups, metadata = parse_series_matrix(
        _matrix("age"),
        bio_id="GEO-test",
        target="age",
        problem_type="regression",
        max_features=2,
    )
    assert X.shape == (3, 2)
    assert y.tolist() == [1.0, 2.0, 1.0]
    assert groups.tolist() == ["GSM1", "GSM2", "GSM3"]
    assert metadata["raw_feature_count"] == 3
    assert metadata["selected_features"] == 2


def test_parse_series_matrix_extracts_class_labels():
    _, y, _, _ = parse_series_matrix(
        _matrix("disease state"),
        bio_id="GEO-test",
        target="disease state",
        problem_type="binary",
        max_features=2,
    )
    assert y.tolist() == ["1", "2", "1"]


def test_parse_series_matrix_can_group_by_technical_batch():
    _, _, groups, metadata = parse_series_matrix(
        _matrix("age"),
        bio_id="GEO-test",
        target="age",
        problem_type="regression",
        max_features=2,
        group_characteristic="batch",
    )
    assert groups.tolist() == ["a", "b", "b"]
    assert metadata["group_strategy"] == "GEO sample characteristic: batch"


def test_parse_series_matrix_can_derive_assay_plate_from_titles():
    _, _, groups, metadata = parse_series_matrix(
        _matrix("status"),
        bio_id="GEO-test",
        target="status",
        problem_type="binary",
        max_features=2,
        group_characteristic=GEO_TITLE_PLATE,
    )
    assert groups.tolist() == ["plate-a", "plate-a", "plate-b"]
    assert metadata["group_strategy"] == "GEO sample-title prefix before '_' (assay plate)"


def test_schizophrenia_label_curation_excludes_controls_and_normalizes_case():
    X = pd.DataFrame({"cg1": [0.1, 0.2, 0.3, 0.4, 0.5]})
    y = pd.Series(["Case", "case", "Control", "control", "NA"], name="status")
    groups = pd.Series(["p1", "p2", "p3", "p4", "control"])
    X, y, groups, metadata = _curate_target_labels(
        X,
        y,
        groups,
        {"n_samples": 5},
        bio_id="GEO-GSE147221-Methylation-Schizophrenia",
        record=RECORDS["GSE147221"],
    )
    assert X.shape == (4, 1)
    assert y.tolist() == ["case", "case", "control", "control"]
    assert groups.tolist() == ["p1", "p2", "p3", "p4"]
    assert metadata["excluded_control_samples"] == 1
    assert metadata["n_samples"] == 4


def test_schizophrenia_excludes_missing_technical_controls_before_probe_filtering():
    stream = io.StringIO(
        "\n".join(
            [
                '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"',
                '!Sample_title\t"plate-a_R01C01"\t"plate-a_R02C01"\t"plate-b_R01C01"\t"control_R01C01"',
                '!Sample_characteristics_ch1\t"status: Case"\t"status: Control"\t"status: case"\t"status: NA"',
                "!series_matrix_table_begin",
                '"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"',
                '"cg1"\t"0.1"\t"0.2"\t"0.3"\t"null"',
                '"cg2"\t"0.4"\t"0.5"\t"0.6"\t"null"',
                "!series_matrix_table_end",
            ]
        )
    )
    X, y, groups, metadata = parse_series_matrix(
        stream,
        bio_id="GEO-GSE147221-Methylation-Schizophrenia",
        target="status",
        problem_type="binary",
        max_features=2,
        group_characteristic=GEO_TITLE_PLATE,
        excluded_targets=("NA",),
    )
    X, y, groups, metadata = _curate_target_labels(
        X,
        y,
        groups,
        metadata,
        bio_id="GEO-GSE147221-Methylation-Schizophrenia",
        record=RECORDS["GSE147221"],
    )
    assert X.shape == (3, 2)
    assert y.tolist() == ["case", "control", "case"]
    assert groups.tolist() == ["plate-a", "plate-a", "plate-b"]
    assert metadata["excluded_control_samples"] == 1


def test_schizophrenia_processed_supplement_aligns_metadata_and_ignores_pvalues():
    metadata_stream = io.StringIO(
        "\n".join(
            [
                '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"',
                '!Sample_title\t"plate-a_R01C01: genomic DNA"\t"plate-a_R02C01: genomic DNA"\t"plate-b_R01C01: genomic DNA"\t"control_R01C01: Meth Control"',
                '!Sample_characteristics_ch1\t"status: Case"\t"status: Control"\t"status: case"\t"status: NA"',
                "!series_matrix_table_begin",
            ]
        )
    )
    sample_accessions, sample_titles, characteristics = _parse_series_metadata(
        metadata_stream,
        bio_id="GEO-GSE147221-Methylation-Schizophrenia",
        target="status",
    )
    supplement = io.StringIO(
        "\n".join(
            [
                ',"plate-a_R01C01","plate-a_R01C01_Detection_Pval","plate-a_R02C01","plate-a_R02C01_Detection_Pval","plate-b_R01C01","plate-b_R01C01_Detection_Pval"',
                '"cg1","0.1","0.01","0.2","0.02","0.3","0.03"',
                '"cg2","0.4","0.04","0.5","0.05","0.6","0.06"',
                '"cg3","0.7","0.07","0.8","0.08","0.9","0.09"',
            ]
        )
    )
    X, y, groups, metadata = parse_processed_signals(
        supplement,
        bio_id="GEO-GSE147221-Methylation-Schizophrenia",
        sample_accessions=sample_accessions,
        sample_titles=sample_titles,
        characteristics=characteristics,
        target="status",
        max_features=2,
        group_characteristic=GEO_TITLE_PLATE,
        excluded_targets=("NA",),
    )
    X, y, groups, metadata = _curate_target_labels(
        X,
        y,
        groups,
        metadata,
        bio_id="GEO-GSE147221-Methylation-Schizophrenia",
        record=RECORDS["GSE147221"],
    )
    assert X.shape == (3, 2)
    assert X.max().max() <= 0.9
    assert y.tolist() == ["case", "control", "case"]
    assert groups.tolist() == ["plate-a", "plate-a", "plate-b"]
    assert metadata["series_samples"] == 4
    assert metadata["supplement_beta_samples"] == 3
    assert metadata["series_samples_absent_from_supplement"] == 1
    assert metadata["excluded_control_samples"] == 1


def test_smoking_label_curation_maps_published_codes():
    X = pd.DataFrame({"cg1": [0.1, 0.2, 0.3]})
    y = pd.Series(["0", "1", "2"], name="smoking")
    groups = pd.Series(["a", "b", "c"])
    _, y, _, _ = _curate_target_labels(
        X,
        y,
        groups,
        {"n_samples": 3},
        bio_id="GEO-GSE50660-Methylation-Smoking",
        record=RECORDS["GSE50660"],
    )
    assert y.tolist() == ["never", "former", "current"]
