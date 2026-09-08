from argparse import Namespace
from copy import deepcopy

from tabbench_bio.cli import _apply_run_filters


def _args(**overrides):
    values = {
        "model": None,
        "include_model": [],
        "exclude_model": [],
        "include_data_type": [],
        "include_dataset": [],
        "exclude_data_type": [],
    }
    values.update(overrides)
    return Namespace(**values)


def _config():
    return {
        "models": ["GBM", "MITRA", "TABFM", "TABPFN-V3"],
        "datasets_classification": ["OpenML-1138", "OpenML-1458"],
        "datasets_regression": ["OpenML-46983", "OpenML-430"],
    }


def test_dataset_filter_preserves_models_and_excludes_other_datasets():
    config = _config()
    _apply_run_filters(config, _args(include_dataset=["OpenML-1138"]))
    assert config["datasets_classification"] == ["OpenML-1138"]
    assert config["datasets_regression"] == []
    assert config["models"] == _config()["models"]


def test_gene_expression_phase_filters_only_the_dataset_modality():
    config = _config()

    _apply_run_filters(config, _args(include_data_type=["Gene Expression"]))

    assert config == {
        "models": ["GBM", "MITRA", "TABFM", "TABPFN-V3"],
        "datasets_classification": ["OpenML-1138"],
        "datasets_regression": ["OpenML-46983"],
    }


def test_gene_expression_phase_filters_loaded_runtime_dataset_names():
    config = {
        **_config(),
        "dataset_names_classification": ["OpenML-1138", "OpenML-1458"],
        "dataset_names_regression": ["OpenML-46983", "OpenML-430"],
    }

    _apply_run_filters(config, _args(include_data_type=["Gene Expression"]))

    assert config["datasets_classification"] == ["OpenML-1138", "OpenML-1458"]
    assert config["datasets_regression"] == ["OpenML-46983", "OpenML-430"]
    assert config["dataset_names_classification"] == ["OpenML-1138"]
    assert config["dataset_names_regression"] == ["OpenML-46983"]


def test_regular_then_deferred_model_phases_partition_remaining_units():
    regular = _config()
    deferred = deepcopy(regular)

    _apply_run_filters(
        regular,
        _args(
            exclude_model=["MITRA", "TABFM", "TABPFN-V3"],
            exclude_data_type=["Gene Expression"],
        ),
    )
    _apply_run_filters(
        deferred,
        _args(
            include_model=["MITRA", "TABFM", "TABPFN-V3"],
            exclude_data_type=["Gene Expression"],
        ),
    )

    assert regular == {
        "models": ["GBM"],
        "datasets_classification": ["OpenML-1458"],
        "datasets_regression": ["OpenML-430"],
    }
    assert deferred == {
        "models": ["MITRA", "TABFM", "TABPFN-V3"],
        "datasets_classification": ["OpenML-1458"],
        "datasets_regression": ["OpenML-430"],
    }
