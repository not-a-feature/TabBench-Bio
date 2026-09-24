"""Explicit device selection and per-fit CPU budgets."""

import pytest

from scripts.feature_sweep import _fit_cpu_budget, _resolve_model_devices, _scope_cell_specs


@pytest.mark.parametrize(
    "requested,expected", [([], {"cpu", "gpu"}), (["cpu"], {"cpu"}), (["gpu"], {"gpu"})]
)
def test_device_selection(requested, expected):
    assert _resolve_model_devices(requested) == expected


@pytest.mark.parametrize("total,gpus,workers", [(160, 0, 5), (32, 1, 0), (64, 1, 1)])
def test_cpu_budget_counts_only_active_pools(total, gpus, workers):
    assert _fit_cpu_budget(total, gpus, workers) == 32


def test_gpu_scope_removes_cpu_tier_without_mutating_frozen_spec():
    original = {
        "full_cfg": "config.json",
        "gpu_cfgs": ["config_gpu.json"],
        "cpu_cfg": "config_cpu.json",
        "n_splits": 5,
        "tag": "cap 2000/n=20",
    }

    scoped = _scope_cell_specs([original], {"gpu"})

    assert scoped[0]["gpu_cfgs"] == ["config_gpu.json"]
    assert scoped[0]["cpu_cfg"] is None
    assert original["cpu_cfg"] == "config_cpu.json"
