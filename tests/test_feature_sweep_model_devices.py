"""Device-lane scheduling and Ferranti H100 safety tests."""

import pytest

from scripts.feature_sweep import _resolve_model_devices, _scope_cell_specs


def test_default_scheduler_keeps_both_model_lanes():
    selected, guarded = _resolve_model_devices([], None)

    assert selected == {"cpu", "gpu"}
    assert not guarded


def test_ferranti_h100_job_is_implicitly_gpu_only():
    selected, guarded = _resolve_model_devices([], "h100-ferranti")

    assert selected == {"gpu"}
    assert guarded


def test_ferranti_h100_job_rejects_cpu_lane():
    with pytest.raises(ValueError, match="may schedule only GPU models"):
        _resolve_model_devices(["cpu"], "h100-ferranti")


def test_gpu_scope_removes_cpu_tier_without_mutating_frozen_spec():
    original = {
        "full_cfg": "config.json",
        "gpu_solo_cfg": "config_gpu_solo.json",
        "gpu_shared_cfg": "config_gpu_shared.json",
        "cpu_cfg": "config_cpu.json",
        "n_splits": 5,
        "tag": "cap 2000/n=20",
    }

    scoped = _scope_cell_specs([original], {"gpu"})

    assert scoped[0]["gpu_solo_cfg"] == "config_gpu_solo.json"
    assert scoped[0]["gpu_shared_cfg"] == "config_gpu_shared.json"
    assert scoped[0]["cpu_cfg"] is None
    assert original["cpu_cfg"] == "config_cpu.json"
