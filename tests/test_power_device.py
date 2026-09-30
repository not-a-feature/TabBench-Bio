import sys
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from tabbench_bio import predictions


@pytest.mark.parametrize("visible", ["2", "3,1", "GPU-allocated", ""])
def test_power_uses_cuda_device_identity(monkeypatch, visible):
    uuid = UUID("12345678-1234-1234-1234-123456789abc")
    cuda = SimpleNamespace(
        is_available=lambda: bool(visible),
        current_device=lambda: 0,
        get_device_properties=Mock(return_value=SimpleNamespace(uuid=uuid)),
    )
    nvml = SimpleNamespace(nvmlDeviceGetHandleByUUID=Mock(return_value="allocated-handle"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(predictions, "_pynvml", nvml)
    handle = predictions._visible_gpu_handle()
    if visible:
        assert handle == "allocated-handle"
        nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-" + str(uuid))
        cuda.get_device_properties.assert_called_once_with(0)
    else:
        assert handle is None
        nvml.nvmlDeviceGetHandleByUUID.assert_not_called()


def test_unresolved_device_does_not_fall_back_to_physical_zero(monkeypatch, caplog):
    predictions._gpu_power_handle.cache_clear()
    monkeypatch.setattr(predictions, "_HAS_PYNVML", True)
    monkeypatch.setattr(predictions, "_HAS_RAPL", False)
    monkeypatch.setattr(predictions, "_pynvml", SimpleNamespace(NVMLError=RuntimeError))

    def unavailable():
        raise RuntimeError("UUID unavailable")

    monkeypatch.setattr(predictions, "_visible_gpu_handle", unavailable)
    with predictions._PowerTracker() as tracker:
        assert tracker._gpu_handle is None
    assert tracker.gpu_mean_power_w is None
    assert tracker.gpu_energy_j is None

    with predictions._PowerTracker():
        pass
    assert caplog.text.count("GPU power measurement unavailable") == 1
    predictions._gpu_power_handle.cache_clear()


def test_power_device_is_resolved_once_per_process(monkeypatch):
    predictions._gpu_power_handle.cache_clear()
    monkeypatch.setattr(predictions, "_HAS_PYNVML", True)
    resolver = Mock(return_value="handle")
    monkeypatch.setattr(predictions, "_visible_gpu_handle", resolver)
    assert predictions._gpu_power_handle(1) == predictions._gpu_power_handle(1) == "handle"
    resolver.assert_called_once()
    assert predictions._gpu_power_handle(2) == "handle"
    assert resolver.call_count == 2
    predictions._gpu_power_handle.cache_clear()
