import pytest

from tabbench_bio.resource_guard import HardwareMismatchError, check_reference_hardware


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "81920")
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "32")
    monkeypatch.delenv("TABBENCH_STRICT_HARDWARE", raising=False)


def test_reference_hardware_passes_silently(reference, capsys):
    observed = check_reference_hardware(gpu_required=True, gpu_name="NVIDIA L40S", strict=True)
    assert observed["allocated_cpus"] == 16
    assert observed["allocated_memory_mb"] == 81920
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "variable,value,fragment",
    [
        ("SLURM_CPUS_PER_TASK", "18", "allocated CPUs=18"),
        ("SLURM_MEM_PER_NODE", "92160", "allocated system RAM=92160"),
    ],
)
def test_mismatch_warns_or_fails(reference, monkeypatch, capsys, variable, value, fragment):
    monkeypatch.setenv(variable, value)
    check_reference_hardware(gpu_required=True, gpu_name="NVIDIA L40S")
    assert fragment in capsys.readouterr().err
    monkeypatch.setenv("TABBENCH_STRICT_HARDWARE", "1")
    with pytest.raises(HardwareMismatchError, match=fragment):
        check_reference_hardware(gpu_required=True, gpu_name="NVIDIA L40S")


def test_wrong_gpu_and_unverifiable_allocation_fail_strict(reference, monkeypatch):
    with pytest.raises(HardwareMismatchError, match="expected NVIDIA L40S"):
        check_reference_hardware(gpu_required=True, gpu_name="NVIDIA A100", strict=True)
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    with pytest.raises(HardwareMismatchError, match="cannot be verified"):
        check_reference_hardware(strict=True)


def test_cpu_only_runs_and_memory_per_cpu(reference, monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "5120")
    assert check_reference_hardware(strict=True)["allocated_memory_mb"] == 81920


def test_main_prediction_entry_fails_before_creating_results(reference, monkeypatch, tmp_path):
    from tabbench_bio import predictions

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "18")
    monkeypatch.setenv("TABBENCH_STRICT_HARDWARE", "1")
    monkeypatch.setattr(predictions, "AutoGluonModel", object())
    output = tmp_path / "results"
    with pytest.raises(HardwareMismatchError, match="allocated CPUs=18"):
        predictions.compute_predictions(
            {"subsample": None, "models": ["RF"], "output_dir": str(output)}
        )
    assert not output.exists()
