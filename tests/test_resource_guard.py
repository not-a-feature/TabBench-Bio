import json
from contextlib import nullcontext

import pandas as pd
import pytest

from tabbench_bio import predictions
from tabbench_bio.dataset import TaskType
from tabbench_bio.resource_guard import HardwareMismatchError, check_reference_hardware
from tabbench_bio.result_store import ResultRepository


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
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "18")
    monkeypatch.setenv("TABBENCH_STRICT_HARDWARE", "1")
    monkeypatch.setattr(predictions, "AutoGluonModel", object())
    output = tmp_path / "results"
    with pytest.raises(HardwareMismatchError, match="allocated CPUs=18"):
        predictions.compute_predictions(
            {"subsample": None, "models": ["RF"], "output_dir": str(output)}
        )
    assert not output.exists()


@pytest.mark.parametrize("existing_metadata", [False, True])
def test_lower_level_runner_records_observed_allocations(
    reference, monkeypatch, debug_config, existing_metadata
):
    monkeypatch.delenv("TABBENCH_RESOURCE_PROFILE", raising=False)
    monkeypatch.delenv("TABBENCH_BIO_RESOURCE_TIER", raising=False)
    monkeypatch.delenv("TABBENCH_MODEL_HARDWARE", raising=False)
    if existing_metadata:
        monkeypatch.setenv("TABBENCH_MODEL_HARDWARE", json.dumps({"model_threads": 32}))
    frame = pd.DataFrame({"feature": range(6), "target": [0.0, 1.0] * 3})

    class Benchmark:
        _key_list = ["toy_0"]
        _task_type_list = [TaskType.Regression]

        def __len__(self):
            return 1

        def __iter__(self):
            yield frame.iloc[:4], frame.iloc[4:], "toy_0", TaskType.Regression

        def training_groups(self, key, data):
            return None

    class Model:
        def __init__(self, **kwargs):
            self.predictor = None
            self.autogluon_path = ""

        def fit(self, data, *, groups):
            pass

        def predict(self, data):
            return data[["target"]]

        def get_fit_stats(self):
            return {}

    monkeypatch.setattr(predictions, "AutoGluonModel", Model)
    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "_time_budget", lambda seconds: nullcontext())
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    monkeypatch.setattr(predictions, "_cleanup_model", lambda model: None)
    predictions.compute_predictions(debug_config)
    attempts = ResultRepository.from_root(debug_config["output_dir"]).current_attempts()
    assert len(attempts) == 2
    for attempt in attempts:
        assert attempt.status == "pass"
        assert attempt.record["hardware"]["allocated_cpus"] == 16
        assert attempt.record["hardware"]["allocated_memory_mb"] == 81920
        if existing_metadata:
            assert attempt.record["hardware"]["model_threads"] == 32
