import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd

from tabbench_bio import predictions
from tabbench_bio.dataset import TaskType
from tabbench_bio.result_store import ResultRepository


def test_prediction_pipeline_preserves_terminal_l40s_oom(tmp_path, monkeypatch, debug_config):
    cell = tmp_path / "experiment/cap_2000_n20"
    debug_config.update(output_dir=str(cell), models=["RF"])
    frame = pd.DataFrame({"feature": range(12), "target": [0, 1] * 6})
    calls = []

    class Benchmark:
        _key_list = ["OpenML-1138_0"]
        _task_type_list = [TaskType.Classification]

        def __len__(self):
            return 1

        def __iter__(self):
            yield frame.iloc[:8], frame.iloc[8:], self._key_list[0], self._task_type_list[0]

        def training_groups(self, key, train):
            return None

    class Model:
        def __init__(self, **kwargs):
            self.autogluon_path = None
            self.predictor = None

        def fit(self, data, *, groups):
            calls.append(len(data))
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(predictions, "configure_benchmark", lambda config: Benchmark())
    monkeypatch.setattr(predictions, "_set_global_seeds", lambda seed: None)
    monkeypatch.setattr(predictions, "AutoGluonModel", Model)
    monkeypatch.setattr(predictions, "_time_budget", lambda seconds: nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                OutOfMemoryError=MemoryError,
                get_device_properties=lambda index: SimpleNamespace(
                    name="NVIDIA L40S", total_memory=48 * 1024**3
                ),
                is_available=lambda: False,
            )
        ),
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "18")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "92160")
    monkeypatch.setenv("TABBENCH_MODEL_CPUS", "32")
    monkeypatch.setenv("TABBENCH_RESOURCE_PROFILE", "l40s-18c-90g-32threads-v1")
    monkeypatch.setattr(
        predictions.os, "sched_getaffinity", lambda _: set(range(32)), raising=False
    )
    monkeypatch.setenv("TABBENCH_BIO_RESOURCE_TIER", "l40s")
    predictions.compute_predictions(debug_config)
    repository = ResultRepository.from_root(cell.parent, cell=cell.name)
    (attempt,) = repository.current_attempts()
    assert (attempt.status, attempt.reason) == ("fail", "fit_oom")
    assert attempt.record["allocated_cpus"] == 32
    assert attempt.record["slurm_cpus_per_task"] == 18
    assert attempt.record["slurm_memory_mb"] == 92160
    assert attempt.record["cpu_affinity"] == list(range(32))
    predictions.compute_predictions(debug_config)
    assert len(calls) == 1
