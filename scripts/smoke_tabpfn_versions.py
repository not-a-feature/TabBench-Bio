"""Check V3 and V3.5 checkpoint selection and GPU predictions before benchmarking."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from tabbench_bio.models.tabpfn_v3 import TabPFNV3Model
from tabbench_bio.models.tabpfn_v3_5 import TabPFNV35Model


def main():
    assert torch.cuda.is_available(), "This smoke test requires a CUDA GPU."
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(30, 12)), columns=[f"x{i}" for i in range(12)])
    targets = {
        "binary": pd.Series(np.arange(30) % 2),
        "regression": pd.Series(X["x0"] + 0.2 * rng.normal(size=30)),
    }
    for cls in (TabPFNV3Model, TabPFNV35Model):
        for task, y in targets.items():
            with TemporaryDirectory(prefix="tabbench-tabpfn-smoke-") as directory:
                model = cls(
                    path=directory,
                    name=cls.ag_name,
                    problem_type=task,
                )
                model.fit(X=X, y=y, num_cpus=1, num_gpus=1)
                prediction = model.predict(X.iloc[:4])
                assert prediction.shape == (4,)
                assert np.isfinite(prediction).all()
                if task == "binary":
                    probabilities = model.predict_proba(X.iloc[:4])
                    assert np.isfinite(probabilities).all()
                    assert ((probabilities >= 0) & (probabilities <= 1)).all()
                backend = (
                    model.model.model_ if cls is TabPFNV3Model and task == "binary" else model.model
                )
                checkpoint = Path(backend.model_path).name
                expected = (
                    f"tabpfn-v3-{'classifier' if task == 'binary' else 'regressor'}-v3_default.ckpt"
                    if cls is TabPFNV3Model
                    else "tabpfn-v3.5-20260909.safetensors"
                )
                assert checkpoint == expected, (checkpoint, expected)
                print(f"PASS {cls.ag_key} {task}: {checkpoint}", flush=True)
                del backend, model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
