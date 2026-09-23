"""Explicit dataset-cell exclusions shared by all benchmark models."""

import json
from pathlib import Path

POLICY_PATH = Path(__file__).parent / "bio" / "data" / "benchmark_exclusions.json"
REASON = "benchmark_exclusion"


def excluded_dataset_keys(cell: str) -> set[str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    cells = policy["excluded_dataset_cells"]
    for name, keys in cells.items():
        assert isinstance(name, str) and name.startswith("cap_"), name
        assert isinstance(keys, list) and all(isinstance(key, str) for key in keys), keys
        assert len(keys) == len(set(keys)), keys
    return set(cells[cell]) if cell in cells else set()


def materialize_exclusions(repository, seeds, models, keys) -> set[str]:
    """Record exclusions before model fitting; retain previous attempts."""
    excluded = excluded_dataset_keys(repository.cell).intersection(keys)
    for seed in seeds:
        for key in sorted(excluded):
            for model in models:
                prior = repository.current(seed, key, model)
                if prior is not None and prior.status == "skip" and prior.reason == REASON:
                    continue
                repository.write(
                    {
                        "dataset": key,
                        "model": model,
                        "status": "skip",
                        "reason": REASON,
                        "error": "Explicit benchmark-wide dataset-cell exclusion (compute budget).",
                        "n_train_samples": None,
                        "n_test_samples": None,
                    },
                    seed=seed,
                )
    return excluded
