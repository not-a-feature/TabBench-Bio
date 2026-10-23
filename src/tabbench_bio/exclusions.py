"""Explicit dataset-cell exclusions shared by all benchmark models."""

import json
from pathlib import Path

from tabbench_bio.tuning import tuning_fingerprint

POLICY_PATH = Path(__file__).parent / "bio" / "data" / "benchmark_exclusions.json"
REASON = "benchmark_exclusion"
DUMMY_SKIP_REASON = "dummy_unavailable"


def dummy_skip_record(source_id, source, model, tuning):
    """Record why a shared split is ineligible without replacing its source attempt."""
    prior_gate = source["reason"] == DUMMY_SKIP_REASON
    record = {
        "dataset": source["dataset"],
        "model": model,
        "status": "skip",
        "reason": DUMMY_SKIP_REASON,
        "error": "DUMMY did not pass; this dataset/cell/fold is excluded for every model.",
        "n_train_samples": source["n_train_samples"] if "n_train_samples" in source else None,
        "n_test_samples": source["n_test_samples"] if "n_test_samples" in source else None,
        "dummy_attempt_id": source["dummy_attempt_id"] if prior_gate else source_id,
        "dummy_status": source["dummy_status"] if prior_gate else source["status"],
        "dummy_reason": source["dummy_reason"] if prior_gate else source["reason"],
    }
    if model in tuning:
        record["tuning"] = {
            "fingerprint": tuning_fingerprint(tuning[model]),
            "specification": tuning[model],
            "status": "skipped",
        }
    return record


def materialize_dummy_skips(repository, seeds, models, keys) -> set[str]:
    """Skip unattempted peers when the matching DUMMY outcome is a failure or skip."""
    excluded = set()
    tuning = (
        repository.cell_config["model_tuning"] if "model_tuning" in repository.cell_config else {}
    )
    for seed in seeds:
        for key in keys:
            dummy = repository.current(seed, key, "DUMMY")
            if dummy is None or dummy.status == "pass":
                continue
            excluded.add(key)
            for model in dict.fromkeys(models):
                if model == "DUMMY":
                    continue
                prior = repository.current(seed, key, model)
                if prior is not None:
                    assert prior.status == "skip", (
                        f"Model attempted without a passing DUMMY: {seed}/{key}/{model}"
                    )
                    continue
                repository.write(
                    dummy_skip_record(dummy.attempt_id, dummy.record, model, tuning), seed=seed
                )
    return excluded


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
