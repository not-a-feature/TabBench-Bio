"""Resolve model memory failures through completed lower-sample grid cells."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

MEMORY_FAILURE_REASONS = frozenset({"model_limit", "fit_oom", "inference_oom"})
_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory error",
    "not enough cuda memory",
    "cuda error: out of memory",
)
_CELL_RE = re.compile(r"^cap_(full|\d+)(?:_n(\d+))?$")
_METRIC_FILES = {
    "classification": "classification_metrics.csv",
    "regression": "regression_metrics.csv",
}
_INTENTIONAL_CONFIG_DIFFERENCES = frozenset(
    {
        "datasets_classification",
        "datasets_regression",
        "models",
        "output_dir",
        "train_subsample",
    }
)
_FALLBACK_COLUMNS = (
    "fallback",
    "nominal_n_train",
    "effective_n_train",
    "fallback_reason",
    "fallback_path",
    "reused_from_cell",
)


def parse_grid_cell(name: str) -> tuple[int | None, int | None]:
    """Return ``(feature_cap, sample_budget)`` encoded by a grid-cell directory."""
    match = _CELL_RE.fullmatch(name)
    assert match is not None, f"Unrecognised grid cell directory: {name}"
    cap_token, sample_token = match.groups()
    cap = None if cap_token == "full" else int(cap_token)
    sample = None if sample_token is None else int(sample_token)
    return cap, sample


def sample_fallback_chain(
    current_n: int,
    sample_budgets: list[int],
    *,
    max_start_n: int | None = None,
) -> list[int]:
    """Choose configured halving points, optionally jumping below a calibrated boundary."""
    assert current_n > 0
    remaining = {int(n) for n in sample_budgets if 0 < int(n) < current_n}
    chain = []
    if max_start_n is not None:
        assert max_start_n >= 0
        remaining = {n for n in remaining if n <= max_start_n}
        if not remaining:
            return chain
        current_n = max(remaining)
        chain.append(current_n)
        remaining.remove(current_n)
    while remaining:
        half = current_n / 2
        next_n = min(remaining, key=lambda n: (abs(n - half), n))
        chain.append(next_n)
        remaining = {n for n in remaining if n < next_n}
        current_n = next_n
    return chain


def log_has_memory_failure(log_path: str | os.PathLike[str]) -> bool:
    """Return whether a legacy unit log explicitly records an OOM."""
    path = Path(log_path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return any(marker in text for marker in _OOM_MARKERS)


def record_is_memory_failure(record: dict, log_path: str | os.PathLike[str]) -> bool:
    """Recognise new and legacy terminal records caused by GPU memory capacity."""
    reason = record["reason"] if "reason" in record else ""
    if reason in MEMORY_FAILURE_REASONS:
        return True
    error = record["error"] if "error" in record else ""
    return reason == "fit_error" and (
        any(marker in error.lower() for marker in _OOM_MARKERS)
        or log_has_memory_failure(log_path)
    )


def _load_config(cell_dir: Path) -> dict:
    path = cell_dir / "config.json"
    assert path.is_file(), f"Missing grid-cell config: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _model_relevant_config(config: dict, model: str) -> dict:
    """Remove axis/path differences and resolve model-indexed settings for one model."""
    shared = {
        key: value
        for key, value in config.items()
        if key not in _INTENTIONAL_CONFIG_DIFFERENCES
    }
    shared["model_overrides"] = config.get("model_overrides", {}).get(model)
    shared["model_limits"] = config.get("model_limits", {}).get(model)
    nan_policy = config.get("nan_policy") or {}
    shared["nan_policy"] = nan_policy.get(model, nan_policy.get("default", "native"))
    return shared


def _assert_compatible_cells(target_dir: Path, source_dir: Path, model: str) -> None:
    target = _load_config(target_dir)
    source = _load_config(source_dir)
    target_shared = _model_relevant_config(target, model)
    source_shared = _model_relevant_config(source, model)
    assert target_shared == source_shared, (
        f"Cannot reuse {source_dir.name} for {target_dir.name}: non-sample config differs."
    )
    assert model in target["models"], f"{model} is absent from {target_dir.name}/config.json"
    assert model in source["models"], f"{model} is absent from {source_dir.name}/config.json"


def _assert_identical_test_target(
    target_dir: Path,
    source_dir: Path,
    seed: int,
    key: str,
) -> None:
    relative = Path(f"seed_{seed}") / "predictions" / f"{key}_ground_truth.csv"
    target_path = target_dir / relative
    source_path = source_dir / relative
    assert target_path.is_file(), f"Missing target-cell ground truth: {target_path}"
    assert source_path.is_file(), f"Missing fallback-cell ground truth: {source_path}"
    target = pd.read_csv(target_path, index_col=0).sort_index()
    source = pd.read_csv(source_path, index_col=0).sort_index()
    pd.testing.assert_frame_equal(target, source, check_dtype=False)


def _load_status_records(cell_dir: Path) -> list[dict]:
    rows = []
    for path_text in sorted(glob.glob(str(cell_dir / "seed_*" / "stats" / "*.json"))):
        path = Path(path_text)
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.parents[1].name.removeprefix("seed_"))
        log_path = path.parents[1] / "logs" / f"{record['dataset']}_{record['model']}.log"
        assert "n_train_samples" in record, f"Legacy record lacks n_train_samples: {path}"
        rows.append(
            {
                "cell": cell_dir.name,
                "seed": seed,
                "key": record["dataset"],
                "model": record["model"],
                "status": record["status"],
                "reason": record["reason"] if "reason" in record else "",
                "error": record["error"] if "error" in record else "",
                "n_train_samples": int(record["n_train_samples"]),
                "recommended_max_n_train": (
                    int(record["recommended_max_n_train"])
                    if "recommended_max_n_train" in record
                    else None
                ),
                "memory_failure": record_is_memory_failure(record, log_path),
            }
        )
    return rows


def _tag_metrics(cell_dir: Path, task: str) -> pd.DataFrame:
    cap, sample = parse_grid_cell(cell_dir.name)
    path = cell_dir / "metrics" / _METRIC_FILES[task]
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    required = {"seed", "key", "dataset", "model"}
    assert required.issubset(frame.columns), f"{path} lacks {sorted(required - set(frame.columns))}"
    frame.insert(0, "task", task)
    frame.insert(0, "n_train", "full" if sample is None else sample)
    frame.insert(0, "max_features", "full" if cap is None else cap)
    frame.insert(0, "cell", cell_dir.name)
    return frame


def resolve_sample_fallbacks(
    results_root: str | os.PathLike[str],
    cell_names: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Return strict metrics, adaptive metrics, and the fallback manifest.

    Source cell artifacts are read-only. A fallback row is a copy of a completed lower-cell
    metric row tagged with both its nominal and effective sample counts.
    """
    root = Path(results_root)
    assert cell_names, "No grid cells supplied for sample-fallback resolution."
    assert len(cell_names) == len(set(cell_names)), "Duplicate grid cells supplied."
    cells = {name: root / name for name in cell_names}
    parsed = {name: parse_grid_cell(name) for name in cell_names}

    status_rows = []
    strict_frames = {task: [] for task in _METRIC_FILES}
    for name, cell_dir in cells.items():
        assert cell_dir.is_dir(), f"Missing grid cell: {cell_dir}"
        status_rows.extend(_load_status_records(cell_dir))
        for task in _METRIC_FILES:
            frame = _tag_metrics(cell_dir, task)
            if not frame.empty:
                strict_frames[task].append(frame)

    status = pd.DataFrame(status_rows)
    assert not status.empty, f"No unit status records found under {root}"
    assert not status.duplicated(["cell", "seed", "key", "model"]).any(), (
        "Duplicate unit status records found."
    )
    status_lookup = {
        (row.cell, int(row.seed), row.key, row.model): row
        for row in status.itertuples(index=False)
    }

    strict = {}
    metric_lookup = {}
    for task, frames in strict_frames.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not frame.empty:
            assert not frame.duplicated(["cell", "seed", "key", "model"]).any(), (
                f"Duplicate {task} metric rows found."
            )
            nominal = []
            for row in frame.to_dict("records"):
                unit = (row["cell"], int(row["seed"]), row["key"], row["model"])
                assert unit in status_lookup, f"Metric row has no status record: {unit}"
                record = status_lookup[unit]
                assert record.status == "pass", f"Metric row is not a passing unit: {unit}"
                nominal.append(int(record.n_train_samples))
                assert unit not in metric_lookup, f"Unit appears in both task metric files: {unit}"
                metric_lookup[unit] = (task, row)
            frame["fallback"] = False
            frame["nominal_n_train"] = nominal
            frame["effective_n_train"] = nominal
            frame["fallback_reason"] = ""
            frame["fallback_path"] = ""
            frame["reused_from_cell"] = ""
        strict[task] = frame

    passing_units = {
        (row.cell, int(row.seed), row.key, row.model)
        for row in status[status["status"] == "pass"].itertuples(index=False)
    }
    missing_passing_metrics = sorted(passing_units - set(metric_lookup))
    assert not missing_passing_metrics, (
        f"{len(missing_passing_metrics)} passing unit(s) lack a metric row; first: "
        f"{missing_passing_metrics[:5]}"
    )

    sample_budgets_by_cap = {}
    cell_by_coordinate = {}
    for name, (cap, sample) in parsed.items():
        cell_by_coordinate[(cap, sample)] = name
        if sample is not None:
            sample_budgets_by_cap.setdefault(cap, []).append(sample)

    adaptive_additions = {task: [] for task in _METRIC_FILES}
    manifest = []

    duplicate_units = status[
        (status["status"] == "skip") & (status["reason"] == "duplicate_cell")
    ]
    for target in duplicate_units.sort_values(
        ["cell", "seed", "key", "model"]
    ).itertuples(index=False):
        target_cap, target_sample = parsed[target.cell]
        assert target_sample is not None, target.cell
        source_cell = cell_by_coordinate[(target_cap, None)]
        source_unit = (source_cell, int(target.seed), target.key, target.model)
        manifest_row = {
            "cell": target.cell,
            "seed": int(target.seed),
            "key": target.key,
            "model": target.model,
            "status": "unresolved",
            "reason": target.reason,
            "nominal_n_train": int(target.n_train_samples),
            "effective_n_train": pd.NA,
            "fallback_path": json.dumps([target.cell, source_cell]),
            "reused_from_cell": "",
            "terminal_reason": "missing_source_status",
        }
        if source_unit not in status_lookup:
            manifest.append(manifest_row)
            continue
        source_status = status_lookup[source_unit]
        manifest_row["terminal_reason"] = (
            f"source_{source_status.status}:{source_status.reason}"
        )
        if source_status.status != "pass":
            manifest.append(manifest_row)
            continue

        assert source_unit in metric_lookup, (
            f"Passing duplicate-cell source has no metric row: {source_unit}"
        )
        task, source_row = metric_lookup[source_unit]
        target_dir = cells[target.cell]
        source_dir = cells[source_cell]
        _assert_compatible_cells(target_dir, source_dir, target.model)
        truth_relative = (
            Path(f"seed_{int(target.seed)}")
            / "predictions"
            / f"{target.key}_ground_truth.csv"
        )
        validation_cells = [
            cells[name]
            for name, (cap, sample) in sorted(
                parsed.items(),
                key=lambda item: (
                    item[1][1] is None,
                    item[1][1] if item[1][1] is not None else 0,
                ),
            )
            if cap == target_cap and (cells[name] / truth_relative).is_file()
        ]
        assert validation_cells, (
            f"No same-cap fold target is available to validate duplicate unit {source_unit}"
        )
        _assert_identical_test_target(
            validation_cells[0], source_dir, int(target.seed), target.key
        )

        row = dict(source_row)
        row["cell"] = target.cell
        row["max_features"] = "full" if target_cap is None else target_cap
        row["n_train"] = target_sample
        row["fallback"] = True
        row["nominal_n_train"] = int(target.n_train_samples)
        row["effective_n_train"] = int(source_status.n_train_samples)
        row["fallback_reason"] = target.reason
        row["fallback_path"] = json.dumps([target.cell, source_cell])
        row["reused_from_cell"] = source_cell
        adaptive_additions[task].append(row)

        manifest_row.update(
            {
                "status": "resolved",
                "effective_n_train": int(source_status.n_train_samples),
                "reused_from_cell": source_cell,
                "terminal_reason": "",
            }
        )
        manifest.append(manifest_row)

    eligible = status[(status["memory_failure"]) & (status["status"].isin(["fail", "skip"]))]
    for target in eligible.sort_values(["cell", "seed", "key", "model"]).itertuples(index=False):
        target_cap, _ = parsed[target.cell]
        budgets = sample_budgets_by_cap[target_cap]
        max_start_n = (
            int(target.recommended_max_n_train)
            if pd.notna(target.recommended_max_n_train)
            else None
        )
        chain = sample_fallback_chain(
            int(target.n_train_samples), budgets, max_start_n=max_start_n
        )
        attempted = [int(target.n_train_samples)]
        resolved = None
        terminal_reason = "no_smaller_sample_cell"

        for budget in chain:
            attempted.append(budget)
            source_cell = cell_by_coordinate[(target_cap, budget)]
            source_unit = (source_cell, int(target.seed), target.key, target.model)
            if source_unit not in status_lookup:
                terminal_reason = f"missing_status:{source_cell}"
                break
            source_status = status_lookup[source_unit]
            if source_status.status == "pass":
                assert source_unit in metric_lookup, (
                    f"Passing fallback source has no metric row: {source_unit}"
                )
                resolved = (source_cell, source_status, metric_lookup[source_unit])
                break
            if not source_status.memory_failure:
                terminal_reason = (
                    f"non_memory_{source_status.status}:{source_status.reason or 'unknown'}"
                )
                break
            terminal_reason = "all_smaller_cells_exhausted"

        manifest_row = {
            "cell": target.cell,
            "seed": int(target.seed),
            "key": target.key,
            "model": target.model,
            "status": "unresolved",
            "reason": target.reason,
            "nominal_n_train": int(target.n_train_samples),
            "effective_n_train": pd.NA,
            "fallback_path": json.dumps(attempted),
            "reused_from_cell": "",
            "terminal_reason": terminal_reason,
        }
        if resolved is None:
            manifest.append(manifest_row)
            continue

        source_cell, source_status, (task, source_row) = resolved
        target_dir = cells[target.cell]
        source_dir = cells[source_cell]
        _assert_compatible_cells(target_dir, source_dir, target.model)
        _assert_identical_test_target(target_dir, source_dir, int(target.seed), target.key)

        row = dict(source_row)
        target_cap, target_sample = parsed[target.cell]
        row["cell"] = target.cell
        row["max_features"] = "full" if target_cap is None else target_cap
        row["n_train"] = "full" if target_sample is None else target_sample
        row["fallback"] = True
        row["nominal_n_train"] = int(target.n_train_samples)
        row["effective_n_train"] = int(source_status.n_train_samples)
        row["fallback_reason"] = target.reason
        row["fallback_path"] = json.dumps(attempted)
        row["reused_from_cell"] = source_cell
        adaptive_additions[task].append(row)

        manifest_row.update(
            {
                "status": "resolved",
                "effective_n_train": int(source_status.n_train_samples),
                "reused_from_cell": source_cell,
                "terminal_reason": "",
            }
        )
        manifest.append(manifest_row)

    adaptive = {}
    for task in _METRIC_FILES:
        additions = pd.DataFrame(adaptive_additions[task])
        if strict[task].empty:
            frame = additions
        elif additions.empty:
            frame = strict[task].copy()
        else:
            frame = pd.concat([strict[task], additions], ignore_index=True)
        if not frame.empty:
            assert not frame.duplicated(["cell", "seed", "key", "model"]).any(), (
                f"Adaptive {task} metrics contain duplicate units."
            )
            missing = [column for column in _FALLBACK_COLUMNS if column not in frame.columns]
            assert not missing, f"Adaptive metrics lack fallback metadata: {missing}"
        adaptive[task] = frame

    manifest_columns = [
        "cell",
        "seed",
        "key",
        "model",
        "status",
        "reason",
        "nominal_n_train",
        "effective_n_train",
        "fallback_path",
        "reused_from_cell",
        "terminal_reason",
    ]
    return strict, adaptive, pd.DataFrame(manifest, columns=manifest_columns)
