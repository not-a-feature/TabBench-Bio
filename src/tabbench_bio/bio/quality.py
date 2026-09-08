"""Explicit quality gates for candidate TabBench Bio datasets.

The gates are intentionally separate from the benchmark execution path: they are
run while curating or refreshing source data and produce an auditable report without
silently changing any dataset at model-training time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from tabbench_bio.bio.loaders.base import BioRawDataset


@dataclass(frozen=True)
class QualityThresholds:
    """Minimum evidence required before a candidate enters the active roster."""

    min_samples: int = 150
    min_features: int = 1_000
    min_nonconstant_features: int = 100
    min_class_samples: int = 30
    min_class_fraction: float = 0.05
    min_groups: int = 5
    min_groups_per_class: int = 5
    min_regression_targets: int = 20


def assess_raw_dataset(
    raw: BioRawDataset,
    thresholds: QualityThresholds = QualityThresholds(),
) -> dict:
    """Return a JSON-serialisable pass/fail report for one raw dataset."""
    issues: list[str] = []
    n_samples, n_features = raw.X.shape
    if len(raw.y) != n_samples:
        issues.append(f"X/y length mismatch ({n_samples} != {len(raw.y)})")
    if raw.groups is not None and len(raw.groups) != n_samples:
        issues.append(f"X/groups length mismatch ({n_samples} != {len(raw.groups)})")
    if n_samples < thresholds.min_samples:
        issues.append(f"only {n_samples} samples (<{thresholds.min_samples})")
    if n_features < thresholds.min_features:
        issues.append(f"only {n_features} features (<{thresholds.min_features})")
    if raw.X.columns.duplicated().any():
        issues.append(f"{int(raw.X.columns.duplicated().sum())} duplicate feature names")

    numeric = raw.X.select_dtypes(include="number")
    non_numeric_features = n_features - numeric.shape[1]
    if non_numeric_features:
        issues.append(f"{non_numeric_features} non-numeric features")
    numeric_values = numeric.to_numpy(dtype=np.float64, copy=False)
    nonfinite_values = int((~np.isfinite(numeric_values)).sum()) if numeric_values.size else 0
    if nonfinite_values:
        issues.append(f"{nonfinite_values} missing/non-finite feature values")
    nonconstant_features = int((numeric.nunique(dropna=False) > 1).sum()) if numeric.shape[1] else 0
    if nonconstant_features < thresholds.min_nonconstant_features:
        issues.append(
            f"only {nonconstant_features} non-constant features "
            f"(<{thresholds.min_nonconstant_features})"
        )

    missing_targets = int(raw.y.isna().sum())
    if missing_targets:
        issues.append(f"{missing_targets} missing targets")

    class_counts: dict[str, int] | None = None
    groups_per_class: dict[str, int] | None = None
    if raw.problem_type in {"binary", "multiclass"}:
        counts = raw.y.value_counts(dropna=False)
        class_counts = {str(label): int(count) for label, count in counts.items()}
        if raw.problem_type == "binary" and len(counts) != 2:
            issues.append(f"binary task has {len(counts)} classes")
        if raw.problem_type == "multiclass" and len(counts) < 3:
            issues.append(f"multiclass task has only {len(counts)} classes")
        if len(counts) and int(counts.min()) < thresholds.min_class_samples:
            issues.append(
                f"smallest class has {int(counts.min())} samples (<{thresholds.min_class_samples})"
            )
        if len(counts) and float(counts.min() / len(raw.y)) < thresholds.min_class_fraction:
            issues.append(
                f"smallest class fraction is {float(counts.min() / len(raw.y)):.3f} "
                f"(<{thresholds.min_class_fraction:.3f})"
            )
    elif raw.problem_type == "regression":
        target = pd.to_numeric(raw.y, errors="coerce").to_numpy(dtype=np.float64)
        nonfinite_targets = int((~np.isfinite(target)).sum())
        if nonfinite_targets:
            issues.append(f"{nonfinite_targets} non-numeric/non-finite targets")
        unique_targets = int(pd.Series(target[np.isfinite(target)]).nunique())
        if unique_targets < thresholds.min_regression_targets:
            issues.append(
                f"only {unique_targets} distinct regression targets "
                f"(<{thresholds.min_regression_targets})"
            )
    else:
        issues.append(f"unknown problem type {raw.problem_type!r}")

    n_groups: int | None = None
    if raw.groups is not None:
        missing_groups = int(raw.groups.isna().sum())
        if missing_groups:
            issues.append(f"{missing_groups} missing group identifiers")
        n_groups = int(raw.groups.nunique(dropna=True))
        if n_groups < thresholds.min_groups:
            issues.append(f"only {n_groups} independence groups (<{thresholds.min_groups})")
        if raw.problem_type in {"binary", "multiclass"} and len(raw.groups) == len(raw.y):
            group_table = pd.DataFrame(
                {"target": raw.y.to_numpy(), "group": raw.groups.to_numpy()}
            ).dropna()
            per_class = group_table.groupby("target")["group"].nunique()
            groups_per_class = {str(label): int(count) for label, count in per_class.items()}
            if len(per_class) and int(per_class.min()) < thresholds.min_groups_per_class:
                issues.append(
                    f"smallest class spans {int(per_class.min())} groups "
                    f"(<{thresholds.min_groups_per_class})"
                )

    return {
        "bio_id": raw.bio_id,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "nonconstant_features": nonconstant_features,
        "missing_targets": missing_targets,
        "nonfinite_feature_values": nonfinite_values,
        "n_groups": n_groups,
        "class_counts": class_counts,
        "groups_per_class": groups_per_class,
        "thresholds": asdict(thresholds),
        "source_metadata": dict(raw.metadata),
    }
