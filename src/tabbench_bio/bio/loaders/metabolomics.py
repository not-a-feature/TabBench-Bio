"""ST001000 stool metabolomics from the primary AN001880 measurement table."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.io_utils import download_file

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


def prepare_st001000(
    table: pd.DataFrame, *, cohort: str = "PRISM"
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Keep measured features and one disease cohort; preserve source non-detections."""
    assert cohort in {"PRISM", "Validation"}
    assert table.columns[:2].tolist() == ["Samples", "group"]
    columns = table.columns[2:]
    assert len(columns) and columns.is_unique
    assert all(re.fullmatch(r"\d+(?:\.\d+)?_\d+(?:\.\d+)?", c) for c in columns), (
        "ST001000: expected only mass/retention-time measurement columns after Samples/group"
    )
    identifiers = table["Samples"].astype(str)
    assert identifiers.is_unique and table["Samples"].notna().all()
    assert set(table["group"].unique()) <= {"Diagnosis:CD", "Diagnosis:UC", "Diagnosis:Control"}
    assert identifiers.str.fullmatch(r"\d+|UMCGIBD\d+|LLDeep_\d+").all(), (
        "ST001000: unrecognized participant IDs; verify cohort membership before ingestion"
    )
    in_cohort = identifiers.str.fullmatch(r"\d+" if cohort == "PRISM" else r"UMCGIBD\d+")
    selected = table.loc[in_cohort & table["group"].isin(["Diagnosis:CD", "Diagnosis:UC"])].copy()
    selected = selected.sort_values("Samples").reset_index(drop=True)
    assert len(selected), f"ST001000: no disease participants for {cohort}"
    abundance = selected.loc[:, columns].apply(pd.to_numeric, errors="raise")
    assert not np.isinf(abundance.to_numpy()).any() and not abundance.lt(0).any().any()
    assert not abundance.duplicated().any(), "ST001000: duplicated participant profiles"
    # This transform has no fitted parameters. Filtering and imputation belong to training.
    X = np.log1p(abundance.astype(np.float32))
    y = selected["group"].map({"Diagnosis:UC": 0, "Diagnosis:CD": 1}).rename("diagnosis")
    groups = (cohort + "|" + selected["Samples"].astype(str)).rename("participant_id")
    assert groups.is_unique and not y.isna().any() and set(y) == {0, 1}
    metadata = {
        "study": "ST001000",
        "analysis": "AN001880",
        "cohort": cohort,
        "class_names": {"0": "UC", "1": "CD"},
        "source_rows": len(table),
        "n_features": X.shape[1],
        "n_participants": len(X),
        "participant_ids": groups.tolist(),
        "transform": "log1p, no fitted parameters",
        "source_missing_fraction": float(abundance.isna().to_numpy().mean()),
        "missingness": "Source NA non-detections retained; no global filtering or imputation",
        "excluded_predictors": ["Samples", "group"],
        "external_validation": "43 UMCGIBD disease participants retained in the source file; excluded from the PRISM task",
    }
    return X, y.astype(int), groups, metadata


class MetabolomicsLoader:
    """Fetch the frozen primary ST001000 discovery cohort without clinical predictors."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or (default_bio_cache_dir() / "metabolomics_raw"))

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        assert spec.fetch_id == "ST001000" and spec.target == "diagnosis"
        assert spec.problem_type == "binary"
        assert spec.download_url and spec.download_sha256, (
            f"{spec.bio_id}: missing download configuration"
        )
        path = download_file(
            spec.download_url,
            self.cache_dir / "ST001000_AN001880.txt",
            checksum=spec.download_sha256,
        )
        table = pd.read_csv(path, sep="\t", dtype={"Samples": str})
        X, y, groups, metadata = prepare_st001000(table)
        assert X.shape == (121, 2121) and y.value_counts().to_dict() == {1: 68, 0: 53}
        metadata["source_sha256"] = spec.download_sha256
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            groups=groups,
            problem_type="binary",
            license=spec.license,
            source_url=spec.source_url,
            citation=spec.citation,
            metadata=metadata,
        )
