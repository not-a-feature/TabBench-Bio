"""The configurable registry of TabBench-Bio datasets.

The registry is the single source of truth for *which* datasets make up the bio
benchmark and *how each is turned into a supervised task* (curated target column,
problem type). It is loaded from a JSON file so the list is **easily configurable
without editing Python**:

- Edit the bundled ``data/bio_datasets.json`` (ships with the package), or
- Point ``$TABBENCH_BIO_DATASETS`` at your own JSON file to override it entirely.

Each entry is a :class:`BioDatasetSpec` pairing a source-specific fetch identifier with a
**curated** task definition. Targets are curated and frozen on purpose: heuristic target
detection is fine for discovery but not for a published benchmark. Entries with
``target=null`` (or ``enabled=false``) are listed but not runnable until curated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

BioSource = (
    "geo",
    "geo_matrix",
    "fusionai",
    "tcga",
    "kaggle",
    "openml",
    "local",
    "mgnify",
    "metagenomics",
    "tdc",
    "gp",
    "chembl",
)
BioProblemType = ("binary", "multiclass", "regression")

#: Default eval metric per problem type (provenance only; the pipeline computes its own
#: metric suite when ranking — this records the intended primary metric).
DEFAULT_EVAL_METRIC: dict[str, str] = {
    "binary": "roc_auc",
    "multiclass": "log_loss",
    "regression": "root_mean_squared_error",
}

#: Env var to point at a custom dataset-registry JSON (overrides the bundled file).
_REGISTRY_ENV = "TABBENCH_BIO_DATASETS"

#: Bundled registry shipped as package data.
_BUNDLED_REGISTRY = Path(__file__).parent / "data" / "bio_datasets.json"


@dataclass(frozen=True)
class BioDatasetSpec:
    """One benchmark dataset: where to fetch it and how to frame the task.

    Attributes
    ----------
    bio_id : str
        Stable, filesystem-safe TabBench-Bio identifier (the ``dataset_name`` used in
        configs, cache keys, and results).
    display_name : str | None
        Human-friendly name shown on the site (Datasets tab, grid selectors, figures).
        ``None`` falls back to :attr:`bio_id`.
    source : str
        Which loader fetches it (``"geo" | "geo_matrix" | "fusionai" | "tcga" | "kaggle" |
        "openml" | "local" | "mgnify" | "metagenomics" | "tdc" | "gp" |
        "chembl"``).
    fetch_id : str
        Source-specific identifier passed to the loader (GEO accession, optionally
        ``"<accession>@<GPL>"``; a ``geo_matrix`` accession for a streamed GEO series
        matrix; the pinned FusionAI NT representation; OpenML dataset id; Kaggle ``owner/slug``; TCGA
        ``project/data_category``; ``local`` path to a bundled CSV, relative to the
        local-data dir or absolute; MGnify study accession (``MGYSxxxxxxxx``);
        ``metagenomics`` MetAML disease code, e.g. ``"cirrhosis"``; or a curated
        Therapeutics Data Commons endpoint slug (e.g. ``"bbb_martins"``); a curated
        genomic-prediction crop; or a ChEMBL target identifier.
    data_type : str | None
        Curated biological data *modality* (e.g. ``"Gene Expression"``, ``"EEG"``,
        ``"Methylation"``, ``"Misc"``). Independent of :attr:`source` (where the data was
        fetched): this is *what kind of data it is*, and is what the site groups by as the
        "Dataset type" in the benchmark breakdown. ``None`` falls back to the source label.
    target : str | None
        Curated target column / characteristic key. ``None`` means *not yet curated* —
        the dataset cannot be built until set.
    problem_type : str | None
        ``"binary" | "multiclass" | "regression"``. ``None`` until the target is curated.
    eval_metric : str | None
        Intended primary metric; falls back to :data:`DEFAULT_EVAL_METRIC`.
    enabled : bool
        Whether this dataset is part of the active benchmark. Disabled entries are
        listed (so the full intended set is documented) but skipped by default.
    redistributable : bool
        Whether the processed data may be re-hosted (provenance / future HF mirror).
    license : str | None
        License string captured for provenance.
    data_file : str | None
        Kaggle-only: which table to load when a dataset ships more than one.
    embedding_column : str | None
        ``local``-only: name of a column holding a per-row embedding as a single
        comma-separated string of floats (e.g. a DNA/protein language-model embedding).
        The local loader explodes it into one numeric feature column per dimension
        (``"<embedding_column>_<i>"``). ``None`` treats every non-target column as a
        plain feature.
    group_column : str | None
        Optional source-table column identifying the biological independence unit.
        The loader removes it from the features and the benchmark keeps equal values
        in one fold.
    max_features : int | None
        Per-dataset feature-cap override; ``None`` uses the benchmark-wide cap.
    notes : str
        Free-text curation notes (label semantics, caveats).
    """

    bio_id: str
    source: str
    fetch_id: str
    display_name: str | None = None
    data_type: str | None = None
    target: str | None = None
    problem_type: str | None = None
    eval_metric: str | None = None
    enabled: bool = True
    redistributable: bool = True
    license: str | None = None
    data_file: str | None = None
    embedding_column: str | None = None
    group_column: str | None = None
    max_features: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source not in BioSource:
            raise ValueError(
                f"{self.bio_id}: unknown source {self.source!r} (expected one of {BioSource})."
            )
        if self.problem_type is not None and self.problem_type not in BioProblemType:
            raise ValueError(f"{self.bio_id}: invalid problem_type {self.problem_type!r}.")

    @property
    def is_curated(self) -> bool:
        """Whether the dataset has a curated target + problem type (i.e. is buildable)."""
        return self.target is not None and self.problem_type is not None

    @property
    def is_classification(self) -> bool:
        return self.problem_type in ("binary", "multiclass")

    def resolved_eval_metric(self) -> str:
        """Eval metric to use, applying the per-problem-type default when unset."""
        if self.eval_metric is not None:
            return self.eval_metric
        if self.problem_type is None:
            raise ValueError(
                f"{self.bio_id}: problem_type must be curated before resolving a metric."
            )
        return DEFAULT_EVAL_METRIC[self.problem_type]


def _registry_path() -> Path:
    """The active registry JSON path ($TABBENCH_BIO_DATASETS or the bundled file)."""
    override = os.environ.get(_REGISTRY_ENV)
    return Path(override) if override else _BUNDLED_REGISTRY


def load_specs(path: str | Path | None = None) -> dict[str, BioDatasetSpec]:
    """Load and validate dataset specs from a registry JSON into a ``bio_id -> spec`` map."""
    path = Path(path) if path is not None else _registry_path()
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"{path}: bio dataset registry must be a JSON array of objects.")

    known = {fld.name for fld in fields(BioDatasetSpec)}
    specs: dict[str, BioDatasetSpec] = {}
    for entry in entries:
        unknown = set(entry) - known
        if unknown:
            raise ValueError(
                f"{path}: dataset {entry.get('bio_id')!r} has unknown field(s) {sorted(unknown)}."
            )
        spec = BioDatasetSpec(**entry)
        if spec.bio_id in specs:
            raise ValueError(f"{path}: duplicate bio_id {spec.bio_id!r}.")
        specs[spec.bio_id] = spec
    return specs


@dataclass
class _Registry:
    """Lazily-loaded view over the active dataset registry."""

    specs: dict[str, BioDatasetSpec] = field(default_factory=load_specs)


#: Loaded once at import (cheap — just parses the JSON). Re-import or call
#: :func:`reload` after changing ``$TABBENCH_BIO_DATASETS`` or the JSON.
BIO_DATASETS: dict[str, BioDatasetSpec] = load_specs()


def reload() -> dict[str, BioDatasetSpec]:
    """Reload the registry from disk (e.g. after editing the JSON) and return it."""
    global BIO_DATASETS
    BIO_DATASETS = load_specs()
    return BIO_DATASETS


def is_bio_dataset(name: str) -> bool:
    """Whether ``name`` is a registered TabBench-Bio dataset id."""
    return name in BIO_DATASETS


def get_spec(bio_id: str) -> BioDatasetSpec:
    """Look up a dataset spec by its ``bio_id`` (fail-fast on unknown id)."""
    try:
        return BIO_DATASETS[bio_id]
    except KeyError:
        raise KeyError(f"Unknown bio dataset {bio_id!r}. Known: {sorted(BIO_DATASETS)}.") from None


def runnable_specs() -> list[BioDatasetSpec]:
    """All enabled + curated specs (the active benchmark)."""
    return [s for s in BIO_DATASETS.values() if s.enabled and s.is_curated]


def bio_dataset_names(problem_type: str | None = None) -> list[str]:
    """Names of enabled+curated bio datasets, optionally filtered by problem type."""
    specs = runnable_specs()
    if problem_type is not None:
        specs = [s for s in specs if s.problem_type == problem_type]
    return [s.bio_id for s in specs]
