"""Human-gut metagenomics loader (MetAML marker table), fetch-by-disease.

``spec.fetch_id`` is a disease code from Pasolli's MetAML marker matrix (e.g.
``"cirrhosis"``). The loader downloads the marker-presence table once (cached under
``<bio-cache>/metagenomics_raw``) and restricts cases and controls to the original
cirrhosis study. Features are unfiltered 0/1 microbial marker calls. Prevalence
filtering belongs to the benchmark's training-only feature preparation.

Downloads use the Python standard library.
"""

from __future__ import annotations

import bz2
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

import numpy as np
import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec

MARKER_URL = (
    "https://raw.githubusercontent.com/segatalab/metaml/master/data/marker_presence.txt.bz2"
)
MIN_PREVALENCE = 0.10  # drop markers present in <10% of samples
HEALTHY = frozenset({"n", "nd", "n_relative"})  # disease codes counted as healthy
COHORT = "Quin_gut_liver_cirrhosis"  # original spelling in MetAML
COHORT_VERSION = "cirrhosis-matched-stool-v1"


class MetagenomicsLoader:
    """Fetch the MetAML gut-marker matrix and frame healthy-vs-<disease> binary tasks."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = default_bio_cache_dir() / "metagenomics_raw"
        self.cache_dir = Path(cache_dir)

    def _marker_matrix(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Read unfiltered markers and metadata for the original study's stool samples."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw = self.cache_dir / "marker_presence.txt.bz2"
        if not raw.exists():
            with urlopen(MARKER_URL, timeout=180) as response:
                raw.write_bytes(response.read())

        metadata, markers, names = {}, [], []
        with bz2.open(raw, "rt") as fh:
            for line in fh:
                name, _, rest = line.partition("\t")
                if not name.startswith("gi|"):
                    assert not names, f"Unexpected metadata after markers: {name}"
                    metadata[name] = rest.rstrip("\r\n").split("\t")
                    continue
                if not names:
                    frame = pd.DataFrame(metadata)
                    keep = frame["dataset_name"].eq(COHORT).to_numpy()
                    selected = frame.loc[keep].reset_index(drop=True)
                    assert selected["bodysite"].eq("stool").all()
                    assert selected["disease"].value_counts().to_dict() == {
                        "cirrhosis": 118,
                        "n": 114,
                    }
                    assert selected["sampleID"].is_unique and selected["subjectID"].is_unique
                    assert not selected["subjectID"].isin(["", "nd", "na", "nan"]).any()
                values = np.fromstring(rest, sep="\t", dtype=np.int8)
                assert len(values) == len(frame), name
                assert np.isin(values, [0, 1]).all(), name
                markers.append(values[keep])
                names.append(name)
        assert names and len(names) == len(set(names))
        X = pd.DataFrame(np.asarray(markers).T, columns=names)
        return X, selected, hashlib.sha256(raw.read_bytes()).hexdigest()

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        """Assemble the healthy-vs-<disease> binary task named by ``spec.fetch_id``."""
        target_disease = spec.fetch_id.strip().lower()
        assert target_disease == "cirrhosis", (
            "Curate a matched source cohort before adding another disease"
        )
        X, metadata, source_digest = self._marker_matrix()
        y = pd.Series(
            ["healthy" if d == "n" else "disease" for d in metadata["disease"]],
            name="target",
        )
        groups = metadata["subjectID"].map(lambda subject: f"{COHORT}:{subject}").rename("group")

        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type=spec.problem_type or "binary",
            license=spec.license or "CC-BY-4.0 (Pasolli et al. 2016, MetAML)",
            source_url="https://github.com/segatalab/metaml",
            citation=(
                "Pasolli E, et al. (2016) Machine Learning Meta-analysis of Large "
                "Metagenomic Datasets: Tools and Biological Insights. PLoS Comput Biol 12(7)."
            ),
            metadata={
                "disease": target_disease,
                "n_markers": int(X.shape[1]),
                "min_prevalence": MIN_PREVALENCE,
                "prevalence_fitted_on": "training_rows_only",
                "cohort_version": COHORT_VERSION,
                "cohort": COHORT,
                "body_site": "stool",
                "sample_ids": metadata["sampleID"].tolist(),
                "subject_ids": metadata["subjectID"].tolist(),
                "source_sha256": source_digest,
            },
            groups=groups,
        )
