"""Curated ChEMBL single-protein bioactivity regression endpoints.

The loader queries a pinned ChEMBL database release for measured pChEMBL values,
then uses the same molecular preparation as the TDC datasets: canonical parent
structures, median targets for exact duplicates, 2,048-bit ECFP4 fingerprints,
and Bemis--Murcko scaffold groups.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import pandas as pd

from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.bio.loaders.tdc import prepare_tdc_table

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
STATUS_URL = f"{API_BASE}/status.json"
ACTIVITY_URL = f"{API_BASE}/activity.json"
EXPECTED_CHEMBL_VERSION = "ChEMBL_37"


@dataclass(frozen=True)
class ChemblEndpoint:
    """Frozen semantics for one selected ChEMBL target."""

    preferred_name: str
    organism: str = "Homo sapiens"
    target_type: str = "SINGLE PROTEIN"


ENDPOINTS: dict[str, ChemblEndpoint] = {
    "CHEMBL206": ChemblEndpoint("Estrogen receptor"),
    "CHEMBL2034": ChemblEndpoint("Glucocorticoid receptor"),
}


def _get_json(url: str, *, params: dict[str, object] | None = None) -> dict:
    """Fetch one JSON API response with normal HTTP validation."""
    import requests

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _activities(target_id: str) -> tuple[list[dict], int]:
    """Fetch all pChEMBL-bearing activity records for one target."""
    records: list[dict] = []
    total_count: int | None = None
    url = ACTIVITY_URL
    params: dict[str, object] | None = {
        "target_chembl_id": target_id,
        "pchembl_value__isnull": "false",
        "limit": 1000,
    }
    while url:
        payload = _get_json(url, params=params)
        if "activities" not in payload or "page_meta" not in payload:
            raise ValueError(f"{target_id}: malformed ChEMBL activity response.")
        records.extend(payload["activities"])
        page_meta = payload["page_meta"]
        if total_count is None:
            total_count = int(page_meta["total_count"])
        next_page = page_meta["next"]
        url = urljoin("https://www.ebi.ac.uk", next_page) if next_page else ""
        params = None
    return records, total_count or 0


def prepare_chembl_activities(
    activities: list[dict], *, bio_id: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Filter measured activities and apply the shared molecular preparation."""
    rows: list[dict[str, object]] = []
    missing_structure = 0
    non_exact_relation = 0
    flagged_validity = 0
    assay_types: Counter[str] = Counter()

    for activity in activities:
        smiles = activity["canonical_smiles"]
        value = activity["pchembl_value"]
        relation = activity["standard_relation"]
        validity = activity["data_validity_comment"]
        assay_type = str(activity["assay_type"])
        assay_types[assay_type] += 1
        if not smiles or value is None:
            missing_structure += 1
            continue
        if relation != "=":
            non_exact_relation += 1
            continue
        if validity not in (None, ""):
            flagged_validity += 1
            continue
        rows.append({"Drug": smiles, "Y": value})

    if not rows:
        raise ValueError(f"{bio_id}: no exact, unflagged ChEMBL activities have structures.")
    table = pd.DataFrame(rows)
    X, y, groups, metadata = prepare_tdc_table(
        table,
        bio_id=bio_id,
        problem_type="regression",
    )
    y = y.rename("pchembl_value")
    metadata.update(
        {
            "api_activity_records": int(len(activities)),
            "missing_structure_or_value": int(missing_structure),
            "non_exact_relation_dropped": int(non_exact_relation),
            "flagged_validity_dropped": int(flagged_validity),
            "assay_type_counts": dict(assay_types),
        }
    )
    return X, y, groups, metadata


class ChemblLoader:
    """Fetch and prepare a pinned human single-protein ChEMBL endpoint."""

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        target_id = spec.fetch_id.strip().upper()
        if target_id not in ENDPOINTS:
            raise ValueError(
                f"{spec.bio_id}: unsupported ChEMBL target {spec.fetch_id!r}; "
                f"curated endpoints are {sorted(ENDPOINTS)}."
            )
        if spec.target != "pchembl_value":
            raise ValueError(
                f"{spec.bio_id}: ChEMBL registry target must be 'pchembl_value', "
                f"not {spec.target!r}."
            )
        if spec.problem_type is not None and spec.problem_type != "regression":
            raise ValueError(
                f"{spec.bio_id}: ChEMBL pChEMBL prediction requires regression, "
                f"not {spec.problem_type!r}."
            )

        status = _get_json(STATUS_URL)
        version = str(status["chembl_db_version"])
        if version != EXPECTED_CHEMBL_VERSION:
            raise ValueError(
                f"{spec.bio_id}: ChEMBL API serves {version!r}, but this benchmark is pinned "
                f"to {EXPECTED_CHEMBL_VERSION!r}; curate and validate a release update first."
            )

        activities, reported_total = _activities(target_id)
        X, y, groups, metadata = prepare_chembl_activities(activities, bio_id=spec.bio_id)
        endpoint = ENDPOINTS[target_id]
        metadata.update(
            {
                "target_chembl_id": target_id,
                "target_name": endpoint.preferred_name,
                "target_type": endpoint.target_type,
                "organism": endpoint.organism,
                "chembl_db_version": version,
                "api_reported_activity_count": int(reported_total),
                "split_unit": "Bemis-Murcko scaffold",
            }
        )
        source_url = f"https://www.ebi.ac.uk/chembl/explore/target/{target_id}"
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type="regression",
            license=spec.license or "CC BY-SA 3.0",
            source_url=source_url,
            citation=(
                "Zdrazil B et al. (2024), The ChEMBL Database in 2023: a drug discovery "
                "platform spanning multiple bioactivity data types and time periods, "
                "Nucleic Acids Research."
            ),
            metadata=metadata,
            groups=groups,
        )
