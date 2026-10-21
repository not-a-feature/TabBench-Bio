"""Curated Therapeutics Data Commons molecular endpoints.

The loader fetches the original TDC tables from their stable Harvard Dataverse
file records and converts SMILES into 2,048-bit ECFP4 fingerprints.  Molecules are
canonicalised before exact-structure deduplication.  Bemis--Murcko scaffolds are
returned as groups, preventing the same structural family from crossing benchmark
folds.

Only the endpoints in :data:`ENDPOINTS` are accepted.  This is deliberate: a
published benchmark needs frozen file identities and task semantics rather than a
moving catalogue or heuristic target selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.io_utils import download_file

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
MAX_INVALID_FRACTION = 0.01
MAX_CONFLICT_FRACTION = 0.01


ENDPOINTS = {
    "bbb_martins": "binary",
    "ames": "binary",
    "cyp3a4_veith": "binary",
    "solubility_aqsoldb": "regression",
    "lipophilicity_astrazeneca": "regression",
    "ld50_zhu": "regression",
}


def _canonicalise(smiles: str) -> tuple[str, object] | None:
    """Return a canonical parent structure and molecule, or ``None`` if invalid."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        parent = rdMolStandardize.FragmentParent(molecule)
        if parent is None or parent.GetNumAtoms() == 0:
            return None
        canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        # A SMILES round trip is an inexpensive strict sanitisation gate and also
        # guarantees that fingerprint/scaffold ring information is initialised.
        sanitised = Chem.MolFromSmiles(canonical)
    except (ValueError, RuntimeError):
        return None
    if sanitised is None:
        return None
    return canonical, sanitised


def _scaffold(molecule: object, canonical_smiles: str) -> str:
    """Return a leakage-safe scaffold group, including a safe acyclic fallback."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
    # An empty Murcko scaffold is normal for acyclic compounds.  Putting every such
    # molecule in one group would make grouped splitting unusable, so exact parent
    # structures become their independence units.
    return f"murcko:{scaffold}" if scaffold else f"acyclic:{canonical_smiles}"


def _fingerprints(molecules: list[object]) -> np.ndarray:
    """Calculate dense 2,048-bit Morgan/ECFP4 vectors with one generator."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS,
        fpSize=FINGERPRINT_BITS,
    )
    matrix = np.zeros((len(molecules), FINGERPRINT_BITS), dtype=np.uint8)
    for row, molecule in enumerate(molecules):
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[row])
    return matrix


def prepare_tdc_table(
    table: pd.DataFrame,
    *,
    bio_id: str,
    problem_type: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Validate, canonicalise, deduplicate, fingerprint, and group a TDC table."""
    smiles_column = "Drug" if "Drug" in table.columns else "X" if "X" in table.columns else None
    missing = ({"Y"} - set(table.columns)) | ({"Drug or X"} if smiles_column is None else set())
    if missing:
        raise ValueError(f"{bio_id}: TDC table is missing required column(s) {sorted(missing)}.")

    frame = table.loc[:, [smiles_column, "Y"]].copy()
    frame.columns = ["Drug", "Y"]
    frame["Y"] = pd.to_numeric(frame["Y"], errors="coerce")
    missing_target = int(frame["Y"].isna().sum())
    if missing_target:
        raise ValueError(f"{bio_id}: TDC table has {missing_target} missing/non-numeric target(s).")

    from rdkit import rdBase

    records: list[tuple[str, object, float]] = []
    # Invalid structures are counted below; RDKit's per-atom parser diagnostics are
    # noisy for large public collections and do not add information to that report.
    with rdBase.BlockLogs():
        for smiles, target in frame.itertuples(index=False, name=None):
            parsed = _canonicalise(smiles)
            if parsed is not None:
                canonical, molecule = parsed
                records.append((canonical, molecule, float(target)))

    invalid = len(frame) - len(records)
    invalid_fraction = invalid / len(frame) if len(frame) else 1.0
    if invalid_fraction > MAX_INVALID_FRACTION:
        raise ValueError(
            f"{bio_id}: {invalid}/{len(frame)} SMILES are invalid ({invalid_fraction:.2%}); "
            f"quality limit is {MAX_INVALID_FRACTION:.0%}."
        )
    if not records:
        raise ValueError(f"{bio_id}: no valid molecules remain.")

    valid = pd.DataFrame(records, columns=["canonical", "molecule", "target"])
    duplicate_rows = len(valid) - valid["canonical"].nunique()
    conflicting_molecules = 0
    cleaned: list[tuple[str, object, float]] = []
    for canonical, group in valid.groupby("canonical", sort=True):
        targets = group["target"]
        if problem_type in {"binary", "multiclass"}:
            if targets.nunique() != 1:
                conflicting_molecules += 1
                continue
            target = float(targets.iloc[0])
        else:
            target = float(targets.median())
        cleaned.append((canonical, group["molecule"].iloc[0], target))

    conflict_fraction = conflicting_molecules / valid["canonical"].nunique()
    if conflict_fraction > MAX_CONFLICT_FRACTION:
        raise ValueError(
            f"{bio_id}: {conflicting_molecules} canonical molecules have conflicting labels "
            f"({conflict_fraction:.2%}); quality limit is {MAX_CONFLICT_FRACTION:.0%}."
        )
    if not cleaned:
        raise ValueError(f"{bio_id}: no molecules remain after deduplication.")

    matrix = _fingerprints([molecule for _, molecule, _ in cleaned])
    X = pd.DataFrame(matrix, columns=[f"ecfp4_{index}" for index in range(FINGERPRINT_BITS)])
    y_values = [target for _, _, target in cleaned]
    if problem_type in {"binary", "multiclass"}:
        y_values = [int(target) for target in y_values]
    y = pd.Series(y_values, name="Y")
    groups = pd.Series(
        [_scaffold(molecule, canonical) for canonical, molecule, _ in cleaned],
        name="murcko_scaffold",
    )

    metadata = {
        "raw_rows": int(len(frame)),
        "valid_rows": int(len(valid)),
        "unique_molecules": int(len(cleaned)),
        "invalid_smiles": int(invalid),
        "duplicate_rows": int(duplicate_rows),
        "conflicting_molecules_dropped": int(conflicting_molecules),
        "n_scaffolds": int(groups.nunique()),
        "fingerprint": "Morgan/ECFP4",
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "n_features": FINGERPRINT_BITS,
    }
    return X, y, groups, metadata


class TdcLoader:
    """Fetch and prepare a frozen TDC molecular endpoint."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or (default_bio_cache_dir() / "tdc_raw"))

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        endpoint_name = spec.fetch_id.strip().lower()
        if endpoint_name not in ENDPOINTS:
            raise ValueError(
                f"{spec.bio_id}: unsupported TDC endpoint {spec.fetch_id!r}; "
                f"curated endpoints are {sorted(ENDPOINTS)}."
            )
        problem_type = ENDPOINTS[endpoint_name]
        if spec.problem_type is not None and spec.problem_type != problem_type:
            raise ValueError(
                f"{spec.bio_id}: registry problem_type={spec.problem_type!r} disagrees with "
                f"curated TDC endpoint type {problem_type!r}."
            )

        assert spec.download_url and spec.download_sha256, (
            f"{spec.bio_id}: missing download configuration"
        )
        path = download_file(
            spec.download_url,
            self.cache_dir / f"{endpoint_name}.tab",
            checksum=spec.download_sha256,
            timeout=180,
        )
        table = pd.read_csv(path, sep="\t")
        X, y, groups, metadata = prepare_tdc_table(
            table,
            bio_id=spec.bio_id,
            problem_type=problem_type,
        )
        metadata.update(
            {
                "endpoint": endpoint_name,
                "dataverse_file_id": int(spec.source_url.rsplit("/", 1)[1]),
                "source_sha256": spec.download_sha256,
            }
        )
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=spec.license or "TDC distribution; underlying source terms apply",
            source_url=spec.source_url,
            citation=spec.citation,
            metadata=metadata,
            groups=groups,
        )
