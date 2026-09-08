#!/usr/bin/env python
"""Build two compact, leakage-aware protein-language-model embedding datasets.

The generated Parquet tables are consumed by TabBench-Bio's existing ``local`` loader.
They contain 1,280 numeric ESM-2 feature columns, one target, and one biological group
column; raw amino-acid sequences and accessions are deliberately excluded.

Datasets
--------
``deeploc2_fungi``
    Fungal proteins from the official DeepLoc 2.0 Swiss-Prot train/validation table.
    Target: membrane associated (binary). Groups: the five published homology
    partitions, constructed to limit cross-partition sequence identity.

``beta_lactamase``
    The PEER/TorchDrug TEM-1 beta-lactamase single-mutant activity task. Target:
    scaled experimental activity (regression). Groups: mutated residue position, so
    alternative amino acids at one site never cross folds.

Embeddings are residue-mean-pooled ``facebook/esm2_t33_650M_UR50D`` representations
at the pinned Hugging Face revision below. Sequences longer than the model context are
split into non-overlapping 1,022-residue chunks; chunk sums are combined with residue
counts, yielding one length-weighted protein vector without discarding residues.

The source artifacts are downloaded from their official hosts and verified against
pinned checksums. Generated tables remain local/ignored because the upstream dataset
licenses do not explicitly grant us permission to re-host derived copies.

Examples
--------
Audit sources without loading ESM-2::

    python scripts/generate_protein_embeddings.py --audit-only

Generate both benchmark tables on a GPU::

    python scripts/generate_protein_embeddings.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import shutil
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("protein_embeddings")

MODEL_ID = "facebook/esm2_t33_650M_UR50D"
MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
EMBEDDING_DIM = 1280
CHUNK_RESIDUES = 1022

DEEPLOC_URL = (
    "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/"
    "Swissprot_Train_Validation_dataset.csv"
)
DEEPLOC_SHA256 = "29a07b293fed2994a966b70bdcd6bacc59915b8b01fa200cb2b07d8db18384a2"

BETA_URL = (
    "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/"
    "peerdata/beta_lactamase.tar.gz"
)
BETA_MD5 = "65766a3969cc0e94b101d4063d204ba4"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "protein_embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "tabbench_bio" / "bio" / "data" / "local"


@dataclass(frozen=True)
class PreparedDataset:
    """Sequences, targets, and leakage-prevention groups ready for embedding."""

    key: str
    output_name: str
    sequences: list[str]
    target: np.ndarray
    target_name: str
    groups: np.ndarray
    group_name: str
    problem_type: str
    source_metadata: dict[str, Any]


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path, *, algorithm: str, expected: str) -> Path:
    """Download once, then verify the immutable source identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        LOG.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)

    observed = _file_digest(path, algorithm)
    if observed.lower() != expected.lower():
        raise ValueError(
            f"Source identity check failed for {path}: "
            f"expected {algorithm}={expected}, observed {observed}."
        )
    return path


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe path {member.name!r} in {archive}.")
        tar.extractall(destination, filter="data")


def _prepare_deeploc2_fungi(csv_path: Path) -> PreparedDataset:
    """Select the complete fungal DeepLoc 2.0 subset and preserve its five partitions."""
    frame = pd.read_csv(csv_path)
    required = {"ACC", "Kingdom", "Partition", "Membrane", "Sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DeepLoc 2.0 source is missing columns: {sorted(missing)}")
    if len(frame) != 28_303:
        raise ValueError(f"DeepLoc 2.0 source changed: expected 28303 rows, found {len(frame)}.")

    frame = frame.loc[
        frame["Kingdom"].eq("Fungi"), ["ACC", "Partition", "Membrane", "Sequence"]
    ].copy()
    frame = frame.sort_values("ACC", kind="stable").reset_index(drop=True)
    frame["Sequence"] = frame["Sequence"].astype(str).str.strip().str.upper()
    frame["Membrane"] = frame["Membrane"].astype(np.int8)
    frame["Partition"] = frame["Partition"].astype(np.int8)

    if len(frame) != 5_841:
        raise ValueError(f"DeepLoc fungal subset changed: expected 5841 rows, found {len(frame)}.")
    if frame[["ACC", "Partition", "Membrane", "Sequence"]].isna().any().any():
        raise ValueError("DeepLoc fungal subset contains missing values.")
    if frame["ACC"].duplicated().any() or frame["Sequence"].duplicated().any():
        raise ValueError("DeepLoc fungal subset contains duplicate accessions or sequences.")
    if set(frame["Partition"]) != set(range(5)):
        raise ValueError("DeepLoc fungal subset no longer contains all five homology partitions.")
    if Counter(frame["Membrane"]) != Counter({0: 4_535, 1: 1_306}):
        raise ValueError("DeepLoc fungal membrane-label counts changed.")

    partition_counts = pd.crosstab(frame["Partition"], frame["Membrane"])
    if partition_counts.shape != (5, 2) or (partition_counts < 30).any().any():
        raise ValueError("DeepLoc partitions do not each contain enough examples of both classes.")

    return PreparedDataset(
        key="deeploc2_fungi",
        output_name="deeploc2_fungi_esm2_t33.parquet",
        sequences=frame["Sequence"].tolist(),
        target=frame["Membrane"].to_numpy(),
        target_name="membrane",
        groups=frame["Partition"].to_numpy(),
        group_name="homology_partition",
        problem_type="binary",
        source_metadata={
            "source_url": DEEPLOC_URL,
            "source_sha256": DEEPLOC_SHA256,
            "source_rows": 28_303,
            "selection": "Kingdom == Fungi (complete subset; no target-dependent sampling)",
            "class_counts": {"soluble": 4_535, "membrane": 1_306},
            "n_groups": 5,
        },
    )


def _read_lmdb_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover - exercised on curation machines only
        raise RuntimeError(
            "Reading the PEER source requires lmdb. Install the bio extra or `pip install lmdb`."
        ) from exc

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as transaction:
        raw_count = transaction.get(b"num_examples")
        if raw_count is None:
            raise ValueError(f"{path} has no num_examples key.")
        count = int(pickle.loads(raw_count))
        rows = []
        for index in range(count):
            value = transaction.get(str(index).encode())
            if value is None:
                raise ValueError(f"{path} is missing row {index}.")
            rows.append(pickle.loads(value))
    env.close()
    return rows


def _mutation_positions(sequences: list[str]) -> tuple[str, np.ndarray]:
    """Infer the TEM-1 consensus and require exactly one mutation in every row."""
    lengths = {len(sequence) for sequence in sequences}
    if lengths != {286}:
        raise ValueError(f"Expected 286-residue TEM-1 variants, found lengths {sorted(lengths)}.")

    consensus = "".join(
        Counter(sequence[position] for sequence in sequences).most_common(1)[0][0]
        for position in range(286)
    )
    positions = []
    for row_index, sequence in enumerate(sequences):
        mismatches = [
            position
            for position, (observed, reference) in enumerate(zip(sequence, consensus))
            if observed != reference
        ]
        if len(mismatches) != 1:
            raise ValueError(
                f"TEM-1 row {row_index} has {len(mismatches)} differences from the consensus; "
                "expected exactly one."
            )
        positions.append(mismatches[0] + 1)
    return consensus, np.asarray(positions, dtype=np.int16)


def _prepare_beta_lactamase(extracted_dir: Path) -> PreparedDataset:
    """Read the three official PEER splits and derive mutation-position groups."""
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        path = extracted_dir / "beta_lactamase" / f"beta_lactamase_{split}.lmdb"
        split_rows = _read_lmdb_rows(path)
        split_counts[split] = len(split_rows)
        rows.extend(split_rows)
    if split_counts != {"train": 4_158, "valid": 520, "test": 520}:
        raise ValueError(f"PEER beta-lactamase split counts changed: {split_counts}.")

    records = []
    for row in rows:
        if not {"primary", "scaled_effect1"} <= set(row):
            raise ValueError("PEER beta-lactamase row is missing sequence or target.")
        target = row["scaled_effect1"]
        if isinstance(target, np.ndarray):
            target = target.item()
        records.append((str(row["primary"]).strip().upper(), float(target)))
    records.sort(key=lambda item: item[0])

    sequences = [sequence for sequence, _ in records]
    target = np.asarray([value for _, value in records], dtype=np.float32)
    if len(sequences) != 5_198 or len(set(sequences)) != 5_198:
        raise ValueError("PEER beta-lactamase must contain 5198 unique single-mutant sequences.")
    if not np.isfinite(target).all() or np.unique(target).size < 20:
        raise ValueError("PEER beta-lactamase target is invalid or effectively constant.")

    consensus, positions = _mutation_positions(sequences)
    position_counts = Counter(positions.tolist())
    if set(position_counts) != set(range(1, 287)):
        raise ValueError("PEER beta-lactamase no longer covers every TEM-1 residue position.")
    if min(position_counts.values()) < 5:
        raise ValueError("A TEM-1 mutation-position group has fewer than five variants.")

    return PreparedDataset(
        key="beta_lactamase",
        output_name="peer_beta_lactamase_esm2_t33.parquet",
        sequences=sequences,
        target=target,
        target_name="activity",
        groups=positions,
        group_name="mutation_position",
        problem_type="regression",
        source_metadata={
            "source_url": BETA_URL,
            "source_md5": BETA_MD5,
            "official_split_counts": split_counts,
            "n_groups": 286,
            "consensus_sha256": hashlib.sha256(consensus.encode()).hexdigest(),
            "group_definition": "1-based position of the sole amino-acid substitution",
        },
    )


def _load_sources(cache_dir: Path, selected: set[str]) -> list[PreparedDataset]:
    prepared = []
    if "deeploc2_fungi" in selected:
        path = _download(
            DEEPLOC_URL,
            cache_dir / "Swissprot_Train_Validation_dataset.csv",
            algorithm="sha256",
            expected=DEEPLOC_SHA256,
        )
        prepared.append(_prepare_deeploc2_fungi(path))

    if "beta_lactamase" in selected:
        archive = _download(
            BETA_URL,
            cache_dir / "beta_lactamase.tar.gz",
            algorithm="md5",
            expected=BETA_MD5,
        )
        extracted = cache_dir / "peer_beta"
        expected_lmdb = extracted / "beta_lactamase" / "beta_lactamase_train.lmdb"
        if not expected_lmdb.exists():
            _safe_extract_tar(archive, extracted)
        prepared.append(_prepare_beta_lactamase(extracted))
    return prepared


def _chunk_sequences(
    sequences: list[str], chunk_residues: int = CHUNK_RESIDUES
) -> list[tuple[int, str]]:
    """Return ``(source-row, chunk)`` pairs without dropping any residues."""
    chunks = []
    for row_index, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError(f"Sequence row {row_index} is empty.")
        chunks.extend(
            (row_index, sequence[start : start + chunk_residues])
            for start in range(0, len(sequence), chunk_residues)
        )
    return chunks


def _embed_sequences(
    sequences: list[str], *, device: str, batch_size: int, chunk_residues: int
) -> np.ndarray:
    """Compute length-weighted residue-mean ESM-2 embeddings."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on embedding machines only
        raise RuntimeError(
            "Embedding requires torch and transformers. Install the `models` extra."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if chunk_residues + special_tokens > 1024:
        raise ValueError(
            f"chunk_residues={chunk_residues} plus {special_tokens} special tokens exceeds 1024."
        )

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    LOG.info("Loading %s at revision %s", MODEL_ID, MODEL_REVISION)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    chunks = _chunk_sequences(sequences, chunk_residues=chunk_residues)
    # Similar lengths together reduce padding. row_index keeps aggregation deterministic.
    chunks.sort(key=lambda item: len(item[1]), reverse=True)
    sums = np.zeros((len(sequences), EMBEDDING_DIM), dtype=np.float64)
    counts = np.zeros(len(sequences), dtype=np.int64)

    LOG.info(
        "Embedding %d proteins as %d chunks (batch_size=%d, device=%s)",
        len(sequences),
        len(chunks),
        batch_size,
        device,
    )
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            row_indices = [row_index for row_index, _ in batch]
            chunk_text = [chunk for _, chunk in batch]
            encoded = tokenizer(
                chunk_text,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_mask = encoded.pop("special_tokens_mask").to(device)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            residue_mask = (encoded["attention_mask"].bool() & ~special_mask.bool()).unsqueeze(-1)
            chunk_sums = (hidden * residue_mask).sum(dim=1).float().cpu().numpy()
            chunk_counts = residue_mask.sum(dim=1).squeeze(-1).cpu().numpy()

            for row_index, vector, residue_count in zip(
                row_indices, chunk_sums, chunk_counts, strict=True
            ):
                sums[row_index] += vector
                counts[row_index] += int(residue_count)
            done = min(start + batch_size, len(chunks))
            if done == len(chunks) or done % max(batch_size * 100, 1) == 0:
                LOG.info("  embedded %d/%d chunks", done, len(chunks))

    expected_counts = np.asarray([len(sequence) for sequence in sequences])
    if not np.array_equal(counts, expected_counts):
        raise ValueError("ESM-2 residue accounting mismatch; one or more residues were dropped.")
    embeddings = (sums / counts[:, None]).astype(np.float32)
    if embeddings.shape != (len(sequences), EMBEDDING_DIM):
        raise ValueError(f"Unexpected embedding shape {embeddings.shape}.")
    if not np.isfinite(embeddings).all():
        raise ValueError("ESM-2 produced missing or non-finite values.")
    return embeddings


def _validate_embedding_table(dataset: PreparedDataset, embeddings: np.ndarray) -> dict[str, Any]:
    std = embeddings.std(axis=0, dtype=np.float64)
    nonconstant = int(np.count_nonzero(std > 1e-12))
    if nonconstant < 1_000:
        raise ValueError(
            f"{dataset.key}: only {nonconstant}/{embeddings.shape[1]} nonconstant features."
        )
    if len(np.unique(dataset.groups)) < 5:
        raise ValueError(f"{dataset.key}: fewer than five biological groups.")
    if dataset.problem_type == "binary":
        counts = Counter(dataset.target.tolist())
        if set(counts) != {0, 1} or min(counts.values()) < 30:
            raise ValueError(f"{dataset.key}: invalid binary class counts {counts}.")
    elif np.unique(dataset.target).size < 20:
        raise ValueError(f"{dataset.key}: regression target has fewer than 20 values.")

    return {
        "n_samples": len(dataset.sequences),
        "n_features": embeddings.shape[1],
        "nonconstant_features": nonconstant,
        "n_groups": int(np.unique(dataset.groups).size),
        "target_unique": int(np.unique(dataset.target).size),
        "sequence_length": {
            "min": min(map(len, dataset.sequences)),
            "median": float(np.median([len(sequence) for sequence in dataset.sequences])),
            "max": max(map(len, dataset.sequences)),
        },
    }


def _write_dataset(
    dataset: PreparedDataset, embeddings: np.ndarray, output_dir: Path, *, force: bool
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / dataset.output_name
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it.")

    quality = _validate_embedding_table(dataset, embeddings)
    feature_names = [f"esm2_{index:04d}" for index in range(EMBEDDING_DIM)]
    frame = pd.DataFrame(embeddings, columns=feature_names)
    frame[dataset.target_name] = dataset.target
    frame[dataset.group_name] = dataset.groups

    temporary = output_path.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    manifest = {
        "dataset": dataset.key,
        "output_file": output_path.name,
        "output_sha256": _file_digest(output_path, "sha256"),
        "output_bytes": output_path.stat().st_size,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pooling": "length-weighted residue mean over non-overlapping chunks",
        "chunk_residues": CHUNK_RESIDUES,
        "quality": quality,
        "source": dataset.source_metadata,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s (%0.1f MiB)", output_path, output_path.stat().st_size / 2**20)
    LOG.info("Quality: %s", quality)
    return output_path


def _audit(dataset: PreparedDataset) -> dict[str, Any]:
    summary = {
        "dataset": dataset.key,
        "n_samples": len(dataset.sequences),
        "n_groups": int(np.unique(dataset.groups).size),
        "target_unique": int(np.unique(dataset.target).size),
        "target_min": float(np.min(dataset.target)),
        "target_max": float(np.max(dataset.target)),
        "sequence_min": min(map(len, dataset.sequences)),
        "sequence_median": float(np.median([len(sequence) for sequence in dataset.sequences])),
        "sequence_max": max(map(len, dataset.sequences)),
        "source": dataset.source_metadata,
    }
    if dataset.problem_type == "binary":
        summary["class_counts"] = dict(sorted(Counter(dataset.target.tolist()).items()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("deeploc2_fungi", "beta_lactamase"),
        default=("deeploc2_fungi", "beta_lactamase"),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-residues", type=int, default=CHUNK_RESIDUES)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    datasets = _load_sources(args.cache_dir, set(args.datasets))
    for dataset in datasets:
        LOG.info("Source audit:\n%s", json.dumps(_audit(dataset), indent=2, sort_keys=True))
        if args.audit_only:
            continue
        embeddings = _embed_sequences(
            dataset.sequences,
            device=args.device,
            batch_size=args.batch_size,
            chunk_residues=args.chunk_residues,
        )
        _write_dataset(dataset, embeddings, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
