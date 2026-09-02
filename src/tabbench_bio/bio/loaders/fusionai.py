"""Nucleotide Transformer embeddings for the FusionAI breakpoint task.

The immutable Zenodo record stores every benchmarked foundation model in one
3.57 GB ZIP archive.  This loader uses HTTP byte ranges and ZIP member CRCs to
materialize only the four middle-NT matrices and their two target files.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.loaders.base import BioRawDataset

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


ZENODO_RECORD_ID = "18713246"
ZENODO_RECORD = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/embeddings.zip/content"
ARCHIVE_BYTES = 3_574_557_943
RANGE_BLOCK_BYTES = 32 * 1024 * 1024
NT_WIDTH_PER_PARTNER = 1_024
N_TRAIN = 36_302
N_TEST = 15_557
FETCH_ID = "18713246/nt-middle"
SOURCE_URL = "https://doi.org/10.5281/zenodo.18713246"
CITATION = (
    "Krupicka R et al. (2026), Benchmarking genomic foundation models for binary "
    "classification of gene fusion breakpoints from DNA sequences, BioData Mining, "
    "doi:10.1186/s13040-026-00553-1."
)


@dataclass(frozen=True)
class ArchiveMember:
    """Pinned identity of one required file in the public ZIP archive."""

    basename: str
    n_bytes: int
    crc32: int


# Filled from the immutable Zenodo record.  Both ZIP metadata and extracted files
# are checked against these identities before a dataset is returned.
MEMBERS: tuple[ArchiveMember, ...] = (
    ArchiveMember("nt_train_seq1.csv", 392_030_177, 0xBE380B9D),
    ArchiveMember("nt_train_seq2.csv", 390_048_003, 0x68CA59A9),
    ArchiveMember("nt_test_seq1.csv", 167_980_476, 0xAFDED191),
    ArchiveMember("nt_test_seq2.csv", 167_195_401, 0x26320FC9),
    ArchiveMember("fusionai_train_target.csv", 72_604, 0x50665CCD),
    ArchiveMember("fusionai_test_target.csv", 31_114, 0x1BFC0AAF),
)


class _HttpRangeReader(io.RawIOBase):
    """Seekable, bounded-memory view of a remote file supporting byte ranges."""

    def __init__(self, url: str, *, expected_bytes: int) -> None:
        import requests

        super().__init__()
        self.url = url
        self.expected_bytes = expected_bytes
        self.position = 0
        self.session = requests.Session()
        self.blocks: OrderedDict[int, bytes] = OrderedDict()

        response = self.session.head(url, timeout=180)
        response.raise_for_status()
        actual_bytes = int(response.headers["Content-Length"])
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"FusionAI embeddings archive has {actual_bytes} bytes; expected "
                f"the pinned Zenodo record with {expected_bytes} bytes."
            )

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.expected_bytes + offset
        else:
            raise ValueError(f"Unsupported seek whence {whence}.")
        if not 0 <= position <= self.expected_bytes:
            raise ValueError(f"Remote ZIP seek {position} is outside [0, {self.expected_bytes}].")
        self.position = position
        return position

    def _block(self, block_index: int) -> bytes:
        if block_index in self.blocks:
            block = self.blocks.pop(block_index)
            self.blocks[block_index] = block
            return block

        start = block_index * RANGE_BLOCK_BYTES
        end = min(start + RANGE_BLOCK_BYTES, self.expected_bytes) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=300,
        )
        if response.status_code != 206:
            raise ValueError(
                f"Zenodo ignored ZIP byte range {start}-{end} "
                f"(HTTP {response.status_code}); refusing to download the full archive."
            )
        expected_range = f"bytes {start}-{end}/{self.expected_bytes}"
        if response.headers["Content-Range"] != expected_range:
            raise ValueError(
                f"Zenodo returned Content-Range {response.headers['Content-Range']!r}; "
                f"expected {expected_range!r}."
            )
        block = response.content
        if len(block) != end - start + 1:
            raise ValueError(f"Zenodo byte range {start}-{end} returned {len(block)} bytes.")
        self.blocks[block_index] = block
        while len(self.blocks) > 2:
            self.blocks.popitem(last=False)
        return block

    def read(self, size: int = -1) -> bytes:
        if self.position == self.expected_bytes:
            return b""
        if size is None or size < 0:
            size = self.expected_bytes - self.position
        size = min(size, self.expected_bytes - self.position)
        output = bytearray()
        while size:
            block_index = self.position // RANGE_BLOCK_BYTES
            within_block = self.position % RANGE_BLOCK_BYTES
            block = self._block(block_index)
            take = min(size, len(block) - within_block)
            output.extend(block[within_block : within_block + take])
            self.position += take
            size -= take
        return bytes(output)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
            self.blocks.clear()
        super().close()


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _validate_local(path: Path, member: ArchiveMember) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != member.n_bytes or _crc32(path) != member.crc32:
        raise ValueError(
            f"{path.name}: cached FusionAI member failed its pinned size/CRC check. "
            "Remove only this file and retry."
        )
    return True


def _extract_required_members(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {member.basename: cache_dir / member.basename for member in MEMBERS}
    missing = [member for member in MEMBERS if not _validate_local(paths[member.basename], member)]
    if not missing:
        return paths

    with _HttpRangeReader(ARCHIVE_URL, expected_bytes=ARCHIVE_BYTES) as remote:
        with zipfile.ZipFile(remote) as archive:
            by_basename: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                basename = PurePosixPath(info.filename).name
                if basename in paths:
                    if basename in by_basename:
                        raise ValueError(
                            f"FusionAI archive contains duplicate member {basename!r}."
                        )
                    by_basename[basename] = info
            absent = sorted(set(paths) - set(by_basename))
            if absent:
                raise ValueError(f"FusionAI archive is missing required member(s): {absent}.")

            for member in missing:
                info = by_basename[member.basename]
                if info.file_size != member.n_bytes or info.CRC != member.crc32:
                    raise ValueError(
                        f"{member.basename}: Zenodo ZIP identity changed "
                        f"(bytes={info.file_size}, crc32={info.CRC:08x}); expected "
                        f"bytes={member.n_bytes}, crc32={member.crc32:08x}."
                    )
                destination = paths[member.basename]
                partial = destination.with_suffix(destination.suffix + ".part")
                try:
                    with archive.open(info) as source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    partial.replace(destination)
                finally:
                    if partial.exists():
                        partial.unlink()
                if not _validate_local(destination, member):
                    raise AssertionError(f"{member.basename}: extracted member validation failed.")
    return paths


def _read_embeddings(path: Path, *, expected_rows: int) -> np.ndarray:
    frame = pd.read_csv(path, header=None, dtype=np.float32)
    if frame.shape != (expected_rows, NT_WIDTH_PER_PARTNER):
        raise ValueError(
            f"{path.name}: found shape {frame.shape}; expected "
            f"({expected_rows}, {NT_WIDTH_PER_PARTNER})."
        )
    matrix = frame.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path.name}: NT embeddings contain missing or non-finite values.")
    return matrix


def _read_targets(path: Path, *, expected_rows: int, expected_counts: dict[str, int]) -> pd.Series:
    frame = pd.read_csv(path, header=None, dtype="string")
    if frame.shape != (expected_rows, 1):
        raise ValueError(f"{path.name}: found shape {frame.shape}; expected ({expected_rows}, 1).")
    target = frame.iloc[:, 0]
    if target.isna().any() or target.nunique() != 2:
        raise ValueError(f"{path.name}: expected a complete binary FusionAI target.")
    counts = {str(label): int(count) for label, count in target.value_counts().items()}
    if counts != expected_counts:
        raise ValueError(
            f"{path.name}: class counts changed to {counts}; expected {expected_counts}."
        )
    return target


def _find(parent: np.ndarray, row: int) -> int:
    while parent[row] != row:
        parent[row] = parent[parent[row]]
        row = int(parent[row])
    return row


def _union(parent: np.ndarray, left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def exact_partner_components(matrix: np.ndarray) -> pd.Series:
    """Group rows connected by an identical embedding for either fusion partner."""
    if matrix.ndim != 2 or matrix.shape[1] != 2 * NT_WIDTH_PER_PARTNER:
        raise ValueError(
            f"FusionAI NT matrix must have {2 * NT_WIDTH_PER_PARTNER} columns; "
            f"found {matrix.shape}."
        )
    parent = np.arange(matrix.shape[0], dtype=np.int64)
    seen: dict[bytes, tuple[int, int]] = {}
    for start in (0, NT_WIDTH_PER_PARTNER):
        stop = start + NT_WIDTH_PER_PARTNER
        for row_index, row in enumerate(matrix[:, start:stop]):
            digest = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
            if digest in seen:
                previous_row, previous_start = seen[digest]
                previous = matrix[
                    previous_row,
                    previous_start : previous_start + NT_WIDTH_PER_PARTNER,
                ]
                if not np.array_equal(row, previous):
                    raise ValueError("BLAKE2b collision while grouping FusionAI NT partners.")
                _union(parent, row_index, previous_row)
            else:
                seen[digest] = (row_index, start)
    roots = np.asarray([_find(parent, row) for row in range(len(parent))])
    groups = pd.Series(pd.factorize(roots, sort=True)[0], name="exact_partner_component")
    return groups


class FusionAiNtLoader:
    """Fetch the published middle-token NT-v2 FusionAI embedding task."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or (default_bio_cache_dir() / "fusionai_raw"))

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        if spec.fetch_id != FETCH_ID:
            raise ValueError(
                f"{spec.bio_id}: unsupported FusionAI representation {spec.fetch_id!r}; "
                f"the curated representation is {FETCH_ID!r}."
            )
        if spec.target != "fusion" or spec.problem_type != "binary":
            raise ValueError(
                f"{spec.bio_id}: FusionAI-NT requires target='fusion' and problem_type='binary'."
            )

        paths = _extract_required_members(self.cache_dir)
        matrix = np.empty(
            (N_TRAIN + N_TEST, 2 * NT_WIDTH_PER_PARTNER),
            dtype=np.float32,
        )
        matrix[:N_TRAIN, :NT_WIDTH_PER_PARTNER] = _read_embeddings(
            paths["nt_train_seq1.csv"], expected_rows=N_TRAIN
        )
        matrix[:N_TRAIN, NT_WIDTH_PER_PARTNER:] = _read_embeddings(
            paths["nt_train_seq2.csv"], expected_rows=N_TRAIN
        )
        matrix[N_TRAIN:, :NT_WIDTH_PER_PARTNER] = _read_embeddings(
            paths["nt_test_seq1.csv"], expected_rows=N_TEST
        )
        matrix[N_TRAIN:, NT_WIDTH_PER_PARTNER:] = _read_embeddings(
            paths["nt_test_seq2.csv"], expected_rows=N_TEST
        )
        y = pd.concat(
            [
                _read_targets(
                    paths["fusionai_train_target.csv"],
                    expected_rows=N_TRAIN,
                    expected_counts={"1": 18_202, "0": 18_100},
                ),
                _read_targets(
                    paths["fusionai_test_target.csv"],
                    expected_rows=N_TEST,
                    expected_counts={"0": 7_803, "1": 7_754},
                ),
            ],
            ignore_index=True,
        ).rename("fusion")
        groups = exact_partner_components(matrix)
        columns = [
            *(f"nt_partner1_{index}" for index in range(NT_WIDTH_PER_PARTNER)),
            *(f"nt_partner2_{index}" for index in range(NT_WIDTH_PER_PARTNER)),
        ]
        X = pd.DataFrame(matrix, columns=columns)
        metadata = {
            "source_record": ZENODO_RECORD,
            "source_archive": "embeddings.zip",
            "source_archive_bytes": ARCHIVE_BYTES,
            "representation": "Nucleotide Transformer v2 500M multi-species, layer 20, middle token",
            "partner_embedding_width": NT_WIDTH_PER_PARTNER,
            "n_features": int(X.shape[1]),
            "published_train_rows": N_TRAIN,
            "published_test_rows": N_TEST,
            "split_unit": "connected component of exact NT partner representations",
            "target": "fusion",
        }
        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type="binary",
            license=spec.license or "CC BY 4.0",
            source_url=SOURCE_URL,
            citation=CITATION,
            metadata=metadata,
            groups=groups,
        )
