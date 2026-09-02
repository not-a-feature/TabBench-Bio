"""Streaming loader for very large GEO series matrices.

GEO methylation series can be several gigabytes even when compressed.  This loader
streams the official matrix once and retains a deterministic, value-independent
subset of complete probes.  Feature selection is based solely on a stable hash of
the probe identifier, so it cannot leak target or test-fold information.  The
assembled matrix is subsequently stored by TabBench-Bio's unified dataset cache.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

import numpy as np
import pandas as pd

from tabbench_bio.bio.loaders.base import BioRawDataset

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


DEFAULT_STREAM_FEATURES = 30_000
GEO_TITLE_PLATE = "geo_title_plate"
GEO_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{accession}/matrix/"
    "{accession}_series_matrix.txt.gz"
)
GSE147221_PROCESSED_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147221/suppl/"
    "GSE147221_Dublin_blood_processed_signals.csv.gz"
)


@dataclass(frozen=True)
class GeoMatrixRecord:
    problem_type: str
    citation: str
    target_mapping: tuple[tuple[str, str], ...] = ()
    excluded_targets: tuple[str, ...] = ()
    processed_signals_url: str | None = None


RECORDS: dict[str, GeoMatrixRecord] = {
    "GSE40279": GeoMatrixRecord(
        "regression",
        "Hannum G et al. (2013), Genome-wide methylation profiles reveal quantitative views of human aging rates.",
    ),
    "GSE42861": GeoMatrixRecord(
        "binary",
        "Liu Y et al. (2013), Epigenome-wide association data implicate DNA methylation as an intermediary of genetic risk in rheumatoid arthritis.",
    ),
    "GSE50660": GeoMatrixRecord(
        "multiclass",
        "Tsaprouni LG et al. (2014), Cigarette smoking reduces DNA methylation levels at multiple genomic loci but the effect is partially reversible upon cessation.",
        target_mapping=(("0", "never"), ("1", "former"), ("2", "current")),
    ),
    "GSE147221": GeoMatrixRecord(
        "binary",
        "Hannon E et al. (2021), DNA methylation meta-analysis reveals cellular alterations in psychosis and markers of treatment-resistant schizophrenia.",
        target_mapping=(
            ("Case", "case"),
            ("case", "case"),
            ("Control", "control"),
            ("control", "control"),
        ),
        excluded_targets=("NA",),
        processed_signals_url=GSE147221_PROCESSED_URL,
    ),
}


def _series_url(accession: str) -> str:
    """Build NCBI's deterministic GEO series-matrix URL."""
    bucket = f"{accession[:-3]}nnn"
    return GEO_MATRIX_URL.format(bucket=bucket, accession=accession)


def _stable_score(probe_id: str) -> int:
    """Stable random-like score used for value-independent feature selection."""
    digest = hashlib.blake2b(probe_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _characteristic(rows: list[list[str]], key: str, bio_id: str) -> pd.Series:
    """Extract one consistently named ``key: value`` GEO sample characteristic."""
    wanted = key.strip().casefold()
    matches: list[list[str]] = []
    for row in rows:
        parsed = [value.partition(":") for value in row]
        keys = {prefix.strip().casefold() for prefix, separator, _ in parsed if separator}
        if keys == {wanted}:
            matches.append([value.strip() for _, _, value in parsed])
    if len(matches) != 1:
        raise ValueError(
            f"{bio_id}: expected exactly one GEO characteristic row for {key!r}, "
            f"found {len(matches)}."
        )
    return pd.Series(matches[0], name=key)


def parse_series_matrix(
    stream: TextIO,
    *,
    bio_id: str,
    target: str,
    problem_type: str,
    max_features: int,
    group_characteristic: str | None = None,
    excluded_targets: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Parse and cap a decompressed GEO series matrix from a text stream."""
    if max_features <= 0:
        raise ValueError(f"{bio_id}: max_features must be positive, got {max_features}.")

    reader = csv.reader(stream, delimiter="\t", quotechar='"')
    sample_accessions: list[str] | None = None
    sample_titles: list[str] | None = None
    characteristics: list[list[str]] = []
    matrix_samples: list[str] | None = None
    matrix_column_count: int | None = None
    retained_sample_indices: list[int] | None = None
    excluded_sample_count = 0
    in_table = False
    raw_feature_count = 0
    malformed_rows = 0
    incomplete_candidates = 0
    # Python's heap is a min-heap. Negative scores put the worst retained probe at
    # the root, making replacement O(log max_features).
    retained: list[tuple[int, str, int, np.ndarray]] = []

    for row in reader:
        if not row:
            continue
        marker = row[0]
        if not in_table:
            if marker == "!Sample_geo_accession":
                sample_accessions = row[1:]
            elif marker == "!Sample_title":
                sample_titles = row[1:]
            elif marker == "!Sample_characteristics_ch1":
                characteristics.append(row[1:])
            elif marker == "!series_matrix_table_begin":
                in_table = True
            continue

        if marker == "!series_matrix_table_end":
            break
        if matrix_samples is None:
            if marker != "ID_REF":
                raise ValueError(
                    f"{bio_id}: GEO matrix header starts with {marker!r}, not 'ID_REF'."
                )
            all_matrix_samples = row[1:]
            if sample_accessions is None:
                raise ValueError(f"{bio_id}: missing !Sample_geo_accession metadata row.")
            if all_matrix_samples != sample_accessions:
                raise ValueError(
                    f"{bio_id}: GEO metadata samples do not align with matrix columns."
                )
            matrix_column_count = len(all_matrix_samples)
            labels = _characteristic(characteristics, target, bio_id).astype("string")
            keep = ~labels.isin(excluded_targets)
            retained_sample_indices = np.flatnonzero(keep.to_numpy()).tolist()
            excluded_sample_count = int((~keep).sum())
            matrix_samples = [all_matrix_samples[index] for index in retained_sample_indices]
            if not matrix_samples:
                raise ValueError(f"{bio_id}: target exclusions removed every sample.")
            if sample_titles is not None:
                sample_titles = [sample_titles[index] for index in retained_sample_indices]
            continue

        raw_feature_count += 1
        assert matrix_column_count is not None
        assert retained_sample_indices is not None
        if len(row) != matrix_column_count + 1:
            malformed_rows += 1
            continue
        probe_id = row[0]
        score = _stable_score(probe_id)
        if len(retained) >= max_features and score >= -retained[0][0]:
            continue
        try:
            values = np.asarray(
                [row[index + 1] for index in retained_sample_indices], dtype=np.float32
            )
        except ValueError:
            incomplete_candidates += 1
            continue
        if not np.isfinite(values).all():
            incomplete_candidates += 1
            continue
        # raw_feature_count is an explicit tiebreaker, so duplicate probe IDs can
        # never make heapq compare numpy arrays.
        item = (-score, probe_id, raw_feature_count, values)
        if len(retained) < max_features:
            heapq.heappush(retained, item)
        else:
            heapq.heapreplace(retained, item)

    if matrix_samples is None:
        raise ValueError(f"{bio_id}: GEO series matrix has no data table.")
    if len(retained) < max_features:
        raise ValueError(
            f"{bio_id}: only {len(retained)} complete probes retained; requested {max_features}."
        )
    if malformed_rows:
        raise ValueError(f"{bio_id}: encountered {malformed_rows} malformed matrix row(s).")

    selected = sorted(
        [(-negative_score, probe_id, values) for negative_score, probe_id, _, values in retained]
    )
    matrix = np.vstack([values for _, _, values in selected]).T
    X = pd.DataFrame(matrix, index=matrix_samples, columns=[probe for _, probe, _ in selected])
    assert retained_sample_indices is not None
    y = _characteristic(characteristics, target, bio_id).iloc[retained_sample_indices]
    y.index = matrix_samples
    if problem_type == "regression":
        y = pd.to_numeric(y, errors="raise").astype(np.float32)
        y.name = target
    if group_characteristic is None:
        groups = pd.Series(matrix_samples, index=matrix_samples, name="geo_sample")
        group_strategy = "GEO sample accession"
    elif group_characteristic == GEO_TITLE_PLATE:
        if sample_titles is None or len(sample_titles) != len(matrix_samples):
            raise ValueError(f"{bio_id}: missing or misaligned !Sample_title metadata row.")
        split_titles = [title.partition("_") for title in sample_titles]
        if any(not separator or not plate for plate, separator, _ in split_titles):
            raise ValueError(f"{bio_id}: cannot derive assay plates from GEO sample titles.")
        groups = pd.Series(
            [plate for plate, _, _ in split_titles],
            index=matrix_samples,
            name=GEO_TITLE_PLATE,
        )
        group_strategy = "GEO sample-title prefix before '_' (assay plate)"
    else:
        groups = _characteristic(characteristics, group_characteristic, bio_id).iloc[
            retained_sample_indices
        ]
        groups.index = matrix_samples
        groups.name = group_characteristic
        group_strategy = f"GEO sample characteristic: {group_characteristic}"
    metadata = {
        "accession": bio_id.removeprefix("GEO-"),
        "target": target,
        "raw_feature_count": int(raw_feature_count),
        "selected_features": int(max_features),
        "selection": "lowest BLAKE2b-64 probe-id hashes among complete probes",
        "incomplete_candidates_skipped": int(incomplete_candidates),
        "n_samples": int(len(matrix_samples)),
        "excluded_control_samples": excluded_sample_count,
        "group_strategy": group_strategy,
    }
    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        groups.reset_index(drop=True),
        metadata,
    )


def _parse_series_metadata(
    stream: TextIO, *, bio_id: str, target: str
) -> tuple[list[str], list[str], list[list[str]]]:
    """Read aligned sample metadata without requiring a populated value table."""
    reader = csv.reader(stream, delimiter="\t", quotechar='"')
    sample_accessions: list[str] | None = None
    sample_titles: list[str] | None = None
    characteristics: list[list[str]] = []
    for row in reader:
        if not row:
            continue
        marker = row[0]
        if marker == "!Sample_geo_accession":
            sample_accessions = row[1:]
        elif marker == "!Sample_title":
            sample_titles = row[1:]
        elif marker == "!Sample_characteristics_ch1":
            characteristics.append(row[1:])
        elif marker == "!series_matrix_table_begin":
            break
    if sample_accessions is None or sample_titles is None:
        raise ValueError(f"{bio_id}: missing GEO sample accessions or titles.")
    if len(sample_accessions) != len(sample_titles):
        raise ValueError(f"{bio_id}: GEO sample accessions and titles are misaligned.")
    _characteristic(characteristics, target, bio_id)
    return sample_accessions, sample_titles, characteristics


def parse_processed_signals(
    stream: TextIO,
    *,
    bio_id: str,
    sample_accessions: list[str],
    sample_titles: list[str],
    characteristics: list[list[str]],
    target: str,
    max_features: int,
    group_characteristic: str | None,
    excluded_targets: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Parse GSE147221's paired beta-value/detection-p-value supplement."""
    if max_features <= 0:
        raise ValueError(f"{bio_id}: max_features must be positive, got {max_features}.")
    if len(sample_accessions) != len(sample_titles):
        raise ValueError(f"{bio_id}: GEO sample metadata are misaligned.")

    normalized_titles = [title.partition(":")[0].strip() for title in sample_titles]
    if len(normalized_titles) != len(set(normalized_titles)):
        raise ValueError(f"{bio_id}: normalized GEO sample titles are not unique.")
    labels = _characteristic(characteristics, target, bio_id).astype("string")
    labels.index = normalized_titles
    accession_by_title = dict(zip(normalized_titles, sample_accessions, strict=True))

    reader = csv.reader(stream, delimiter=",", quotechar='"')
    header = next(reader)
    if not header or header[0]:
        raise ValueError(f"{bio_id}: processed-signal header lacks an empty probe column.")
    beta_indices = [
        index
        for index, value in enumerate(header[1:], start=1)
        if not value.endswith("_Detection_Pval")
    ]
    if not beta_indices:
        raise ValueError(f"{bio_id}: processed supplement has no beta-value columns.")
    for index in beta_indices:
        if index + 1 >= len(header) or header[index + 1] != (f"{header[index]}_Detection_Pval"):
            raise ValueError(
                f"{bio_id}: beta column {header[index]!r} lacks its adjacent detection P-value."
            )
    supplement_titles = [header[index] for index in beta_indices]
    if len(supplement_titles) != len(set(supplement_titles)):
        raise ValueError(f"{bio_id}: duplicate beta-value sample columns in supplement.")
    unexpected_titles = sorted(set(supplement_titles) - set(normalized_titles))
    if unexpected_titles:
        raise ValueError(
            f"{bio_id}: supplement contains samples absent from GEO metadata: {unexpected_titles}."
        )

    selected_pairs = [
        (index, title)
        for index, title in zip(beta_indices, supplement_titles, strict=True)
        if labels.loc[title] not in excluded_targets
    ]
    selected_indices = [index for index, _ in selected_pairs]
    selected_titles = [title for _, title in selected_pairs]
    if not selected_titles:
        raise ValueError(f"{bio_id}: target exclusions removed every supplement sample.")

    raw_feature_count = 0
    malformed_rows = 0
    incomplete_candidates = 0
    retained: list[tuple[int, str, int, np.ndarray]] = []
    for row in reader:
        if not row:
            continue
        raw_feature_count += 1
        if len(row) != len(header):
            malformed_rows += 1
            continue
        probe_id = row[0]
        score = _stable_score(probe_id)
        if len(retained) >= max_features and score >= -retained[0][0]:
            continue
        try:
            values = np.asarray([row[index] for index in selected_indices], dtype=np.float32)
        except ValueError:
            incomplete_candidates += 1
            continue
        if not np.isfinite(values).all():
            incomplete_candidates += 1
            continue
        item = (-score, probe_id, raw_feature_count, values)
        if len(retained) < max_features:
            heapq.heappush(retained, item)
        else:
            heapq.heapreplace(retained, item)

    if len(retained) < max_features:
        raise ValueError(
            f"{bio_id}: only {len(retained)} complete processed probes retained; "
            f"requested {max_features}."
        )
    if malformed_rows:
        raise ValueError(f"{bio_id}: encountered {malformed_rows} malformed supplement row(s).")

    selected = sorted(
        [(-negative_score, probe_id, values) for negative_score, probe_id, _, values in retained]
    )
    matrix = np.vstack([values for _, _, values in selected]).T
    X = pd.DataFrame(matrix, columns=[probe for _, probe, _ in selected])
    y = labels.loc[selected_titles].reset_index(drop=True).rename(target)
    if group_characteristic == GEO_TITLE_PLATE:
        split_titles = [title.partition("_") for title in selected_titles]
        if any(not separator or not plate for plate, separator, _ in split_titles):
            raise ValueError(f"{bio_id}: cannot derive assay plates from supplement titles.")
        groups = pd.Series([plate for plate, _, _ in split_titles], name=GEO_TITLE_PLATE)
        group_strategy = "GEO sample-title prefix before '_' (assay plate)"
    elif group_characteristic is None:
        groups = pd.Series(
            [accession_by_title[title] for title in selected_titles], name="geo_sample"
        )
        group_strategy = "GEO sample accession"
    else:
        group_values = _characteristic(characteristics, group_characteristic, bio_id)
        group_values.index = normalized_titles
        groups = group_values.loc[selected_titles].reset_index(drop=True)
        groups.name = group_characteristic
        group_strategy = f"GEO sample characteristic: {group_characteristic}"

    missing_from_supplement = set(normalized_titles) - set(supplement_titles)
    metadata = {
        "target": target,
        "raw_feature_count": int(raw_feature_count),
        "selected_features": int(max_features),
        "selection": "lowest BLAKE2b-64 probe-id hashes among complete beta-value probes",
        "incomplete_candidates_skipped": int(incomplete_candidates),
        "n_samples": int(len(selected_titles)),
        "group_strategy": group_strategy,
        "series_samples": int(len(normalized_titles)),
        "supplement_beta_samples": int(len(supplement_titles)),
        "series_samples_absent_from_supplement": int(len(missing_from_supplement)),
        "excluded_control_samples": int(labels.isin(excluded_targets).sum()),
        "supplement_layout": "alternating beta-value and detection-P-value columns; beta values retained",
    }
    return X, y, groups, metadata


def _curate_target_labels(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: dict,
    *,
    bio_id: str,
    record: GeoMatrixRecord,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Apply one record's explicit label vocabulary and control exclusions."""
    if not record.target_mapping:
        return X, y, groups, metadata

    labels = y.astype("string")
    mapping = dict(record.target_mapping)
    known = set(mapping) | set(record.excluded_targets)
    unexpected = sorted(set(labels.unique()) - known)
    if unexpected:
        raise ValueError(
            f"{bio_id}: unexpected target label(s) {unexpected}; expected {sorted(known)}."
        )
    keep = ~labels.isin(record.excluded_targets)
    curated = labels.loc[keep].map(mapping)
    if curated.isna().any():
        raise ValueError(f"{bio_id}: target mapping produced missing labels.")
    metadata = dict(metadata)
    excluded_before = (
        int(metadata["excluded_control_samples"]) if "excluded_control_samples" in metadata else 0
    )
    metadata["excluded_control_samples"] = excluded_before + int((~keep).sum())
    metadata["target_mapping"] = mapping
    metadata["n_samples"] = int(keep.sum())
    return (
        X.loc[keep].reset_index(drop=True),
        curated.reset_index(drop=True).rename(y.name),
        groups.loc[keep].reset_index(drop=True),
        metadata,
    )


class GeoMatrixLoader:
    """Stream one curated, very large GEO series matrix."""

    def fetch(self, spec: BioDatasetSpec) -> BioRawDataset:
        import requests

        accession = spec.fetch_id.strip().upper()
        if accession not in RECORDS:
            raise ValueError(
                f"{spec.bio_id}: unsupported streamed GEO record {accession!r}; "
                f"curated records are {sorted(RECORDS)}."
            )
        record = RECORDS[accession]
        if spec.target is None:
            raise ValueError(f"{spec.bio_id}: geo_matrix loader needs a curated target.")
        if spec.problem_type is not None and spec.problem_type != record.problem_type:
            raise ValueError(
                f"{spec.bio_id}: problem_type={spec.problem_type!r} disagrees with "
                f"curated record type {record.problem_type!r}."
            )

        url = _series_url(accession)
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", errors="strict", newline=""
            ) as text:
                if record.processed_signals_url is None:
                    X, y, groups, metadata = parse_series_matrix(
                        text,
                        bio_id=spec.bio_id,
                        target=spec.target,
                        problem_type=record.problem_type,
                        max_features=spec.max_features or DEFAULT_STREAM_FEATURES,
                        group_characteristic=spec.group_column,
                        excluded_targets=record.excluded_targets,
                    )
                else:
                    sample_accessions, sample_titles, characteristics = _parse_series_metadata(
                        text, bio_id=spec.bio_id, target=spec.target
                    )
        if record.processed_signals_url is not None:
            response = requests.get(record.processed_signals_url, stream=True, timeout=180)
            response.raise_for_status()
            response.raw.decode_content = False
            with gzip.GzipFile(fileobj=response.raw) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf-8", errors="strict", newline=""
                ) as text:
                    X, y, groups, metadata = parse_processed_signals(
                        text,
                        bio_id=spec.bio_id,
                        sample_accessions=sample_accessions,
                        sample_titles=sample_titles,
                        characteristics=characteristics,
                        target=spec.target,
                        max_features=spec.max_features or DEFAULT_STREAM_FEATURES,
                        group_characteristic=spec.group_column,
                        excluded_targets=record.excluded_targets,
                    )
            metadata["processed_signals_url"] = record.processed_signals_url
        X, y, groups, metadata = _curate_target_labels(
            X,
            y,
            groups,
            metadata,
            bio_id=spec.bio_id,
            record=record,
        )
        metadata["accession"] = accession

        return BioRawDataset(
            bio_id=spec.bio_id,
            X=X,
            y=y,
            problem_type=record.problem_type,
            license=spec.license or "NCBI GEO public data; study-specific terms apply",
            source_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
            citation=record.citation,
            metadata=metadata,
            groups=groups,
        )
