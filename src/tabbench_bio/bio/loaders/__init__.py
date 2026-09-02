"""Per-source fetch-by-id loaders for TabBench-Bio.

Each loader turns a :class:`~tabbench_bio.bio.datasets.BioDatasetSpec` into a
:class:`~tabbench_bio.bio.loaders.base.BioRawDataset`. :func:`get_loader` dispatches on
the spec's ``source`` and forwards per-source options (GEO platform via the ``@GPL``
fetch-id, Kaggle ``data_file``, local ``embedding_column``). Heavy/optional source dependencies (GEOparse, biopython,
kagglehub, openml, requests) are imported lazily inside each loader so importing this
package never forces them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabbench_bio.bio.loaders.base import BioLoader, BioRawDataset

if TYPE_CHECKING:
    from tabbench_bio.bio.datasets import BioDatasetSpec


def get_loader(spec: BioDatasetSpec, *, cache_dir: str | None = None) -> BioLoader:
    """Return the loader instance for a given spec (fail-fast on unknown source).

    Parameters
    ----------
    spec : BioDatasetSpec
        The dataset spec; its ``source`` selects the loader and per-source options
        (``data_file`` for Kaggle, ``embedding_column`` for local) are forwarded.
    cache_dir : str | None
        Bio cache root; the per-source caches (TCGA matrix, GEO SOFT) live under it.
        ``None`` uses each loader's default.
    """
    from pathlib import Path

    source = spec.source
    if source == "geo":
        from tabbench_bio.bio.loaders.geo import GeoLoader

        destdir = Path(cache_dir) / "geo_raw" if cache_dir else None
        return GeoLoader(destdir=destdir)
    if source == "geo_matrix":
        from tabbench_bio.bio.loaders.geo_matrix import GeoMatrixLoader

        return GeoMatrixLoader()
    if source == "fusionai":
        from tabbench_bio.bio.loaders.fusionai import FusionAiNtLoader

        return FusionAiNtLoader(cache_dir=Path(cache_dir) / "fusionai_raw" if cache_dir else None)
    if source == "tcga":
        from tabbench_bio.bio.loaders.tcga import TcgaLoader

        tcga_cache = Path(cache_dir) / "tcga_raw" if cache_dir else None
        return TcgaLoader(cache_dir=tcga_cache)
    if source == "kaggle":
        from tabbench_bio.bio.loaders.kaggle import KaggleLoader

        return KaggleLoader(data_file=spec.data_file)
    if source == "openml":
        from tabbench_bio.bio.loaders.openml import OpenMLLoader

        return OpenMLLoader()
    if source == "local":
        from tabbench_bio.bio.loaders.local import LocalLoader

        return LocalLoader(
            embedding_column=spec.embedding_column,
            group_column=spec.group_column,
        )
    if source == "mgnify":
        from tabbench_bio.bio.loaders.mgnify import MgnifyLoader

        return MgnifyLoader(cache_dir=Path(cache_dir) / "mgnify_raw" if cache_dir else None)
    if source == "metagenomics":
        from tabbench_bio.bio.loaders.metagenomics import MetagenomicsLoader

        return MetagenomicsLoader(
            cache_dir=Path(cache_dir) / "metagenomics_raw" if cache_dir else None
        )
    if source == "tdc":
        from tabbench_bio.bio.loaders.tdc import TdcLoader

        return TdcLoader(cache_dir=Path(cache_dir) / "tdc_raw" if cache_dir else None)
    if source == "gp":
        from tabbench_bio.bio.loaders.genomic_prediction import GenomicPredictionLoader

        return GenomicPredictionLoader(cache_dir=Path(cache_dir) / "gp_raw" if cache_dir else None)
    if source == "chembl":
        from tabbench_bio.bio.loaders.chembl import ChemblLoader

        return ChemblLoader()
    raise ValueError(f"Unknown bio source: {source!r}")


__all__ = ["BioLoader", "BioRawDataset", "get_loader"]
