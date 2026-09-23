"""Audit obvious covariate imbalance in the curated GEO methylation datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import pandas as pd

from tabbench_bio.bio.loaders.geo_matrix import _series_url


def read_characteristics(accession: str) -> pd.DataFrame:
    """Read only the sample-metadata header of an official GEO series matrix."""
    import requests

    response = requests.get(_series_url(accession), stream=True, timeout=180)
    response.raise_for_status()
    response.raw.decode_content = False
    characteristics: dict[str, list[str]] = {}
    with gzip.GzipFile(fileobj=response.raw) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            for row in csv.reader(text, delimiter="\t", quotechar='"'):
                if not row:
                    continue
                if row[0] == "!series_matrix_table_begin":
                    break
                if row[0] != "!Sample_characteristics_ch1":
                    continue
                parsed = [value.partition(":") for value in row[1:]]
                keys = {key.strip() for key, separator, _ in parsed if separator}
                if len(keys) == 1:
                    characteristics[next(iter(keys))] = [value.strip() for _, _, value in parsed]
    return pd.DataFrame(characteristics)


def _numeric_by_group(frame: pd.DataFrame, group: str, value: str) -> dict:
    numeric = pd.to_numeric(frame[value], errors="raise")
    summary = (
        frame.assign(**{value: numeric}).groupby(group)[value].agg(["count", "min", "mean", "max"])
    )
    return {
        str(label): {metric: round(float(number), 3) for metric, number in row.items()}
        for label, row in summary.iterrows()
    }


def build_audit() -> dict:
    aging = read_characteristics("GSE40279")
    arthritis = read_characteristics("GSE42861")
    return {
        "GSE40279": {
            "n_samples": int(len(aging)),
            "age_by_plate": _numeric_by_group(aging, "plate", "age (y)"),
            "age_by_source": _numeric_by_group(aging, "source", "age (y)"),
        },
        "GSE42861": {
            "n_samples": int(len(arthritis)),
            "age_by_disease_state": _numeric_by_group(arthritis, "disease state", "age"),
            "gender_by_disease_state": {
                str(label): {str(gender): int(count) for gender, count in row.items()}
                for label, row in pd.crosstab(
                    arthritis["disease state"], arthritis["gender"]
                ).iterrows()
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rendered = json.dumps(build_audit(), indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
