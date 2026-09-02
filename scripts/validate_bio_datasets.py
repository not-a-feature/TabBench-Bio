"""Fetch candidate datasets and write an auditable TabBench Bio quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabbench_bio.bio.adapter import load_bio_dataset
from tabbench_bio.bio.datasets import bio_dataset_names
from tabbench_bio.bio.quality import assess_raw_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="*", help="Registry IDs (default: every runnable dataset)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--force-refetch", action="store_true")
    args = parser.parse_args()

    dataset_ids = args.dataset or bio_dataset_names()
    reports = []
    for dataset_id in dataset_ids:
        try:
            raw = load_bio_dataset(
                dataset_id,
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                force_refetch=args.force_refetch,
            )
            reports.append(assess_raw_dataset(raw))
        except Exception as exc:  # report source failures alongside data-quality failures
            reports.append(
                {
                    "bio_id": dataset_id,
                    "status": "error",
                    "issues": [f"{type(exc).__name__}: {exc}"],
                }
            )

    payload = {
        "status": "pass" if all(r["status"] == "pass" for r in reports) else "fail",
        "datasets": reports,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
