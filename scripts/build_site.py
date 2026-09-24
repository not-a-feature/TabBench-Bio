"""Build the packaged website directly from a result database."""

import argparse
from pathlib import Path

from tabbench_bio.dashboard import build_website
from tabbench_bio.elo import DEFAULT_N_BOOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-sqlite", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-rounds", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--reference-cell")
    parser.add_argument("--results-sqlite-url", default="")
    args = parser.parse_args()
    build_website(
        args.results_sqlite,
        args.site_dir,
        workers=args.workers,
        n_boot=args.bootstrap_rounds,
        reference_cell=args.reference_cell,
        results_url=args.results_sqlite_url,
    )


if __name__ == "__main__":
    main()
