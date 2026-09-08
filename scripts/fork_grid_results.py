#!/usr/bin/env python3
"""Seed an expanded benchmark result tree without modifying its source run."""

from __future__ import annotations

import argparse
import json

from tabbench_bio.result_fork import fork_grid_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    marker = fork_grid_results(args.source, args.destination)
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
