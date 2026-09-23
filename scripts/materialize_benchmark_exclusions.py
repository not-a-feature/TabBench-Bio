"""Apply the shared exclusion policy to existing frozen cells without fitting models."""

import argparse
from pathlib import Path

from tabbench_bio.config import load_config
from tabbench_bio.exclusions import excluded_dataset_keys, materialize_exclusions
from tabbench_bio.result_store import ResultRepository
from tabbench_bio.seeds import get_seeds
from tabbench_bio.site import _expected_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    args = parser.parse_args()
    assert args.results_root.is_dir(), args.results_root
    for cell in sorted(args.results_root.glob("cap_*")):
        if not excluded_dataset_keys(cell.name):
            continue
        config = load_config(str(cell / "config.json"))
        repository = ResultRepository(cell, config)
        excluded = materialize_exclusions(
            repository, get_seeds(config), config["models"], _expected_keys(config)
        )
        print(f"{cell.name}: excluded {sorted(excluded)} across all configured models and folds")


if __name__ == "__main__":
    main()
