"""Materialise status records for dataset-level null splits before grid finalisation.

The benchmark iterator can deliberately return ``(None, None)`` when configured
preprocessing leaves no usable labelled split, for example when fewer than two classes
remain after rare-class filtering.  The prediction loop cannot fit any model in that case,
but older runs did not emit the per-model design-skip records required by the coverage gate.

This runner helper fills only that narrow gap.  A seed/target is eligible when every model
is absent from both metrics and status records, and the iterator independently reproduces
the null split.  All other missing units remain untouched and therefore block publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tabbench_bio.benchmark import configure_benchmark
from tabbench_bio.config import load_config
from tabbench_bio.coverage import load_status, missing_units
from tabbench_bio.predictions import _write_skip_record
from tabbench_bio.seeds import get_seeds
from tabbench_bio.site import _expected_keys

SKIP_REASON = "empty_split_after_filtering"
SKIP_DETAIL = (
    "no train/test split: configured preprocessing left fewer than two target classes "
    "or no labelled rows"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Restrict validation to one cell directory name; may be repeated",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_metrics(cell_dir: Path) -> pd.DataFrame:
    frames = []
    for filename in ("classification_metrics.csv", "regression_metrics.csv"):
        path = cell_dir / "metrics" / filename
        if path.is_file():
            frames.append(pd.read_csv(path))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["seed", "key", "model"])


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    assert results_root.is_dir(), results_root

    if args.cell:
        cell_dirs = [results_root / name for name in args.cell]
        assert len(cell_dirs) == len(set(cell_dirs)), "Duplicate --cell argument."
        for cell_dir in cell_dirs:
            assert cell_dir.is_dir(), cell_dir
    else:
        cell_dirs = sorted(path for path in results_root.glob("cap_*") if path.is_dir())
    assert cell_dirs, f"No grid cells found under {results_root}."

    planned: list[tuple[Path, str, str]] = []
    non_null_gaps = []
    for cell_dir in cell_dirs:
        config_path = cell_dir / "config.json"
        assert config_path.is_file(), config_path
        config = load_config(str(config_path))
        config["output_dir"] = str(cell_dir)
        models = config["models"]
        seeds = get_seeds(config)
        missing = missing_units(
            read_metrics(cell_dir),
            load_status(str(cell_dir)),
            keys=_expected_keys(config),
            models=models,
            seeds=seeds,
        )
        if missing.empty:
            continue

        counts = missing.groupby(["seed", "key"])["model"].nunique()
        candidates = [
            (int(seed), str(key))
            for (seed, key), count in counts.items()
            if int(count) == len(models)
        ]
        for seed in sorted({seed for seed, _key in candidates}):
            seed_keys = {key for candidate_seed, key in candidates if candidate_seed == seed}
            seed_config = dict(config)
            seed_config["random_state"] = seed
            benchmark = configure_benchmark(seed_config)
            observed = {
                key: (data_train, data_test)
                for data_train, data_test, key, _task_type in benchmark
                if key in seed_keys
            }
            assert set(observed) == seed_keys, (
                f"{cell_dir.name}/seed_{seed}: configured candidate keys missing from iterator: "
                f"{sorted(seed_keys - set(observed))}"
            )
            for key in sorted(seed_keys):
                data_train, data_test = observed[key]
                if data_train is not None or data_test is not None:
                    non_null_gaps.append((cell_dir.name, seed, key))
                    continue
                stats_dir = cell_dir / f"seed_{seed}" / "stats"
                for model in models:
                    status_path = stats_dir / f"{key}_{model}.json"
                    assert not status_path.exists(), status_path
                    planned.append((stats_dir, key, model))

    for cell, seed, key in non_null_gaps:
        print(f"untouched incomplete target: {cell}/seed_{seed}/{key}")
    for stats_dir, key, model in planned:
        print(f"{'would write' if args.dry_run else 'writing'}: {stats_dir}/{key}_{model}.json")
        if not args.dry_run:
            stats_dir.mkdir(parents=True, exist_ok=True)
            _write_skip_record(
                str(stats_dir),
                key,
                model,
                0,
                0,
                SKIP_REASON,
                SKIP_DETAIL,
            )
    print(
        f"Materialised {0 if args.dry_run else len(planned)} design-skip records; "
        f"{len(planned)} eligible; {len(non_null_gaps)} whole-target gaps remained incomplete."
    )


if __name__ == "__main__":
    main()
