"""Command-line interface for TabBench Bio.

Usage
-----
::

    # Run the full pipeline (predictions → metrics) for one grid cell's config
    tabbench-bio run --config results/feature_sweep/cap_full/config.json

    # Run individual steps
    tabbench-bio run --config <cfg> --step predictions
    tabbench-bio run --config <cfg> --step metrics

    # Show a leaderboard built from a results directory
    tabbench-bio leaderboard --results-dir results/feature_sweep/cap_full

    # Build a static GitHub Pages site from a results directory
    tabbench-bio site --results-dir results/feature_sweep/cap_full --out docs

    # Package and dataset info
    tabbench-bio info

See Also
--------
- Source: https://github.com/not-a-feature/TabBench-Bio
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from tabbench_bio.elo import DEFAULT_N_BOOT
from tabbench_bio.logging_utils import LOG_FORMAT
from tabbench_bio.model_run import run_model
from tabbench_bio.split_manifest import freeze_manifest

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")


def _apply_run_filters(config, args):
    """Apply scheduling-only filters without changing a frozen cell configuration."""
    from tabbench_bio.bio.datasets import get_spec

    for declared_key, runtime_key in (
        ("datasets_classification", "dataset_names_classification"),
        ("datasets_regression", "dataset_names_regression"),
    ):
        key = runtime_key if runtime_key in config else declared_key
        config[key] = [
            dataset
            for dataset in config[key]
            if get_spec(dataset).enabled
            and (not args.include_dataset or dataset in args.include_dataset)
        ]

    if args.model:
        config["models"] = [args.model]

    include_models = set(args.include_model)
    exclude_models = set(args.exclude_model)
    if include_models and exclude_models:
        raise ValueError("--include-model and --exclude-model are mutually exclusive")
    if args.model and include_models:
        raise ValueError("--model and --include-model are mutually exclusive")
    if include_models:
        config["models"] = [model for model in config["models"] if model in include_models]
    if exclude_models:
        config["models"] = [model for model in config["models"] if model not in exclude_models]

    include_data_types = set(args.include_data_type)
    exclude_data_types = set(args.exclude_data_type)
    if include_data_types and exclude_data_types:
        raise ValueError("--include-data-type and --exclude-data-type are mutually exclusive")
    if include_data_types or exclude_data_types:

        def keep(dataset):
            data_type = get_spec(dataset).data_type
            if include_data_types:
                return data_type in include_data_types
            return data_type not in exclude_data_types

        for declared_key, runtime_key in (
            ("datasets_classification", "dataset_names_classification"),
            ("datasets_regression", "dataset_names_regression"),
        ):
            key = runtime_key if runtime_key in config else declared_key
            config[key] = [dataset for dataset in config[key] if keep(dataset)]


def cmd_run(args):
    """Run the benchmark pipeline (predictions → metrics)."""
    from tabbench_bio.config import load_config

    config = load_config(args.config)

    if args.output:
        config["output_dir"] = args.output
    if args.cache_dir:
        config["cache_dir"] = args.cache_dir

    _apply_run_filters(config, args)

    step = args.step

    if step == "prepare":
        from tabbench_bio.predictions import prepare_splits

        prepare_splits(config, seed_index=args.seed_index)
        return

    if step in ("all", "predictions"):
        from tabbench_bio.predictions import compute_predictions

        compute_predictions(
            config,
            seed_index=args.seed_index,
            overwrite=args.overwrite,
            reverse=args.reverse,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )

    if step in ("all", "metrics"):
        from tabbench_bio.evaluation import compute_metrics_from_predictions

        compute_metrics_from_predictions(config)


def _write_leaderboard_exports(lb, destination: Path, task: str) -> None:
    """Write the CSV table and PNG figure for one cell and task."""
    destination.mkdir(parents=True, exist_ok=True)
    ranking = lb.rank(task)
    ranking.to_csv(destination / "leaderboard.csv", index=False)
    if not ranking.empty and ranking["Elo"].notna().any():
        figure = lb.plot(task=task)
        figure.savefig(destination / "elo.png", dpi=200, bbox_inches="tight")


def cmd_leaderboard(args):
    """Print a leaderboard built from a result directory or published SQLite bundle."""
    from tabbench_bio.leaderboard import Leaderboard

    if args.database or not (args.sqlite or args.results_dir):
        database = Path(args.database or "results/merged/results.sqlite")
        output = Path(args.out or "website")
        from tabbench_bio.dashboard import build_website

        dashboard, frames = build_website(
            database,
            output,
            cells=[args.cell] if args.cell else None,
            workers=args.workers,
            n_boot=args.bootstrap_rounds,
            reference_cell=args.reference_cell,
            results_url=args.results_url,
        )
        for option in dashboard["cell_options"]:
            cell = option["id"]
            assert Path(cell).name == cell and cell not in (".", ".."), f"Invalid cell: {cell}"
            lb = Leaderboard(
                frames["regression"][frames["regression"]["cell"] == cell],
                frames["classification"][frames["classification"]["cell"] == cell],
            )
            if args.task == "overall":
                lb._ratings["overall"] = pd.DataFrame(
                    [row for row in dashboard["cell_elo"] if row["cell"] == cell],
                    columns=["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"],
                )
            print(f"\n{cell}\n{lb.summary(task=args.task)}")
            destination = output / "local" / cell / args.task
            _write_leaderboard_exports(lb, destination, args.task)
            print(f"Local exports: {destination}")
        return

    lb = (
        Leaderboard.from_sqlite(args.sqlite, cell=args.cell)
        if args.sqlite
        else Leaderboard.from_results_dir(args.results_dir)
    )
    print(lb.summary(task=args.task))

    if args.csv:
        lb.rank(args.task).to_csv(args.csv, index=False)
        print(f"Saved {args.csv}")

    if args.plot:
        fig = lb.plot(task=args.task)
        fig.savefig(args.plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved {args.plot_path}")


def cmd_site(args):
    """Build a static, GitHub Pages-ready leaderboard site from a results directory."""
    from tabbench_bio.site import build_site

    path = build_site(
        results_dir=args.results_dir,
        out_dir=args.out,
        config_path=args.config,
        title=args.title,
    )
    print(f"Wrote leaderboard site to {path}")
    print(
        f"Publish: commit {args.out}/ and enable GitHub Pages (Deploy from branch → /{args.out})."
    )


def cmd_info(_args):
    """Print package and ecosystem info."""
    import tabbench_bio

    print(f"tabbench-bio {tabbench_bio.__version__}")
    print()
    print("A benchmark for ML on high-dimensional biological data (GEO/TCGA/Kaggle/OpenML).")
    print()
    print("  Source: https://github.com/not-a-feature/TabBench-Bio")
    print('  Install from the clone: uv pip install -e ".[bio]"')
    print("  Model environments: see environments/README.md")


def cmd_results(args):
    """Manage transactional result bundles."""
    from tabbench_bio.result_store import (
        ResultRepository,
        consolidate_results,
        import_legacy_results,
        install_snapshots,
        merge_results,
        snapshot_database,
        snapshot_writers,
    )

    if args.result_action == "merge":
        path = merge_results(args.source, args.output)
        print(f"Merged result bundles into {path}")
    elif args.result_action == "import-legacy":
        path = import_legacy_results(args.results_dir, bundle_writer=args.writer_id)
        print(f"Imported legacy artifacts into {path}")
    elif args.result_action == "freeze-splits":
        path = freeze_manifest(ResultRepository.from_root(args.results_dir), folds=args.folds)
        print(f"Frozen cross-validation splits at {path}")
    elif args.result_action == "snapshot":
        path = snapshot_database(args.source, args.output)
        print(f"Wrote consistent result snapshot to {path}")
    elif args.result_action == "snapshot-all":
        path = snapshot_writers(args.results_dir, exclude_writers=tuple(args.exclude_writer))
        print(f"Wrote writer snapshot manifest to {path}")
    elif args.result_action == "install-snapshots":
        paths = install_snapshots(args.results_dir, args.snapshot_dir)
        print(f"Installed {len(paths)} writer snapshot(s)")
    elif args.result_action == "consolidate":
        path = consolidate_results(args.results_dir)
        print(f"Consolidated writer bundles into {path}")
    elif args.result_action == "status":
        repository = ResultRepository.from_root(args.results_dir)
        assert repository.bundle_paths(), f"No result database found under {args.results_dir}"
        frame = repository.current_frame()
        print(f"Current units: {len(frame)}")
        if not frame.empty:
            print(frame["status"].value_counts().sort_index().to_string())
    else:
        raise AssertionError(args.result_action)


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        prog="tabbench-bio",
        description="TabBench Bio — ML benchmark for high-dimensional biological data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    model_p = sub.add_parser(
        "model",
        help="Run MODELKEY in its environment profile (also: tabbench-bio MODELKEY)",
        description="Run a model in its installed .venvs/<profile> environment.",
    )
    model_p.add_argument("model_key")
    model_p.add_argument(
        "--full-grid", action="store_true", help="Run all 28 cells; default: reference cell"
    )
    model_p.add_argument("--output", help="Result root; default: results/<modelkey>")
    model_p.add_argument("--cache-dir", help="Dataset cache; also accepts TABBENCH_CACHE_DIR")
    model_p.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Model/library threads, independent of CPU allocation (default: 32)",
    )
    model_p.add_argument("--device", choices=["cpu", "gpu"], help="Override the registered device")
    model_p.set_defaults(func=run_model)

    simple_merge = sub.add_parser("merge", help="Merge databases or result roots into a new file")
    simple_merge.add_argument("source", nargs="+", help="Input databases or result roots")
    simple_merge.add_argument("--output", default="results/merged/results.sqlite")
    simple_merge.set_defaults(func=cmd_results, result_action="merge")

    # ---- run ----
    run_p = sub.add_parser("run", help="Run the benchmark pipeline")
    run_p.add_argument("--config", required=True, help="Path to config JSON")
    run_p.add_argument(
        "--step",
        choices=["all", "prepare", "predictions", "metrics"],
        default="all",
        help="Pipeline step (default: all). 'prepare' only warms the split cache.",
    )
    run_p.add_argument("--output", default=None, help="Override output directory")
    run_p.add_argument(
        "--cache-dir", help="Override the runtime cache without changing frozen settings"
    )
    run_p.add_argument("--seed-index", type=int, default=None, help="Run only this seed index")
    run_p.add_argument("--model", default=None, help="Run only this model")
    run_p.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Run only these models (repeatable; scheduling only)",
    )
    run_p.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Defer these models (repeatable; scheduling only)",
    )
    run_p.add_argument(
        "--include-data-type",
        action="append",
        default=[],
        help="Run only datasets with these curated data types (repeatable)",
    )
    run_p.add_argument(
        "--exclude-data-type",
        action="append",
        default=[],
        help="Skip datasets with these curated data types (repeatable)",
    )
    run_p.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    run_p.add_argument("--reverse", action="store_true", help="Iterate datasets in reverse order")
    run_p.add_argument(
        "--include-dataset",
        action="append",
        default=[],
        help="Run only these dataset IDs without changing the cell config (repeatable)",
    )
    run_p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the model x dataset grid across this many workers (multi-GPU).",
    )
    run_p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this worker runs (0 <= shard-index < num-shards).",
    )
    run_p.set_defaults(func=cmd_run)

    # ---- leaderboard ----
    lb_p = sub.add_parser(
        "leaderboard", help="Show a leaderboard from a result directory or SQLite bundle"
    )
    lb_source = lb_p.add_mutually_exclusive_group()
    lb_source.add_argument(
        "database",
        nargs="?",
        help="SQLite database (default: results/merged/results.sqlite); generates website and local exports",
    )
    lb_p.add_argument(
        "--out", help="Website directory; PNG/CSV exports go in local/ (default: website/)"
    )
    lb_p.add_argument(
        "--workers", type=int, default=1, help="Parallel fold-metric and dashboard Elo workers"
    )
    lb_p.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=DEFAULT_N_BOOT,
        help=f"Target-bootstrap rounds for dashboard and overall reports (default: {DEFAULT_N_BOOT})",
    )
    lb_p.add_argument(
        "--reference-cell",
        help="Website reference cell (default: cap_10000_n100, or first available)",
    )
    lb_p.add_argument(
        "--results-url",
        default="",
        help="Optional URL of this exact database for the website download link",
    )
    lb_source.add_argument("--results-dir", help="Pipeline results directory")
    lb_source.add_argument("--sqlite", help="Published TabBench Bio results.sqlite file")
    lb_p.add_argument("--cell", help="Feature/sample cell within a multi-cell SQLite bundle")
    lb_p.add_argument(
        "--task",
        choices=["overall", "classification", "regression"],
        default="overall",
    )
    lb_p.add_argument("--plot", action="store_true", help="Save leaderboard.png")
    lb_p.add_argument(
        "--plot-path", default="leaderboard.png", help="Figure output path (PNG, SVG, or PDF)"
    )
    lb_p.add_argument("--csv", help="Save the selected fold-level Elo leaderboard as CSV")
    lb_p.set_defaults(func=cmd_leaderboard)

    # ---- site ----
    site_p = sub.add_parser("site", help="Build a static GitHub Pages leaderboard site")
    site_p.add_argument("--results-dir", required=True, help="Pipeline results directory")
    site_p.add_argument("--out", default="docs", help="Output directory (default: docs)")
    site_p.add_argument(
        "--config",
        default=None,
        help="Optional benchmark config (used to look up dataset feature counts)",
    )
    site_p.add_argument(
        "--title",
        default="TabBench Bio Leaderboard",
        help="Page title",
    )
    site_p.set_defaults(func=cmd_site)

    # ---- transactional results ----
    results_p = sub.add_parser("results", help="Manage transactional result bundles")
    results_sub = results_p.add_subparsers(dest="result_action", required=True)
    merge_p = results_sub.add_parser(
        "merge", help="Combine separate model runs into a new SQLite database"
    )
    merge_p.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source SQLite file or result root (repeat for each input)",
    )
    merge_p.add_argument(
        "--output", required=True, help="New database path; must not already exist"
    )
    merge_p.set_defaults(func=cmd_results)

    freeze_p = results_sub.add_parser(
        "freeze-splits", help="Freeze a consistent complete CV split manifest"
    )
    freeze_p.add_argument("--results-dir", required=True)
    freeze_p.add_argument("--folds", type=int, required=True)
    freeze_p.set_defaults(func=cmd_results)

    import_p = results_sub.add_parser(
        "import-legacy", help="Import a legacy JSON/CSV result tree without modifying it"
    )
    import_p.add_argument("--results-dir", required=True)
    import_p.add_argument("--writer-id", default="legacy")
    import_p.set_defaults(func=cmd_results)

    snapshot_p = results_sub.add_parser(
        "snapshot", help="Create an integrity-checked backup of one live writer bundle"
    )
    snapshot_p.add_argument("--source", required=True)
    snapshot_p.add_argument("--output", required=True)
    snapshot_p.set_defaults(func=cmd_results)

    snapshot_all_p = results_sub.add_parser(
        "snapshot-all", help="Snapshot every live writer bundle for transfer"
    )
    snapshot_all_p.add_argument("--results-dir", required=True)
    snapshot_all_p.add_argument("--exclude-writer", action="append", default=[])
    snapshot_all_p.set_defaults(func=cmd_results)

    install_p = results_sub.add_parser(
        "install-snapshots", help="Validate and install transferred writer snapshots"
    )
    install_p.add_argument("--results-dir", required=True)
    install_p.add_argument("--snapshot-dir", required=True)
    install_p.set_defaults(func=cmd_results)

    consolidate_p = results_sub.add_parser(
        "consolidate", help="Merge all writer bundles into the canonical result database"
    )
    consolidate_p.add_argument("--results-dir", required=True)
    consolidate_p.set_defaults(func=cmd_results)

    status_p = results_sub.add_parser("status", help="Show current-unit counts across bundles")
    status_p.add_argument("--results-dir", required=True)
    status_p.set_defaults(func=cmd_results)

    # ---- info ----
    info_p = sub.add_parser("info", help="Show package and ecosystem info")
    info_p.set_defaults(func=cmd_info)

    arguments = sys.argv[1:]
    if arguments and not arguments[0].startswith("-") and arguments[0] not in sub.choices:
        arguments = ["model", *arguments]
    args = parser.parse_args(arguments)

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
