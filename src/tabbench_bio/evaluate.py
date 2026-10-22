"""Benchmark your estimator and export fold metrics, Elo, and a local HTML report."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits

from tabbench_bio import Leaderboard
from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.evaluation_report import write_report
from tabbench_bio.io_utils import atomic_write_json
from tabbench_bio.leaderboard import _CURRENT_SQLITE_ATTEMPTS, _open_results_sqlite, _sqlite_frame
from tabbench_bio.metrics import PRIMARY_CLF_METRIC, PRIMARY_REG_METRIC
from tabbench_bio.model_registry import MODEL_DISPLAY
from tabbench_bio.releases import latest_reference
from tabbench_bio.seeds import get_seeds
from tabbench_bio.split_manifest import split_versions, unit_id


def reference_inputs(path, cell, output, datasets, task):
    """Recover the published config and exact held-out row identities read-only."""
    with closing(_open_results_sqlite(path)) as connection:
        row = connection.execute("SELECT config_json FROM cells WHERE cell=?", (cell,)).fetchone()
        assert row is not None, f"Unknown cell: {cell}"
        config = json.loads(row[0])
        tasks = ("classification", "regression") if task == "overall" else (task,)
        selected = [
            name
            for selected_task in tasks
            for name in config["datasets_" + selected_task]
            if get_spec(name).enabled
        ]
        assert set(datasets) <= set(selected), "Choose datasets belonging to this cell and task"
        available = {
            row[0].rsplit("_", 1)[0]
            for row in connection.execute(
                "SELECT DISTINCT dataset FROM attempts WHERE cell=? AND ground_truth_sha256 IS NOT NULL",
                (cell,),
            )
        }
        selected = datasets or [name for name in selected if name in available]
        folds = config["cv_folds"]
        assert folds and folds >= 2, "Published comparisons require complete CV partitions"
        truths = {}
        digests = {}
        for dataset in selected:
            key = dataset + "_0"
            rows = connection.execute(
                "SELECT DISTINCT seed,ground_truth_sha256 FROM attempts "
                "WHERE cell=? AND dataset=? AND ground_truth_sha256 IS NOT NULL",
                (cell, key),
            )
            for seed, digest in rows:
                unit = (int(seed), key)
                if unit in digests:
                    assert digests[unit] == digest, f"Conflicting published test splits: {unit}"
                digests[unit] = digest
                truths[unit] = _sqlite_frame(connection, digest).sort_index()
        units = {}
        seeds = get_seeds(config)
        for dataset in selected:
            key = dataset + "_0"
            assert all((seed, key) in truths for seed in seeds), f"Incomplete published CV: {key}"
            combined = pd.concat([truths[seed, key] for seed in seeds])
            assert combined.index.is_unique, f"Overlapping published test folds: {key}"
            for seed in seeds:
                test = truths[seed, key].index.tolist()
                units[unit_id(seed, key)] = {
                    "test_indices": test,
                    "train_indices": sorted(set(combined.index) - set(test)),
                    "ground_truth_sha256": digests[seed, key],
                }
    atomic_write_json(
        output / "split_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": output.name,
            "versions": split_versions(),
            "cv_folds": folds,
            "units": units,
        },
    )
    for selected_task in ("classification", "regression"):
        field = "datasets_" + selected_task
        config[field] = [
            name for name in config[field] if name in selected and selected_task in tasks
        ]
    return config


def load_model_factory(reference: str):
    """Load a factory from a user-owned Python file."""
    filename, separator, factory_name = reference.rpartition(":")
    if not separator or not filename.endswith(".py"):
        filename, factory_name = reference, "create_model"
    path = Path(filename).resolve()
    assert path.is_file(), f"Model file not found: {path}"
    assert factory_name.isidentifier(), f"Invalid factory name: {factory_name}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"Cannot import model file: {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    factory = vars(module)[factory_name]
    assert callable(factory), f"Model factory is not callable: {reference}"
    return factory


def add_arguments(parser):
    parser.add_argument(
        "--model",
        default="my_model.py",
        help="Python file[:factory]; default factory: create_model(task)",
    )
    parser.add_argument("--name", default="My model")
    parser.add_argument(
        "--dataset", action="append", default=[], help="Repeat to select dataset IDs"
    )
    parser.add_argument(
        "--task", choices=["overall", "classification", "regression"], default="classification"
    )
    parser.add_argument(
        "--baseline",
        default="latest",
        help="Published SQLite path, 'latest' (default: download current release), or 'none' for local smoke baselines",
    )
    parser.add_argument("--cell", default="cap_10000_n100")
    parser.add_argument("--output", type=Path, default=Path("my_model_results"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Shared dataset and metric cache; default: TABBENCH_BIO_CACHE or ~/.cache/tabbench_bio",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate a saved output's report without fitting or downloading",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Omit a model ID from this report; repeat as needed",
    )


def report_scores_sha256(lb, task):
    """Identify the exact scored folds behind a cached Elo table."""
    digest = hashlib.sha256(task.encode())
    for frame, metric in (
        (lb._reg_metrics, PRIMARY_REG_METRIC),
        (lb._clf_metrics, PRIMARY_CLF_METRIC),
    ):
        if not frame.empty:
            ordered = frame[["key", "seed", "model", metric]].sort_values(["key", "seed", "model"])
            digest.update(ordered.to_csv(index=False).encode())
    return digest.hexdigest()


def generate_report(args, lb, config, status):
    """Render the selected comparison without changing stored fold metrics."""
    excluded = set(args.exclude_model)
    assert args.name not in excluded, "Cannot omit the evaluated model"
    known = set().union(
        *(set(frame["model"]) for frame in (lb._reg_metrics, lb._clf_metrics) if not frame.empty)
    )
    assert excluded <= known, f"Unknown model IDs: {sorted(excluded - known)}"
    display = []
    for model in sorted(known):
        parent = model.removesuffix("-TUNED")
        label = MODEL_DISPLAY[parent] if parent in MODEL_DISPLAY else model
        display.append(
            {
                "model_id": model,
                "Model": label
                + (" (tuned)" if model.endswith("-TUNED") and parent in MODEL_DISPLAY else ""),
            }
        )
    display_meta = pd.concat(
        [lb._display_meta, pd.DataFrame(display)], ignore_index=True
    ).drop_duplicates("model_id", keep="first")
    view = Leaderboard(
        (
            lb._reg_metrics.loc[~lb._reg_metrics["model"].isin(excluded)]
            if not lb._reg_metrics.empty
            else pd.DataFrame()
        ),
        (
            lb._clf_metrics.loc[~lb._clf_metrics["model"].isin(excluded)]
            if not lb._clf_metrics.empty
            else pd.DataFrame()
        ),
        display_meta,
    )
    view._added_models = [args.name]
    scores_sha256 = report_scores_sha256(view, args.task)
    report_config = args.output / "report_config.json"
    rank_path = args.output / "leaderboard.csv"
    if args.report_only and report_config.exists() and rank_path.exists():
        previous = json.loads(report_config.read_text(encoding="utf-8"))
        if "scores_sha256" in previous and previous["scores_sha256"] == scores_sha256:
            saved = pd.read_csv(rank_path)
            view._ratings[args.task] = saved[
                ["model_id", "Elo", "Elo_lo", "Elo_hi", "# Targets"]
            ].rename(columns={"# Targets": "n_targets"})
    print("Generating report: computing Elo and 95% confidence intervals...", flush=True)
    with threadpool_limits(limits=1):
        ranking = view.rank(args.task, show_progress=True)
        baseline = args.output / "baseline.json"
        reference = (
            json.loads(baseline.read_text(encoding="utf-8"))["tag"]
            if baseline.exists()
            else ("Local smoke baselines" if args.baseline == "none" else "Published frozen folds")
        )
        settings_path = args.output / "model.json"
        model_settings = (
            json.loads(settings_path.read_text(encoding="utf-8"))["estimator"]
            if settings_path.exists()
            else "Not recorded"
        )
        write_report(
            view,
            ranking,
            args.output,
            focus=args.name,
            task=args.task,
            cell=args.cell,
            config=config,
            status=status,
            reference=reference,
            model_settings=model_settings,
        )
    atomic_write_json(
        args.output / "report_config.json",
        {
            "name": args.name,
            "task": args.task,
            "cell": args.cell,
            "exclude_models": sorted(excluded),
            "scores_sha256": scores_sha256,
        },
    )
    print(f"\nReport: {args.output / 'report.html'}")


def evaluate(args):
    assert args.threads > 0
    if args.report_only:
        assert not args.dataset, "Saved reports use the recorded dataset selection"
        config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        frames = []
        for task in ("regression", "classification"):
            path = args.output / "metrics" / f"{task}_metrics.csv"
            frames.append(pd.read_csv(path) if path.exists() else pd.DataFrame())
        status_path = args.output / "attempts.csv"
        status = (
            pd.read_csv(status_path, keep_default_na=False)
            if status_path.exists()
            else pd.DataFrame(columns=["seed", "key", "model", "status", "reason"])
        )
        generate_report(args, Leaderboard(*frames), config, status)
        return
    factory = load_model_factory(args.model)
    model = (
        {task: factory(task) for task in ("classification", "regression")}
        if args.task == "overall"
        else factory(args.task)
    )
    assert args.task != "overall" or args.baseline != "none", (
        "Use a published reference for an overall comparison"
    )
    assert args.name not in {"RF", "DUMMY"}, "Use a distinct model name"
    output = args.output.resolve()
    assert not output.exists(), f"Choose a new output directory: {output}"
    output.mkdir(parents=True)
    atomic_write_json(output / "model.json", {"factory": args.model, "estimator": repr(model)})
    cache_dir = (args.cache_dir or default_bio_cache_dir().parent).resolve()
    print(f"Cache directory: {cache_dir}", flush=True)
    os.environ["TABBENCH_MODEL_CPUS"] = str(args.threads)
    baseline = None
    status = pd.DataFrame(columns=["seed", "key", "model", "status", "reason"])
    if args.baseline == "latest":
        baseline, provenance = latest_reference(cache_dir)
        atomic_write_json(output / "baseline.json", provenance)
    elif args.baseline != "none":
        baseline = Path(args.baseline).resolve()
    if baseline:
        config = reference_inputs(baseline, args.cell, output, args.dataset, args.task)
        lb = Leaderboard.from_sqlite(
            baseline,
            cell=args.cell,
            metric_cache=cache_dir / "fold_metrics.sqlite",
            show_progress=True,
            datasets=config["datasets_classification"] + config["datasets_regression"],
        )
        selected_keys = {
            name + "_0"
            for name in config["datasets_classification"] + config["datasets_regression"]
        }
        with closing(_open_results_sqlite(baseline)) as connection:
            status = pd.DataFrame(
                [
                    (row[0], row[1], row[2], row[3], row[4])
                    for row in connection.execute(_CURRENT_SQLITE_ATTEMPTS, (args.cell,))
                    if row[1] in selected_keys
                ],
                columns=["seed", "key", "model", "status", "reason"],
            )
    else:
        assert args.task == "classification" or args.dataset, (
            "Choose a regression dataset with --dataset"
        )
        datasets = args.dataset or ["OpenML-1083", "OpenML-1088"]
        config = {
            "datasets_classification": datasets if args.task == "classification" else [],
            "datasets_regression": datasets if args.task == "regression" else [],
            "models": ["DUMMY", "RF"],
            "test_size": 0.2,
            "random_state": 42,
            "n_repetitions": 1,
            "cv_folds": 5,
            "min_samples_per_class": 10,
            "group_regression_splits": False,
            "bio_max_features": 10000,
            "max_classes": None,
            "train_subsample": 100,
            "model_limits": {},
            "model_overrides": {},
            "autogluon_time_limit": 3600,
            "autogluon_presets": "medium_quality",
            "optimize": False,
            "ensemble": False,
            "num_hpo_trials": 0,
            "subsample": None,
            "nan_policy": None,
            "exclude_keys": [],
            "exclude_datasets": [],
            "exclude_targets": [],
        }
        lb = Leaderboard(pd.DataFrame(), pd.DataFrame())
    config["output_dir"] = str(output / args.cell)
    config["cache_dir"] = str(cache_dir)
    config_path = output / "config.json"
    atomic_write_json(config_path, config)
    frames = []
    if not baseline:
        dummy = (
            DummyClassifier(strategy="prior") if args.task == "classification" else DummyRegressor()
        )
        forest_class = (
            RandomForestClassifier if args.task == "classification" else RandomForestRegressor
        )
        forest = forest_class(n_estimators=100, random_state=0, n_jobs=args.threads)
        for name, baseline_model in [("DUMMY", dummy), ("RF", forest)]:
            print(f"Evaluating {name} on all configured folds...", flush=True)
            with threadpool_limits(limits=args.threads):
                frame = lb.evaluate_and_add(name, baseline_model, str(config_path), task=args.task)
            frames.append(frame)
        lb._added_models.clear()
    print(f"Evaluating {args.name} on all configured folds...", flush=True)
    with threadpool_limits(limits=args.threads):
        frames.append(lb.evaluate_and_add(args.name, model, str(config_path), task=args.task))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    (output / "metrics").mkdir()
    for task, frame in [("classification", lb._clf_metrics), ("regression", lb._reg_metrics)]:
        if not frame.empty:
            frame.to_csv(output / "metrics" / f"{task}_metrics.csv", index=False)
    passed = metrics[["seed", "key", "model"]].assign(status="pass", reason="")
    status = pd.concat([status, passed], ignore_index=True)
    status.to_csv(output / "attempts.csv", index=False)
    generate_report(args, lb, config, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
