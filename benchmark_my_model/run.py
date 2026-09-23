"""Benchmark your estimator and export fold metrics, Elo, and a local HTML report."""

import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd
from my_model import create_model
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits

from tabbench_bio import Leaderboard
from tabbench_bio.bio.datasets import get_spec
from tabbench_bio.io_utils import atomic_write_json
from tabbench_bio.leaderboard import _open_results_sqlite, _sqlite_frame
from tabbench_bio.split_manifest import split_versions, unit_id


def reference_inputs(path, cell, output, datasets, task):
    """Recover the published config and exact held-out row identities read-only."""
    with _open_results_sqlite(path) as connection:
        row = connection.execute("SELECT config_json FROM cells WHERE cell=?", (cell,)).fetchone()
        assert row is not None, f"Unknown cell: {cell}"
        config = json.loads(row[0])
        selected = [name for name in config["datasets_" + task] if get_spec(name).enabled]
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
        for dataset in selected:
            key = dataset + "_0"
            for repeat in range(config["n_repetitions"] or 1):
                seeds = list(range(repeat * folds, (repeat + 1) * folds))
                assert all((seed, key) in truths for seed in seeds), (
                    f"Incomplete published CV: {key}"
                )
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
    config["datasets_" + task] = selected
    config["datasets_regression" if task == "classification" else "datasets_classification"] = []
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="My model")
    parser.add_argument(
        "--dataset", action="append", default=[], help="Repeat to select dataset IDs"
    )
    parser.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument("--baseline", type=Path, help="Published SQLite results; never modified")
    parser.add_argument("--cell", default="cap_10000_n100")
    parser.add_argument("--output", type=Path, default=Path("my_model_results"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    assert args.threads > 0
    assert args.name not in {"RF", "DUMMY"}, "Use a distinct model name"
    output = args.output.resolve()
    assert not output.exists(), f"Choose a new output directory: {output}"
    output.mkdir(parents=True)
    os.environ["TABBENCH_MODEL_CPUS"] = str(args.threads)
    if args.baseline:
        config = reference_inputs(
            args.baseline.resolve(), args.cell, output, args.dataset, args.task
        )
        lb = Leaderboard.from_sqlite(args.baseline, cell=args.cell)
        mode = "Comparison with published baselines on matching frozen folds"
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
        mode = (
            "Small local comparison; these freshly fitted baselines are not the published benchmark"
        )
    config["output_dir"] = str(output / args.cell)
    config["cache_dir"] = str(output / "cache")
    config_path = output / "config.json"
    atomic_write_json(config_path, config)
    frames = []
    if not args.baseline:
        dummy = (
            DummyClassifier(strategy="prior") if args.task == "classification" else DummyRegressor()
        )
        forest_class = (
            RandomForestClassifier if args.task == "classification" else RandomForestRegressor
        )
        forest = forest_class(n_estimators=100, random_state=0, n_jobs=args.threads)
        for name, model in [("DUMMY", dummy), ("RF", forest)]:
            print(f"Evaluating {name} on all configured folds...", flush=True)
            with threadpool_limits(limits=args.threads):
                frame = lb.evaluate_and_add(name, model, str(config_path), task=args.task)
            frames.append(frame)
        lb._added_models.clear()
    print(f"Evaluating {args.name} on all configured folds...", flush=True)
    with threadpool_limits(limits=args.threads):
        frames.append(
            lb.evaluate_and_add(
                args.name, create_model(args.task), str(config_path), task=args.task
            )
        )
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    (output / "metrics").mkdir()
    for task, frame in [("classification", lb._clf_metrics), ("regression", lb._reg_metrics)]:
        if not frame.empty:
            frame.to_csv(output / "metrics" / f"{task}_metrics.csv", index=False)
    ranking = lb.rank(args.task)
    ranking.to_csv(output / "leaderboard.csv", index=False)
    lb.plot(args.task).savefig(output / "leaderboard.png", dpi=200, bbox_inches="tight")
    lb.plot(args.task).savefig(output / "leaderboard.svg", bbox_inches="tight")
    table = ranking[["Rank", "Model", "Elo", "Elo_lo", "Elo_hi", "# Targets"]].to_html(
        index=False, border=0
    )
    (output / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>TabBench-Bio · Your model</title><style>body{font:17px system-ui;color:#172033;"
        "background:#f4f6fa;margin:0}main{max-width:1050px;margin:40px auto;padding:32px;background:white;"
        "border-radius:18px}h1{font-size:36px}p{color:#475569;line-height:1.6}img{width:100%;height:auto}"
        "table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px;text-align:left;"
        "border-bottom:1px solid #e2e8f0}th{background:#eff6ff}a{color:#2563eb}</style><main>"
        f"<h1>TabBench-Bio · {html.escape(args.name)}</h1><p>{html.escape(mode)}.</p>"
        "<p>Fold-level Bradley–Terry Elo; Random Forest = 1000. Intervals resample datasets. "
        "A small dataset subset is a smoke test, not a benchmark-wide claim. Baseline coverage may differ.</p>"
        '<img src="leaderboard.svg" alt="Elo ratings and 95 percent target-bootstrap intervals">'
        + table
        + '<p><a href="leaderboard.csv">Leaderboard CSV</a> · '
        '<a href="fold_metrics.csv">Fold metrics</a> · <a href="leaderboard.png">PNG</a> · '
        '<a href="config.json">Configuration</a></p></main></html>',
        encoding="utf-8",
    )
    print(lb.summary(args.task))
    print(f"\nReport: {output / 'report.html'}")


if __name__ == "__main__":
    main()
