"""Generate an overview and modality reports from saved benchmark results."""

import html
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path, PureWindowsPath

import pandas as pd
from threadpoolctl import threadpool_limits

from tabbench_bio import Leaderboard
from tabbench_bio.bio.cache import default_bio_cache_dir
from tabbench_bio.coverage import coverage_counts
from tabbench_bio.dashboard import (
    CELL_RE,
    cell_label,
    cell_sort_key,
    load_model_registry,
    model_meta,
)
from tabbench_bio.dashboard_data import dataset_id, dataset_metadata, read_inputs
from tabbench_bio.elo import DEFAULT_ANCHOR, compute_elo, fold_scores
from tabbench_bio.evaluate import read_saved_evaluation
from tabbench_bio.evaluation_report import (
    SCRIPT,
    STYLE,
    _table,
    ranking_cards,
    write_report,
)
from tabbench_bio.io_utils import _atomic_path
from tabbench_bio.model_metadata import hidden_model_ids
from tabbench_bio.plots import plot_elo_ranking

REFERENCE_CELL = "cap_10000_n100"
METRICS = {
    "f1_macro": "Macro-F1",
    "matthews_corrcoef": "MCC",
    "balanced_accuracy": "Balanced accuracy",
    "roc_auc": "ROC AUC",
    "rmse": "RMSE",
}
PAIR_COUNTS = ["Pairs", "Ahead", "Ties", "Behind"]
UNIT = ["cell", "key"]


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def _saved_inputs(directory, focus, cells):
    config, frames, status = read_saved_evaluation(directory)
    saved_report = directory / "report" / "report_config.json"
    settings = (
        json.loads(saved_report.read_text(encoding="utf-8")) if saved_report.is_file() else {}
    )
    if focus is None:
        if "name" in settings:
            focus = settings["name"]
        else:
            models = pd.read_csv(directory / "fold_metrics.csv", usecols=["model"])[
                "model"
            ].unique()
            assert len(models) == 1, "Specify --model for a saved run with several evaluated models"
            focus = models[0]
    cell = PureWindowsPath(config["output_dir"]).name if "output_dir" in config else ""
    if not CELL_RE.fullmatch(cell):
        cell = settings["cell"]
    assert not cells or cells == [cell], f"Saved evaluation only contains {cell}"
    excluded = set(settings["exclude_models"]) if "exclude_models" in settings else set()
    configured = set(config["datasets_classification"] + config["datasets_regression"])
    for task, frame in [*frames.items(), ("status", status)]:
        frame["cell"] = cell
        frame["dataset"] = (
            frame["key"].map(lambda key: dataset_id(key, configured)) if len(frame) else []
        )
        if task != "status" and not frame.empty:
            frames[task] = frame.loc[~frame["model"].isin(excluded)]
    return {cell: config}, frames, status.loc[~status["model"].isin(excluded)], str(focus)


def _metric_scores(frame, metric):
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()
    valid = frame[metric].notna()
    if "imputed" in frame.columns:
        imputed = frame["imputed"].eq(True).fillna(False)
        clean = ~imputed.groupby([frame[column] for column in [*UNIT, "model"]]).transform("any")
    else:
        clean = pd.Series(True, index=frame.index)
    subset = frame.loc[valid]
    if metric == "rmse":
        scores = fold_scores(None, subset)
    else:
        scores = fold_scores(subset, None, clf_metric=metric)
    return scores.assign(
        cell=subset["cell"].to_numpy(),
        dataset=subset["dataset"].to_numpy(),
        metric=metric,
        clean=clean.loc[valid].to_numpy(),
    )


def _paired_summary(scores, focus):
    shared = [*UNIT, "seed"]
    focus_scores = scores.loc[scores["model"].eq(focus), [*shared, "score"]]
    pairs = scores.loc[~scores["model"].eq(focus)].merge(
        focus_scores.rename(columns={"score": "focus_score"}), on=shared, validate="many_to_one"
    )
    deltas = (
        pairs.assign(delta=pairs["focus_score"] - pairs["score"])
        .groupby(["model", "metric", *UNIT])["delta"]
        .mean()
        .round(12)
    )
    counts = (
        pd.DataFrame({"Ahead": deltas.gt(0), "Ties": deltas.eq(0), "Behind": deltas.lt(0)})
        .groupby(level="model")
        .sum()
    )
    counts.insert(0, "Pairs", counts.sum(axis=1))
    counts.loc[focus, "Pairs"] = len(focus_scores[UNIT].drop_duplicates())
    raw_deltas = deltas.where(deltas.index.get_level_values("metric") != "rmse", -deltas)
    medians = (
        raw_deltas.groupby(level=["model", "metric"])
        .median()
        .unstack("metric")
        .rename(columns={metric: f"Median Δ {label}" for metric, label in METRICS.items()})
    )
    return counts.join(medians).rename_axis("model_id")


def _score_heatmaps(scores, models, labels, dataset_labels, optional):
    mean_by_cell = scores.groupby(["metric", *UNIT, "dataset", "model"])["score"].mean()
    averaged = mean_by_cell.groupby(["metric", "key", "dataset", "model"]).mean()
    heatmaps = []
    for metric, values in averaged.groupby(level="metric", sort=False):
        means = values.droplevel("metric").unstack("model")
        ranks = means.rank(axis=1, ascending=True, pct=True)
        ordered = [model for model in models if model in means.columns]
        rows = []
        for key, dataset in means.index:
            name = dataset_labels[dataset]
            if key not in {dataset, f"{dataset}_0"}:
                name += f" · target {key.rsplit('_', 1)[-1]}"
            cells = []
            for model in ordered:
                value = means.loc[(key, dataset), model]
                if pd.isna(value):
                    cells.append(None)
                else:
                    raw_value = -value if metric == "rmse" else value
                    cells.append([f"{raw_value:.3f}", round(ranks.loc[(key, dataset), model] * 43)])
            rows.append({"name": name, "cells": cells})
        heatmaps.append(
            {
                "caption": METRICS[metric],
                "columns": [
                    {"name": labels[model], "optional": model in optional} for model in ordered
                ],
                "rows": rows,
            }
        )
    return heatmaps


def _cell_text(value):
    if pd.isna(value):
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _table_data(frame, focus_label, optional_names):
    return {
        "columns": list(frame.columns),
        "rows": [[_cell_text(value) for value in row] for row in frame.itertuples(index=False)],
        "optional": [name in optional_names for name in frame["Model"]],
        "focus": [name == focus_label for name in frame["Model"]],
    }


def _mark_optional(table, names):
    """Tag server-rendered table rows of default-hidden models for the page toggle."""
    for name in names:
        row = "<tr>\n      <td>" + html.escape(name) + "</td>"
        tagged = '<tr class="optional-model">\n      <td>' + html.escape(name) + "</td>"
        table = table.replace(row, tagged)
    return table


def _elo_svgs(ranking, focus, title, optional, directory, prefix, stem):
    """Write the Elo chart for all models and, when it differs, for the default models."""
    rated = ranking.dropna(subset=["Elo"])
    if rated.empty:
        return None
    paths = {"alt": f"{title} Elo ranking", "default": None}
    shown = rated.loc[~rated["model_id"].isin(optional)]
    versions = [("all", rated)] + ([("default", shown)] if len(shown) < len(rated) else [])
    for name, frame in versions:
        filename = f"{stem}-{name}.svg"
        figure = plot_elo_ranking(frame, scope=title, highlighted=(focus,))
        figure.savefig(directory / filename, format="svg", bbox_inches="tight")
        paths[name] = f"{prefix}/{filename}"
    return paths


def _modality_cards(ranking, focus, targets, scored_folds):
    focus_row = ranking.loc[ranking["model_id"].eq(focus)].iloc[0]
    ranks = ranking["Elo"].rank(method="min", ascending=False).astype("Int64")
    return ranking_cards(
        focus_row,
        rank=ranks.loc[focus_row.name],
        model_count=len(ranking),
        targets=targets,
        scored_folds=scored_folds,
    )


def _render_panel(job, *, focus, labels, dataset_labels, n_boot, optional):
    cell, metric, scores, directory, prefix = job
    elo_scores = scores
    if cell == "pooled":
        elo_scores = scores.assign(seed=scores["cell"] + "/" + scores["seed"].astype(str))
    comparable = elo_scores.groupby(["key", "seed"])["model"].transform("size").gt(1)
    if DEFAULT_ANCHOR not in set(elo_scores.loc[comparable, "model"]):
        elo_scores = elo_scores.iloc[:0]
    ranking = compute_elo(elo_scores, n_boot=n_boot)
    ids = sorted(scores["model"].unique())
    ranking = ranking.merge(pd.DataFrame({"model_id": ids}), how="right", on="model_id")
    ranking = ranking.sort_values(["Elo", "model_id"], ascending=[False, True], ignore_index=True)
    deltas = [f"Median Δ {METRICS[key]}" for key in scores["metric"].unique()]
    columns = [*PAIR_COUNTS, *deltas]
    full = ranking.join(_paired_summary(scores, focus).reindex(columns=columns), on="model_id")
    successful = _paired_summary(scores.loc[scores["clean"]], focus).reindex(columns=columns)
    full = full.join(successful.add_prefix("Successful "), on="model_id")
    successful_columns = [f"Successful {column}" for column in PAIR_COUNTS]
    for count_columns in (PAIR_COUNTS, successful_columns):
        full[count_columns] = full[count_columns].fillna(0).astype("Int64")
        full.loc[full["model_id"].eq(focus), count_columns[1:]] = pd.NA
    full = full.assign(Model=full["model_id"].map(labels))
    display = full.assign(
        **{
            "Elo": full["Elo"].astype("Int64"),
            "Elo targets": full["n_targets"].astype("Int64"),
            "95% CI": [
                f"{lo:.0f}–{hi:.0f}" if pd.notna(lo) else "—"
                for lo, hi in zip(full["Elo_lo"], full["Elo_hi"], strict=True)
            ],
        }
    )
    visible = full.loc[~full["model_id"].isin(optional), "model_id"]
    selected_models = list(dict.fromkeys([*visible.head(5), *full["model_id"].head(5), focus]))
    optional_names = {labels[model] for model in optional & set(full["model_id"])}
    score_note = (
        "Scores average each model’s available cell means."
        if cell == "pooled"
        else "Scores average the scored outer folds."
    )
    mixed = metric != "rmse" and scores["metric"].eq("rmse").any()
    title = METRICS[metric] + (" + RMSE" if mixed else "")
    focus_keys = scores.loc[scores["model"].eq(focus), "key"]
    successful_table = display[["Model", *[f"Successful {column}" for column in columns]]]
    panel = {
        "cell": cell,
        "metric": metric,
        "title": title,
        "plot": _elo_svgs(full, focus, title, optional, directory, prefix, f"{cell}-{metric}"),
        "cards": _modality_cards(ranking, focus, focus_keys.nunique(), len(focus_keys)),
        "table": _table_data(
            display[["Model", "Elo", "95% CI", "Elo targets", *columns]],
            labels[focus],
            optional_names,
        ),
        "successful": _table_data(
            successful_table.rename(columns=lambda column: column.removeprefix("Successful ")),
            labels[focus],
            optional_names,
        ),
        "note": score_note,
        "heatmaps": _score_heatmaps(scores, selected_models, labels, dataset_labels, optional),
    }
    export = full.assign(cell=cell, metric=metric, includes_regression=mixed or metric == "rmse")
    return panel, export


def _document(title, content, script=""):
    escaped = html.escape(title)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escaped} · TabBench-Bio</title><style>{STYLE}{REPORT_STYLE}</style></head>"
        f'<body><div class="site-header"><div class="wordmark">TabBench-<strong>Bio</strong></div></div>'
        f"<main>{content}</main><script>{SCRIPT}{script}</script></body></html>"
    )


def _write_page(path, content):
    with _atomic_path(path) as temporary:
        temporary.write_text(content, encoding="utf-8")


def _modality_jobs(frames, *, modality, focus, reference, directory, prefix):
    classification, regression = frames["classification"], frames["regression"]
    scoring = {
        metric: _metric_scores(regression if metric == "rmse" else classification, metric)
        for metric in METRICS
        if metric != "roc_auc" or modality != "Gene expression"
    }
    cells = sorted(
        {
            cell
            for frame in scoring.values()
            if not frame.empty
            for cell in frame.loc[frame["model"].eq(focus), "cell"]
        },
        key=cell_sort_key,
    )
    assert cells, f"{modality} has no reportable scores for {focus}"
    default_cell = reference if reference in cells else cells[0]
    scoring = {
        metric: frame.loc[frame["cell"].isin(cells)]
        for metric, frame in scoring.items()
        if not frame.empty
    }
    jobs = []
    for cell in [*cells, *(["pooled"] if len(cells) > 1 else [])]:
        available = {
            metric: frame if cell == "pooled" else frame.loc[frame["cell"].eq(cell)]
            for metric, frame in scoring.items()
            if not frame.empty
        }
        for metric, scores in available.items():
            if not scores["model"].eq(focus).any():
                continue
            if metric != "rmse" and "rmse" in available:
                scores = pd.concat([scores, available["rmse"]], ignore_index=True)
            jobs.append((cell, metric, scores, directory, prefix))
    return cells, default_cell, jobs


def _modality_report(modality, plan, results, status, output, *, focus, labels, optional):
    filename = _slug(modality)
    cells, default_cell, jobs = plan
    default_metric = next(job[1] for job in jobs if job[0] == default_cell)
    panels = [panel for panel, _ in results]
    exports = [export for _, export in results]
    data = {"panels": panels}
    with _atomic_path(output / filename / "data.js") as temporary:
        temporary.write_text(
            "window.REPORT_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
            encoding="utf-8",
        )
    table = pd.concat(exports, ignore_index=True)
    with _atomic_path(output / f"{filename}.csv") as temporary:
        table.to_csv(temporary, index=False)

    cell_options = [(cell, cell_label(cell)) for cell in cells] + (
        [("pooled", "Pooled (all cells)")] if len(cells) > 1 else []
    )
    metric_options = [
        (metric, label) for metric, label in METRICS.items() if metric in set(table["metric"])
    ]
    controls = _control("Evaluation cell", "cell", cell_options, default_cell)
    controls += _control("Metric", "metric", metric_options, default_metric)
    coverage = coverage_counts(status.loc[status["cell"].isin(cells)])
    optional_names = [labels[model] for model in optional & set(table["model_id"])]
    if optional_names:
        controls += (
            '<label class="toggle"><input type="checkbox" id="all-models"> All models</label>'
        )
    coverage.insert(0, "Model", coverage.pop("model_id").map(labels))
    content = (
        '<nav><a href="../report.html">Report overview</a></nav>'
        f'<header><p class="eyebrow">{html.escape(labels[focus])}</p><h1>{html.escape(modality)}</h1></header>'
        '<div id="modality-summary" class="modality-summary"></div>'
        f'<section><div class="controls">{controls}</div>'
        '<p class="muted">Elo compares scored folds with Random Forest anchored at 1,000. '
        "The main table includes the benchmark’s constant-baseline scores for failed fits. "
        "Ahead, ties and behind count shared dataset targets at each selected setting, using "
        "the mean difference on shared folds. Δ is the focus model’s score minus the comparator’s. "
        "Successful-only pairs exclude a setting when "
        "either model has an imputed fold for that target. Pooled Elo weights each target "
        "equally across its included folds and cells.</p>"
        '<div id="panel"></div></section>'
        f'<script src="{filename}/data.js"></script>'
        '<section><h2>Coverage across reported cells</h2><p class="muted">Counts are recorded folds.</p>'
        + _mark_optional(_table(coverage, labels[focus]), optional_names)
        + f'<p><a href="{filename}.csv" download>Download Elo and paired comparisons</a></p></section>'
    )
    _write_page(output / f"{filename}.html", _document(modality, content, SELECT_SCRIPT))
    reference_row = table.loc[
        table["cell"].eq(default_cell)
        & table["metric"].eq(default_metric)
        & table["model_id"].eq(focus)
    ].iloc[0]
    return {
        "name": modality,
        "file": f"categories/{filename}.html",
        "cell": cell_label(default_cell),
        "metric": METRICS[default_metric]
        + (" + RMSE" if default_metric != "rmse" and reference_row["includes_regression"] else ""),
        "elo": f"{reference_row['Elo']:.0f}" if pd.notna(reference_row["Elo"]) else "—",
        "cells": len(cells),
    }


def _control(label, name, options, default):
    if len(options) == 1:
        return f"<p>{label} <strong>{options[0][1]}</strong></p>"
    rendered = [
        f'<option value="{value}"{" selected" if value == default else ""}>{text}</option>'
        for value, text in options
    ]
    return f'<label>{label}<select id="{name}">{"".join(rendered)}</select></label>'


def _modality_section(summary):
    rows = "".join(
        f'<tr><td><a href="{row["file"]}">{html.escape(row["name"])}</a></td>'
        f"<td>{row['cell']}</td><td>{row['metric']}</td><td>{row['elo']}</td><td>{row['cells']}</td></tr>"
        for row in summary
    )
    return (
        '<section id="modalities"><h2>Modalities</h2><div class="scroll"><table><thead><tr>'
        '<th scope="col">Modality</th><th scope="col">Initial cell</th><th scope="col">Metric</th>'
        '<th scope="col">Elo</th><th scope="col">Cells with scores</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def _write_overview(
    output, configs, frames, status, *, focus, labels, reference, source, summary, n_boot
):
    focus_cells = {
        cell
        for frame in frames.values()
        if not frame.empty
        for cell in frame.loc[frame["model"].eq(focus), "cell"]
    }
    cell = reference if reference in focus_cells else min(focus_cells, key=cell_sort_key)
    scoped = {task: frame.loc[frame["cell"].eq(cell)] for task, frame in frames.items()}
    focus_tasks = [
        task for task, frame in scoped.items() if not frame.empty and frame["model"].eq(focus).any()
    ]
    task = "overall" if len(focus_tasks) == 2 else focus_tasks[0]
    regression = scoped["regression"] if task != "classification" else pd.DataFrame()
    classification = scoped["classification"] if task != "regression" else pd.DataFrame()
    display = pd.DataFrame({"model_id": list(labels), "Model": list(labels.values())})
    leaderboard = Leaderboard(regression, classification, display)
    leaderboard._added_models = [focus]
    leaderboard._ratings[task] = compute_elo(fold_scores(classification, regression), n_boot=n_boot)
    output.mkdir(parents=True, exist_ok=True)
    write_report(
        leaderboard,
        leaderboard.rank(task),
        output,
        focus=focus,
        task=task,
        cell=cell,
        config=configs[cell],
        status=status.loc[status["cell"].eq(cell)],
        reference=source,
        modality_section=_modality_section(summary),
        bootstrap_rounds=n_boot,
    )
    return output / "report.html"


def generate_model_report(args):
    assert args.workers > 0 and args.bootstrap_rounds > 0
    cells = list(dict.fromkeys(args.cell)) if args.cell else None
    if args.evaluation:
        directory = args.evaluation.resolve()
        configs, frames, status, focus = _saved_inputs(directory, args.model, cells)
        output = args.output or directory / "category-report"
        assert output.resolve() != (directory / "report").resolve(), (
            "Choose a separate model report directory"
        )
        source = f"Saved evaluation · {directory.name}"
    else:
        assert args.model, "--model is required with --database"
        database = args.database.resolve()
        cache = (
            args.metric_cache or default_bio_cache_dir().parent / "fold_metrics.sqlite"
        ).resolve()
        assert cache != database, "Metric cache must be separate from results"
        configs, views, status, _ = read_inputs(
            database, cells, workers=args.workers, metric_cache=cache
        )
        frames, focus = views["strict"], args.model
        output = args.output or Path("results/model-reports") / _slug(focus)
        source = f"Benchmark SQLite · {database.parent.name}/{database.name}"

    populated = [frame[["model", "dataset"]] for frame in frames.values() if not frame.empty]
    assert populated, "The selected results contain no scored folds"
    scored = pd.concat(populated)
    focus_datasets = set(scored.loc[scored["model"].eq(focus), "dataset"])
    assert focus_datasets, (
        f"No scored results for model {focus!r}. Available models: {', '.join(sorted(scored['model'].unique()))}"
    )
    metadata = dataset_metadata(configs)
    modalities = {row["modality"] for row in metadata if row["dataset_id"] in focus_datasets}
    assert len({_slug(modality) for modality in modalities}) == len(modalities)
    registry = load_model_registry(configs.values())
    labels = {
        model: str(model_meta(model, registry)["display"])
        for model in set(scored["model"]) | set(status["model"])
    }
    dataset_labels = {row["dataset_id"]: row["display_name"] for row in metadata}
    optional = frozenset((set(hidden_model_ids(registry)) | {"AUTOGLUON"}) - {focus})
    reference = args.reference_cell or REFERENCE_CELL
    plans = {}
    for modality in sorted(modalities):
        datasets = {row["dataset_id"] for row in metadata if row["modality"] == modality}
        directory = output / "categories" / _slug(modality)
        if directory.exists():
            shutil.rmtree(directory)
        (directory / "svg").mkdir(parents=True)
        plans[modality] = (
            _modality_jobs(
                {
                    task: frame.loc[frame["dataset"].isin(datasets)]
                    for task, frame in frames.items()
                },
                modality=modality,
                focus=focus,
                reference=reference,
                directory=directory / "svg",
                prefix=f"{_slug(modality)}/svg",
            ),
            status.loc[status["dataset"].isin(datasets)],
        )
    render = partial(
        _render_panel,
        focus=focus,
        labels=labels,
        dataset_labels=dataset_labels,
        n_boot=args.bootstrap_rounds,
        optional=optional,
    )
    jobs = [(modality, job) for modality, (plan, _) in plans.items() for job in plan[2]]
    print(f"Rendering {len(jobs)} panels with {args.workers} worker(s)", flush=True)

    summary = []
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(limits=1))
        executor = (
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=args.workers, initializer=threadpool_limits, initargs=(1,)
                )
            )
            if args.workers > 1
            else None
        )
        # Submit every panel at once, largest score tables first, so no worker idles.
        order = sorted(range(len(jobs)), key=lambda index: -len(jobs[index][1][2]))
        futures = (
            {index: executor.submit(render, jobs[index][1]) for index in order} if executor else {}
        )
        for modality, (plan, modality_status) in plans.items():
            indices = [index for index, (name, _) in enumerate(jobs) if name == modality]
            results = [
                futures[index].result() if executor else render(jobs[index][1]) for index in indices
            ]
            print(f"Writing {modality} report", flush=True)
            summary.append(
                _modality_report(
                    modality,
                    plan,
                    results,
                    modality_status,
                    output / "categories",
                    focus=focus,
                    labels=labels,
                    optional=optional,
                )
            )

        report = _write_overview(
            output,
            configs,
            frames,
            status,
            focus=focus,
            labels=labels,
            reference=reference,
            source=source,
            summary=summary,
            n_boot=args.bootstrap_rounds,
        )
    print(f"Report: {report.resolve()}")
    return report


REPORT_STYLE = """
a{color:var(--accent);text-underline-offset:3px}.wordmark{gap:0;font-size:20px;font-weight:700}
.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:end;margin-bottom:18px}
.controls p{margin:0}.controls label{font-size:13px;font-weight:600}
.controls select{display:block;font:inherit;margin-top:5px;padding:7px 10px;border:1px solid var(--line);
border-radius:5px;background:var(--surface);color:var(--ink)}
.scores{margin-bottom:14px}.scores caption{text-align:left;font-weight:600;margin:4px 0}
.scores th{cursor:default}.scores td{background:color-mix(in srgb,var(--accent) var(--tint,0%),white)}
.controls .toggle{display:flex;gap:6px;align-items:center;padding-bottom:8px}
body:not(.show-all) .optional-model,body:not(.show-all) .all-models,body.show-all .default-models{display:none}
"""

SELECT_SCRIPT = """
const DATA = window.REPORT_DATA;
const cell = document.getElementById("cell");
const metric = document.getElementById("metric");
const summary = document.getElementById("modality-summary");
const target = document.getElementById("panel");
function element(tag, attributes = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function dataTable(data) {
  const wrapper = element("div", {class: "scroll"});
  const table = element("table", {class: "dataframe sortable"});
  const head = table.createTHead().insertRow();
  for (const column of data.columns) head.append(element("th", {}, column));
  const body = table.createTBody();
  data.rows.forEach((row, index) => {
    const line = body.insertRow();
    if (data.optional[index]) line.className = "optional-model";
    row.forEach((value, column) => {
      line.append(element("td", column === 0 && data.focus[index] ? {class: "focus"} : {}, value));
    });
  });
  wrapper.append(table);
  return wrapper;
}
function heatmap(data) {
  const wrapper = element("div", {class: "scroll"});
  const table = element("table", {class: "scores"});
  table.append(element("caption", {}, data.caption));
  const head = table.createTHead().insertRow();
  head.append(element("th", {scope: "col"}, "Dataset"));
  for (const column of data.columns) {
    head.append(element("th", column.optional ? {scope: "col", class: "optional-model"} : {scope: "col"}, column.name));
  }
  const body = table.createTBody();
  for (const row of data.rows) {
    const line = body.insertRow();
    line.append(element("th", {scope: "row"}, row.name));
    row.cells.forEach((value, column) => {
      const attributes = data.columns[column].optional ? {class: "optional-model"} : {};
      if (value) attributes.style = `--tint:${value[1]}%`;
      line.append(element("td", attributes, value ? value[0] : "—"));
    });
  }
  wrapper.append(table);
  return wrapper;
}
function plot(data) {
  if (!data) return element("p", {class: "muted"}, "Elo is unavailable without a scored Random Forest comparison.");
  const box = element("div");
  if (data.default) {
    const shown = element("div", {class: "default-models"});
    shown.append(element("img", {alt: data.alt, src: data.default}));
    box.append(shown);
  }
  const all = element("div", data.default ? {class: "all-models"} : {});
  all.append(element("img", {alt: data.alt, src: data.all}));
  box.append(all);
  return box;
}
function render(panel) {
  summary.innerHTML = panel.cards;
  const details = element("details");
  details.append(element("summary", {}, "Successful fits only"), dataTable(panel.successful));
  target.replaceChildren(
    element("h3", {}, panel.title),
    plot(panel.plot),
    element("h4", {}, "Paired comparison and Elo"),
    dataTable(panel.table),
    details,
    element("h4", {}, "Dataset scores"),
    element("p", {class: "muted"}, panel.note + " Darker cells indicate higher ranks within a dataset."),
    ...panel.heatmaps.map(heatmap),
  );
  enhanceTables(target);
}
function updateReport() {
  const selectedCell = cell ? cell.value : DATA.panels[0].cell;
  const available = DATA.panels.filter(panel => panel.cell === selectedCell);
  if (metric) {
    const options = new Set(available.map(panel => panel.metric));
    for (const option of metric.options) option.disabled = !options.has(option.value);
    if (!options.has(metric.value)) metric.value = available[0].metric;
  }
  const selectedMetric = metric ? metric.value : available[0].metric;
  render(available.find(panel => panel.metric === selectedMetric));
}
document.getElementById("all-models")?.addEventListener("change", (event) => {
  document.body.classList.toggle("show-all", event.target.checked);
});
cell?.addEventListener("change", updateReport);
metric?.addEventListener("change", updateReport);
updateReport();
"""
