"""Website dashboard exports computed from fold-level SQLite results."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import math
import re
import shutil
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from tabbench_bio.bio.loaders.tdc import DATAVERSE_URL, ENDPOINTS
from tabbench_bio.dashboard_data import dataset_metadata, progress_summary, read_inputs
from tabbench_bio.elo import DEFAULT_N_BOOT, compute_elo, fold_scores
from tabbench_bio.io_utils import atomic_write_json
from tabbench_bio.model_constraints import REGULAR_MAX_FEATURES
from tabbench_bio.seeds import get_seeds
from tabbench_bio.web_metadata import write_agent_metadata

PACKAGE_ROOT = Path(__file__).parent
DOMAIN_ELO_IMPLEMENTATION_FILES = (
    PACKAGE_ROOT / "elo.py",
    PACKAGE_ROOT / "_vendor" / "tabarena_elo_utils.py",
)
TRAINING_DATA_OVERLAP = {"TABDPT"}

MODEL_CATEGORY = {
    "DUMMY": "Baseline",
    "KNN": "Traditional ML",
    "LR": "Traditional ML",
    "RF": "Tree-based",
    "XT": "Tree-based",
    "CAT": "Gradient Boosting",
    "GBM": "Gradient Boosting",
    "XGB": "Gradient Boosting",
    "NN_TORCH": "Deep Learning",
    "REALMLP": "Deep Learning",
    "TABM": "Deep Learning",
    "MITRA": "Tabular Foundation",
    "REALTABPFN-V2": "Tabular Foundation",
    "REALTABPFN-V2.5": "Tabular Foundation",
    "TABPFN-V3": "Tabular Foundation",
    "TABPFN-V3.5": "Tabular Foundation",
    "TABPFN-WIDE": "Tabular Foundation",
    "TABPFN-WIDE-5K-NE3": "Tabular Foundation",
    "TABFM": "Tabular Foundation",
    "TABDPT": "Tabular Foundation",
    "TABICL": "Tabular Foundation",
    "AUTOGLUON": "AutoML",
}


MODEL_DISPLAY = {
    "DUMMY": "Constant",
    "KNN": "KNN",
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XT": "Extra Trees",
    "CAT": "CatBoost",
    "GBM": "LightGBM",
    "XGB": "XGBoost",
    "NN_TORCH": "MLP",
    "REALMLP": "RealMLP",
    "TABM": "TabM",
    "MITRA": "MITRA",
    "REALTABPFN-V2": "RealTabPFN v2",
    "REALTABPFN-V2.5": "RealTabPFN 2.5",
    "TABPFN-V3": "TabPFN 3",
    "TABPFN-V3.5": "TabPFN 3.5",
    "TABPFN-WIDE": "TabPFN Wide (8k)",
    "TABPFN-WIDE-5K-NE3": "TabPFN Wide 5k (ne3)",
    "TABFM": "TabFM",
    "TABDPT": "TabDPT",
    "TABICL": "TabICL",
    "AUTOGLUON": "AutoGluon",
}


CATEGORY_COLORS = {
    "Traditional ML": "#2563eb",
    "Tree-based": "#059669",
    "Gradient Boosting": "#d97706",
    "Deep Learning": "#dc2626",
    "Tabular Foundation": "#7c3aed",
    "AutoML": "#111827",
    "Baseline": "#9ca3af",
}


ELO_METRICS = (
    "f1_macro",
    "matthews_corrcoef",
    "balanced_accuracy",
    "roc_auc",
)


DOMAIN_ELO_DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "scipy")


DOMAIN_ELO_OUTPUT_FIELDS = (
    "cell",
    "cell_label",
    "domain",
    "metric",
    "model_id",
    "Elo",
    "Elo_lo",
    "Elo_hi",
    "n_targets",
)


CELL_RE = re.compile(r"^cap_(full|\d+)(?:_n(\d+))?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    atomic_write_json(path, clean_json(payload))


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return None if pd.isna(value) else value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_meta(model_id: str) -> dict[str, object]:
    category = MODEL_CATEGORY[model_id] if model_id in MODEL_CATEGORY else "Custom"
    return {
        "id": model_id,
        "display": MODEL_DISPLAY[model_id] if model_id in MODEL_DISPLAY else model_id,
        "category": category,
        "color": CATEGORY_COLORS[category] if category in CATEGORY_COLORS else "#64748b",
        "regular_max_features": REGULAR_MAX_FEATURES[model_id]
        if model_id in REGULAR_MAX_FEATURES
        else None,
        "training_data_overlap": model_id in TRAINING_DATA_OVERLAP,
    }


def dataset_source_url(source: str, fetch_id: str) -> str | None:
    source = source.lower()
    if source == "openml":
        return f"https://www.openml.org/search?type=data&sort=runs&id={fetch_id}&status=active"
    if source in {"geo", "geo_matrix"}:
        accession = fetch_id.split("@", maxsplit=1)[0]
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
    if source == "tcga":
        project = fetch_id.split("/", maxsplit=1)[0]
        return f"https://portal.gdc.cancer.gov/projects/{project}"
    if source == "mgnify":
        return f"https://www.ebi.ac.uk/metagenomics/studies/{fetch_id}"
    if source == "kaggle":
        return f"https://www.kaggle.com/datasets/{fetch_id}"
    if source == "metagenomics":
        return "https://github.com/segatalab/metaml"
    if source == "tdc":
        assert fetch_id in ENDPOINTS, f"Unknown frozen TDC endpoint: {fetch_id}"
        return DATAVERSE_URL.format(file_id=ENDPOINTS[fetch_id].file_id)
    if source == "gp":
        return "https://doi.org/10.5061/dryad.xksn02vb9"
    if source == "chembl":
        return f"https://www.ebi.ac.uk/chembl/explore/target/{fetch_id}"
    return None


def add_dataset_performance(
    datasets: list[dict[str, object]],
    models: dict[str, dict[str, str]],
    reference_cell: str,
    reference_label: str,
    classification: list[dict],
    regression: pd.DataFrame,
) -> None:
    classification_metrics = (
        ("f1_macro", "Macro-F1", "high"),
        ("matthews_corrcoef", "MCC", "high"),
        ("balanced_accuracy", "Balanced accuracy", "high"),
        ("roc_auc", "ROC-AUC", "high"),
    )
    regression_metrics = (
        ("rmse", "RMSE", "low"),
        ("mae", "MAE", "low"),
        ("r2", "R²", "high"),
    )
    reference_cap, reference_samples = parse_cell(reference_cell)

    def closest_cell(cells: set[str]) -> str:
        def distance(cell: str) -> tuple[float, float, float, float]:
            cap, samples = parse_cell(cell)
            return (
                0 if cap == reference_cap else 1,
                abs((cap or math.inf) - (reference_cap or math.inf)),
                0 if samples == reference_samples else 1,
                abs((samples or math.inf) - (reference_samples or math.inf)),
            )

        return min(cells, key=distance)

    def cell_label(cell: str) -> str:
        cap, samples = parse_cell(cell)
        cap_label = "all" if cap is None else f"{cap:,}"
        sample_label = "all" if samples is None else f"{samples:,}"
        return f"p={cap_label}, n={sample_label}"

    for dataset in datasets:
        is_classification = dataset["task"] == "Classification"
        metric_specs = classification_metrics if is_classification else regression_metrics
        dataset_id = str(dataset["dataset_id"])
        available_cells = (
            {str(row["cell"]) for row in classification if row["dataset"] == dataset_id}
            if is_classification
            else set(regression.loc[regression["dataset"] == dataset_id, "cell"].astype(str))
        )
        selected_cell = (
            reference_cell
            if reference_cell in available_cells or not available_cells
            else closest_cell(available_cells)
        )
        performance_rows: list[dict[str, object]] = []
        if is_classification:
            source_rows = [
                row
                for row in classification
                if row["dataset"] == dataset_id and row["cell"] == selected_cell
            ]
            for source_row in source_rows:
                model_id = str(source_row["model"])
                values = {key: source_row[f"{key}__mean"] for key, _label, _better in metric_specs}
                performance_rows.append(
                    {
                        **models[model_id],
                        "values": values,
                        "folds": max(
                            int(source_row[f"{key}__count"] or 0)
                            for key, _label, _better in metric_specs
                        ),
                    }
                )
        else:
            source_rows = regression[
                (regression["dataset"] == dataset_id) & (regression["cell"] == selected_cell)
            ]
            for model_id, frame in source_rows.groupby("model", sort=False):
                values = {key: float(frame[key].mean()) for key, _label, _better in metric_specs}
                performance_rows.append(
                    {
                        **models[str(model_id)],
                        "values": values,
                        "folds": int(frame[metric_specs[0][0]].count()),
                    }
                )

        ranking_key, _ranking_label, ranking_direction = metric_specs[0]
        performance_rows.sort(
            key=lambda row: (
                row["values"][ranking_key] is None,
                (
                    -float(row["values"][ranking_key])
                    if ranking_direction == "high" and row["values"][ranking_key] is not None
                    else (
                        float(row["values"][ranking_key])
                        if row["values"][ranking_key] is not None
                        else math.inf
                    )
                ),
            )
        )
        dataset["performance"] = {
            "cell": selected_cell,
            "cell_label": (
                reference_label if selected_cell == reference_cell else cell_label(selected_cell)
            ),
            "is_reference_cell": selected_cell == reference_cell,
            "metrics": [
                {"key": key, "label": label, "better": better}
                for key, label, better in metric_specs
            ],
            "models": performance_rows,
        }


def write_dataset_explorer(
    site_dir: Path, dashboard: dict, classification: list[dict], regression: pd.DataFrame
) -> None:
    """Export a compact registry and per-dataset scores for every nominal grid cell."""
    regression_means = (
        regression.groupby(["dataset", "cell", "model"])[["rmse", "mae", "r2"]].mean().reset_index()
    )
    output_dir = site_dir / "data" / "datasets"
    registry = []
    for dataset in dashboard["datasets"]:
        dataset_id = dataset["dataset_id"]
        metrics = dataset["performance"]["metrics"]
        rows = []
        if dataset["task"] == "Classification":
            for row in classification:
                if row["dataset"] != dataset_id:
                    continue
                rows.append(
                    {
                        "cell": row["cell"],
                        "model_id": row["model"],
                        "values": {
                            metric["key"]: row[f"{metric['key']}__mean"] for metric in metrics
                        },
                    }
                )
        else:
            for row in regression_means.loc[regression_means["dataset"] == dataset_id].to_dict(
                "records"
            ):
                rows.append(
                    {
                        "cell": row["cell"],
                        "model_id": row["model"],
                        "values": {metric["key"]: row[metric["key"]] for metric in metrics},
                    }
                )
        for row in rows:
            assert row["model_id"] in dashboard["models"], row
            row["values"] = {
                key: round(float(value), 8) if pd.notna(value) else None
                for key, value in row["values"].items()
            }
        assert len({(row["cell"], row["model_id"]) for row in rows}) == len(rows), dataset_id
        filename = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16] + ".json"
        metadata = {key: value for key, value in dataset.items() if key != "performance"}
        registry.append(
            {
                **metadata,
                "metrics": metrics,
                "default_cell": dataset["performance"]["cell"],
                "scores_file": filename,
            }
        )
        write_json(output_dir / filename, {"dataset_id": dataset_id, "scores": rows})
    write_json(
        output_dir / "index.json",
        {
            "meta": dashboard["meta"],
            "models": dashboard["models"],
            "cell_options": dashboard["cell_options"],
            "datasets": registry,
        },
    )


def _domain_elo_task(
    task: tuple[
        str,
        str,
        str,
        str,
        pd.DataFrame,
        int,
    ],
) -> list[dict[str, object]]:
    cell, cell_label_value, metric, domain, table, n_boot = task
    rows = []
    ranking = compute_elo(
        table,
        n_boot=n_boot,
        random_state=0,
    )
    for ranking_row in ranking.to_dict(orient="records"):
        rows.append(
            {
                "cell": cell,
                "cell_label": cell_label_value,
                "domain": domain,
                "metric": metric,
                "model_id": str(ranking_row["model_id"]),
                "Elo": int(ranking_row["Elo"]),
                "Elo_lo": int(ranking_row["Elo_lo"]),
                "Elo_hi": int(ranking_row["Elo_hi"]),
                "n_targets": int(ranking_row["n_targets"]),
            }
        )
    return rows


def domain_elo_algorithm_sha256() -> str:
    digest = hashlib.sha256(inspect.getsource(_domain_elo_task).encode("utf-8"))
    digest.update(json.dumps(DOMAIN_ELO_OUTPUT_FIELDS).encode("ascii"))
    for path in DOMAIN_ELO_IMPLEMENTATION_FILES:
        digest.update(path.read_bytes())
    for dependency in DOMAIN_ELO_DEPENDENCIES:
        digest.update(f"{dependency}=={version(dependency)}\n".encode("ascii"))
    return digest.hexdigest()


def fold_scores_sha256(scores: pd.DataFrame) -> str:
    ordered = scores.sort_values(["model", "key", "seed"])
    payload = [
        [str(row.model), str(row.key), int(row.seed), float(row.score).hex()]
        for row in ordered.itertuples(index=False)
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def domain_elo_task_input_sha256(
    cell: str,
    cell_label_value: str,
    metric: str,
    domain: str,
    bootstrap_rounds: int,
    table: pd.DataFrame,
) -> str:
    payload = {
        "cell": cell,
        "cell_label": cell_label_value,
        "metric": metric,
        "domain": domain,
        "bootstrap_rounds": bootstrap_rounds,
        "fold_scores_sha256": fold_scores_sha256(table),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def domain_elo_output_sha256(rows: list[dict[str, object]]) -> str:
    payload = [
        {field: row[field] for field in DOMAIN_ELO_OUTPUT_FIELDS}
        for row in sorted(
            rows,
            key=lambda row: tuple(str(row[field]) for field in DOMAIN_ELO_OUTPUT_FIELDS),
        )
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_domain_elo(
    elo: list[dict[str, object]],
    models: dict[str, dict[str, str]],
    datasets: list[dict[str, object]],
    complete_cells: list[str],
    bootstrap_rounds: int,
    workers: int = 1,
    reuse_dashboards: tuple[dict[str, object], ...] = (),
    *,
    clf: pd.DataFrame,
    reg: pd.DataFrame,
) -> tuple[list[str], list[str], list[dict[str, object]], dict[str, object]]:
    assert workers > 0
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    metric_datasets = set(clf["dataset"].astype(str)) | set(reg["dataset"].astype(str))
    missing = sorted(metric_datasets - set(domain_of))
    assert not missing, f"Metric datasets have no registered modality: {missing}"
    domains = sorted({domain_of[dataset] for dataset in metric_datasets})

    metrics = list(ELO_METRICS) if not clf.empty else ["f1_macro"]
    rows = []
    cell_labels = {cell: cell_label(cell) for cell in complete_cells}
    task_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for cell in complete_cells:
        clf_cell = clf[clf["cell"] == cell]
        reg_cell = reg[reg["cell"] == cell]
        for metric in metrics:
            for domain in ["all", *domains]:
                domain_datasets = (
                    metric_datasets
                    if domain == "all"
                    else {dataset for dataset in metric_datasets if domain_of[dataset] == domain}
                )
                table = fold_scores(
                    clf_cell[clf_cell["dataset"].isin(domain_datasets)],
                    reg_cell[reg_cell["dataset"].isin(domain_datasets)],
                    clf_metric=metric,
                )
                task_tables[(cell, metric, domain)] = table

    algorithm_sha256 = domain_elo_algorithm_sha256()
    input_sha256 = {
        task: domain_elo_task_input_sha256(
            cell,
            cell_labels[cell],
            metric,
            domain,
            bootstrap_rounds,
            table,
        )
        for task, table in task_tables.items()
        for cell, metric, domain in [task]
    }
    source_indexes = []
    for source in reuse_dashboards:
        if "domain_elo_cache" not in source or "domain_elo" not in source:
            continue
        cached = source["domain_elo_cache"]
        if cached["algorithm_sha256"] != algorithm_sha256:
            continue
        rows_by_task: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in source["domain_elo"]:
            task = (str(row["cell"]), str(row["metric"]), str(row["domain"]))
            if task in task_tables:
                rows_by_task.setdefault(task, []).append(row)
        if "tasks" in cached:
            source_indexes.append((cached["tasks"], rows_by_task))

    reused_tasks: set[tuple[str, str, str]] = set()
    reused_rows: list[dict[str, object]] = []
    for task in task_tables:
        cache_key = json.dumps(task, separators=(",", ":"))
        for cached_tasks, rows_by_task in source_indexes:
            task_rows = rows_by_task[task] if task in rows_by_task else []
            if cache_key not in cached_tasks:
                continue
            cached_task = cached_tasks[cache_key]
            if cached_task["input_sha256"] != input_sha256[task]:
                continue
            if cached_task["output_sha256"] != domain_elo_output_sha256(task_rows):
                continue
            reused_tasks.add(task)
            reused_rows.extend(dict(row) for row in task_rows)
            break
    for row in reused_rows:
        row.update(models[str(row["model_id"])])
    rows.extend(reused_rows)

    refresh_tasks = set(task_tables) - reused_tasks
    print(
        f"Domain Elo cache: reused {len(reused_tasks)}/{len(task_tables)} tasks; "
        f"recomputing {len(refresh_tasks)} tasks."
    )
    tasks = [
        (
            cell,
            cell_labels[cell],
            metric,
            domain,
            table,
            bootstrap_rounds,
        )
        for (cell, metric, domain), table in task_tables.items()
        if (cell, metric, domain) in refresh_tasks
    ]
    if workers == 1 or len(tasks) <= 1:
        results = map(_domain_elo_task, tasks)
        for cell_rows in tqdm(results, total=len(tasks), desc="Elo", unit="pool", mininterval=1):
            for row in cell_rows:
                row.update(models[str(row["model_id"])])
                rows.append(row)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=threadpool_limits,
            initargs=(1,),
        ) as pool:
            futures = [pool.submit(_domain_elo_task, task) for task in tasks]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(tasks),
                desc="Elo",
                unit="pool",
                mininterval=1,
            ):
                cell_rows = future.result()
                for row in cell_rows:
                    row.update(models[str(row["model_id"])])
                    rows.append(row)
    metric_order = {metric: index for index, metric in enumerate(metrics)}
    domain_order = {domain: index for index, domain in enumerate(["all", *domains])}
    rows.sort(
        key=lambda row: (
            cell_sort_key(str(row["cell"])),
            metric_order[str(row["metric"])],
            domain_order[str(row["domain"])],
            -int(row["Elo"]),
            str(row["model_id"]),
        )
    )
    available = {str(row["domain"]) for row in rows}
    domains = [domain for domain in domains if domain in available]
    cache = {
        "algorithm_sha256": algorithm_sha256,
        "tasks": {
            json.dumps(task, separators=(",", ":")): {
                "input_sha256": input_sha256[task],
                "output_sha256": domain_elo_output_sha256(
                    [row for row in rows if (row["cell"], row["metric"], row["domain"]) == task]
                ),
            }
            for task in task_tables
        },
    }
    return ["all", *domains], metrics, rows, cache


def build_model_card_coverage(
    datasets: list[dict[str, object]],
    complete_cells: list[str],
    scores: pd.DataFrame,
) -> list[dict[str, object]]:
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    scores = scores[scores["cell"].isin(complete_cells)].copy()
    scores["domain"] = scores["dataset"].astype(str).map(domain_of)
    assert scores["domain"].notna().all(), "Model-card datasets lack modality metadata"
    assert scores["imputed"].dtype == bool, scores["imputed"].dtype

    rows = []
    for (cell, model, domain), frame in scores.groupby(["cell", "model", "domain"], sort=False):
        failed = int(frame["imputed"].sum())
        rows.append(
            {
                "cell": str(cell),
                "model_id": str(model),
                "domain": str(domain),
                "successful_fits": int(len(frame) - failed),
                "failed_fits": failed,
            }
        )
    rows.sort(
        key=lambda row: (
            cell_sort_key(str(row["cell"])),
            str(row["domain"]),
            str(row["model_id"]),
        )
    )
    return rows


def build_cost_grid(
    models: dict[str, dict[str, str]],
    datasets: list[dict[str, object]],
    complete_cells: list[str],
    domains: list[str],
    metrics: list[str],
    scores: pd.DataFrame,
    run_stats: pd.DataFrame,
    adaptive: bool = False,
) -> list[dict[str, object]]:
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    if scores.empty:
        return []
    scores = scores[scores["cell"].isin(complete_cells)].copy()
    scores["domain"] = scores["dataset"].astype(str).map(domain_of)
    assert scores["domain"].notna().all(), "Cost-grid datasets lack modality metadata"
    scores["timing_cell"] = scores["cell"]
    if adaptive:
        fallback = scores["fallback"].fillna(False).astype(bool)
        scores.loc[fallback, "timing_cell"] = scores.loc[fallback, "reused_from_cell"]
        assert scores.loc[fallback, "timing_cell"].astype(bool).all(), (
            "Adaptive fallback rows must identify their timing source cell"
        )

    run_stats = run_stats[run_stats["status"] == "pass"][
        ["cell", "seed", "key", "model", "train_time_s", "inference_time_s"]
    ].rename(columns={"cell": "timing_cell"})
    timing_columns = ["train_time_s", "inference_time_s"]
    run_stats = run_stats.dropna(subset=timing_columns)
    run_stats["total_time_s"] = run_stats["train_time_s"] + run_stats["inference_time_s"]
    assert not run_stats.duplicated(["timing_cell", "seed", "key", "model"]).any()
    merged = scores.merge(run_stats, on=["timing_cell", "seed", "key", "model"], how="inner")

    rows: list[dict[str, object]] = []
    for cell in complete_cells:
        cell_rows = merged[merged["cell"] == cell]
        for domain in domains:
            selected = cell_rows if domain == "all" else cell_rows[cell_rows["domain"] == domain]
            if selected.empty:
                continue
            aggregated = selected.groupby("model", sort=True).agg(
                train_time_s=("train_time_s", "median"),
                inference_time_s=("inference_time_s", "median"),
                total_time_s=("total_time_s", "median"),
                **{metric: (metric, "mean") for metric in metrics},
            )
            for model_id, values in aggregated.iterrows():
                model_id = str(model_id)
                row: dict[str, object] = {
                    "cell": cell,
                    "domain": domain,
                    "model_id": model_id,
                    "train_time_s": float(values["train_time_s"]),
                    "inference_time_s": float(values["inference_time_s"]),
                    "total_time_s": float(values["total_time_s"]),
                }
                row.update({metric: float(values[metric]) for metric in metrics})
                row.update(models[model_id])
                rows.append(row)
    return rows


def parse_cell(cell: str) -> tuple[int | None, int | None]:
    match = CELL_RE.fullmatch(cell)
    assert match, f"Invalid cell identifier: {cell}"
    cap_token, sample_token = match.groups()
    cap = None if cap_token == "full" else int(cap_token)
    samples = None if sample_token is None else int(sample_token)
    return cap, samples


def cell_sort_key(cell: str) -> tuple[float, float]:
    cap, samples = parse_cell(cell)
    return (math.inf if cap is None else cap, math.inf if samples is None else samples)


def write_leaderboard_export(dashboard: dict[str, object], output: Path) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "snapshot_utc": dashboard["meta"]["snapshot_utc"],
        "reference_cell": dashboard["meta"]["reference_cell"],
        "primary_analysis_view": "strict",
        "models": dashboard["models"],
        "cell_options": dashboard["cell_options"],
        "reference": dashboard["reference"],
        "cell_elo": dashboard["cell_elo"],
        "analysis_views": {
            name: {
                "reference": view["reference"],
                "cell_elo": view["cell_elo"],
            }
            for name, view in dashboard["analysis_views"].items()
        },
    }
    write_json(output, payload)
    return {
        "name": "Leaderboard JSON",
        "path": "data/leaderboard.json",
        "records": len(payload["cell_elo"]),
        "bytes": output.stat().st_size,
        "sha256": digest(output),
        "source": "generated from data/dashboard.json",
        "format": "json",
        "available": True,
    }


def cell_label(cell: str) -> str:
    cap, samples = parse_cell(cell)
    return f"p={'all' if cap is None else f'{cap:,}'}, n={'all' if samples is None else f'{samples:,}'}"


def classification_summary(frame):
    metrics = list(ELO_METRICS)
    summary = (
        frame.groupby(["cell", "dataset", "model"], dropna=False)[metrics]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary.columns = ["__".join(column).rstrip("_") for column in summary.columns]
    return clean_json(summary.to_dict("records"))


def build_view(
    frames, models, datasets, cells, reference_cell, run_stats, n_boot, workers, reuse, adaptive
):
    clf, reg = frames["classification"], frames["regression"]
    domains, metrics, domain_elo, cache = build_domain_elo(
        [],
        models,
        datasets,
        cells,
        n_boot,
        workers,
        reuse,
        clf=clf,
        reg=reg,
    )
    elo = [
        dict(row) for row in domain_elo if row["domain"] == "all" and row["metric"] == "f1_macro"
    ]
    reference = [dict(row) for row in elo if row["cell"] == reference_cell]
    for metric, frame in (("f1_macro", clf), ("rmse", reg)):
        means = frame[frame["cell"] == reference_cell].groupby("model")[metric].mean()
        for row in reference:
            row[metric] = float(means[row["model_id"]]) if row["model_id"] in means else None
    if elo:
        table = (
            pd.DataFrame(elo)
            .pivot(index="model_id", columns="cell", values="Elo")
            .reindex(columns=cells)
        )
        correlations = clean_json(table.corr(method="spearman").to_numpy().tolist())
    else:
        correlations = [[None for _ in cells] for _ in cells]
    return {
        "cell_options": [
            {
                "id": cell,
                "label": cell_label(cell),
                "feature_cap": parse_cell(cell)[0],
                "n_train": parse_cell(cell)[1],
            }
            for cell in cells
        ],
        "cell_elo": elo,
        "reference": reference,
        "domains": domains,
        "elo_metrics": metrics,
        "domain_elo": domain_elo,
        "domain_elo_cache": cache,
        "model_card_coverage": build_model_card_coverage(datasets, cells, clf),
        "cost_grid": build_cost_grid(
            models, datasets, cells, domains, metrics, clf, run_stats, adaptive
        ),
        "rank_correlations": {"cells": cells, "matrix": correlations},
    }


def build_website(
    database: Path,
    output: Path,
    *,
    cells=None,
    workers=1,
    n_boot=DEFAULT_N_BOOT,
    reference_cell=None,
    results_url="",
):
    """Write the deployable static website and return its dashboard and strict metrics."""
    assert workers > 0 and n_boot > 0
    database, output = database.resolve(), output.resolve()
    metric_cache = output.with_name(output.name + ".cache") / "fold_metrics.sqlite"
    print(f"Fold-metric cache: {metric_cache}", flush=True)
    configs, frames, status, attempt_count = read_inputs(
        database, cells, workers=workers, metric_cache=metric_cache
    )
    ordered_cells = sorted(configs, key=cell_sort_key)
    if reference_cell is None:
        reference_cell = "cap_10000_n100" if "cap_10000_n100" in configs else ordered_cells[0]
    assert reference_cell in configs, f"Reference cell absent: {reference_cell}"
    model_ids = sorted(
        {model for config in configs.values() for model in config["models"]} | set(status["model"])
    )
    models = {key: model_meta(key) for key in model_ids}
    datasets = dataset_metadata(configs)
    for dataset in datasets:
        dataset["source_url"] = dataset_source_url(dataset["source"], dataset.pop("fetch_id"))
    progress = progress_summary(configs, status)
    dashboard_path = output / "data" / "dashboard.json"
    prior = ()
    if dashboard_path.is_file():
        previous = json.loads(dashboard_path.read_text(encoding="utf-8"))
        if "analysis_views" in previous:
            prior = tuple(previous["analysis_views"].values())
    views = {}
    for name, metrics in frames.items():
        print(f"Building {name} dashboard ({len(ordered_cells)} cells)", flush=True)
        views[name] = build_view(
            metrics,
            models,
            datasets,
            ordered_cells,
            reference_cell,
            status,
            n_boot,
            workers,
            (*views.values(), *prior),
            name == "adaptive",
        )
    summary = classification_summary(frames["strict"]["classification"])
    add_dataset_performance(
        datasets,
        models,
        reference_cell,
        cell_label(reference_cell),
        summary,
        frames["strict"]["regression"],
    )
    dashboard = {
        "schema_version": 9,
        "meta": {
            "title": "TabBench Bio",
            "tagline": "A living benchmark for tabular learning in biomedical HDLSS regimes.",
            "snapshot_utc": progress["snapshot_utc"],
            "reference_cell": reference_cell,
            "paper_url": "https://arxiv.org/abs/2609.07441",
            "github_url": "https://github.com/not-a-feature/TabBench-Bio",
            "domain": "tabbench-bio.eu",
            "contact_url": "https://github.com/not-a-feature/TabBench-Bio/issues",
            "affiliation": "Methods in Medical Informatics, University of Tübingen",
            "plot_excluded_models": ["DUMMY"],
            "primary_analysis_view": "strict_nominal_cell",
            "adaptive_analysis_role": "sensitivity_only",
            "paper_analysis_view": "strict",
            "configured_model_count": len(set(model_ids) - {"DUMMY", "AUTOGLUON"}),
            "evaluation_points_per_model": sum(
                len(get_seeds(config))
                * (len(config["datasets_classification"]) + len(config["datasets_regression"]))
                for config in configs.values()
            ),
            "bootstrap_rounds": n_boot,
        },
        "progress": progress,
        "models": models,
        "datasets": datasets,
        **views["strict"],
        "analysis_views": views,
        "raw_exports": [
            {
                "name": "Results SQLite",
                "path": results_url,
                "records": attempt_count,
                "bytes": database.stat().st_size,
                "sha256": sha256_file(database),
                "source": database.name,
                "format": "sqlite3",
                "available": bool(results_url),
                "upload_filename": database.name,
            }
        ],
    }
    # Template files contain no generated results or local paths.
    shutil.copytree(PACKAGE_ROOT / "web", output, dirs_exist_ok=True)
    ignore_path = output / ".gitignore"
    ignore_rules = ignore_path.read_text(encoding="utf-8") if ignore_path.exists() else ""
    if "/local/" not in ignore_rules.splitlines():
        ignore_path.write_text(ignore_rules.rstrip("\n") + "\n/local/\n", encoding="utf-8")
    entry = write_leaderboard_export(dashboard, output / "data" / "leaderboard.json")
    dashboard["raw_exports"].insert(0, entry)
    write_json(dashboard_path, dashboard)
    write_dataset_explorer(output, dashboard, summary, frames["strict"]["regression"])
    write_agent_metadata(output, dashboard)
    print(f"Website: {output / 'index.html'}", flush=True)
    return dashboard, frames["strict"]
