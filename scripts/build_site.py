"""Build the public TabBench Bio site from TabBench-Bio publication artifacts.

The script reads only frozen aggregation artifacts and monitoring exports. It does not
run models, recompute metrics, or modify benchmark results. All public data is written as
JSON under the selected TabBench-Bio site checkout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from tabbench_bio.bio.loaders.tdc import DATAVERSE_URL, ENDPOINTS
from tabbench_bio.elo import compute_elo, score_table
from tabbench_bio.model_constraints import REGULAR_MAX_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DATA = PROJECT_ROOT / "tex" / "generated" / "data"
GENERATED_MANIFEST = PROJECT_ROOT / "tex" / "generated" / "manifest.json"
PRELIMINARY_DATA = PROJECT_ROOT / "results" / "publication_snapshot"
PUBLIC_SITE_DIR = PROJECT_ROOT.parent / "TabBench-Bio"
DEFAULT_PAPER_URL = "https://arxiv.org/abs/XXXX.XXXXX"
DEFAULT_RESULTS_SQLITE = (
    PROJECT_ROOT / "results" / "feature_sweep_all_v4" / "results.sqlite"
)
DATASET_REGISTRY = PROJECT_ROOT / "src" / "tabbench_bio" / "bio" / "data" / "bio_datasets.json"
CLASSIFICATION_DATASETS = PROJECT_ROOT / "configs" / "datasets" / "bio_classification.json"
REGRESSION_DATASETS = PROJECT_ROOT / "configs" / "datasets" / "bio_regression.json"
GRID_CONFIG = PROJECT_ROOT / "configs" / "grid_sweep_all.json"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "models" / "all.json"
TRAINING_DATA_OVERLAP = {
    model["key"] for model in json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    if "training_data_overlap" in model and model["training_data_overlap"] is True
}

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
    "REALTABPFN-V2.5": "RealTabPFN v2.5",
    "TABPFN-V3": "TabPFN v3",
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

DOMAIN_ELO_IMPLEMENTATION_FILES = (
    PROJECT_ROOT / "src" / "tabbench_bio" / "elo.py",
    PROJECT_ROOT / "src" / "tabbench_bio" / "_vendor" / "tabarena_elo_utils.py",
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

MONITORING_EXPORTS = {
    "status": ".csv",
    "run_stats": ".csv",
    "progress": ".json",
}
LATEX_EXCLUDED_MODELS = {"DUMMY", "AUTOGLUON"}
SITEMAP_PATHS = (
    "",
    "datasets.html",
    "artifacts.html",
    "changelog.html",
    "llms.txt",
    "citation.html",
    "CITATION.cff",
    "CITATION.bib",
)

CELL_RE = re.compile(r"^cap_(full|\d+)(?:_n(\d+))?$")
INTEGER_RE = re.compile(r"^-?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_generated_inputs_current() -> None:
    assert GENERATED_MANIFEST.is_file(), (
        f"Missing {GENERATED_MANIFEST.relative_to(PROJECT_ROOT)}; "
        "run tex/generate_paper_assets.py before scripts/build_site.py."
    )
    manifest = json.loads(GENERATED_MANIFEST.read_text(encoding="utf-8"))
    stale = []
    for entry in manifest["inputs"]:
        path = PROJECT_ROOT / str(entry["path"])
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]):
            stale.append(str(entry["path"]))
    assert not stale, (
        "Generated publication assets have stale inputs: "
        f"{stale}. Re-run tex/generate_paper_assets.py before scripts/build_site.py."
    )


def configured_dataset_ids() -> list[str]:
    classification = json.loads(CLASSIFICATION_DATASETS.read_text(encoding="utf-8"))
    regression = json.loads(REGRESSION_DATASETS.read_text(encoding="utf-8"))
    dataset_ids = [str(dataset_id) for dataset_id in classification + regression]
    assert len(dataset_ids) == len(set(dataset_ids)), "Dataset configuration contains duplicates."
    return dataset_ids


def assert_dataset_metadata_current(rows: list[dict[str, object]]) -> None:
    expected = configured_dataset_ids()
    actual = [str(row["dataset_id"]) for row in rows]
    missing = [dataset_id for dataset_id in expected if dataset_id not in actual]
    extra = [dataset_id for dataset_id in actual if dataset_id not in expected]
    assert actual == expected, (
        "Generated dataset_metadata.csv is stale or reordered: "
        f"missing={missing}, extra={extra}. "
        "Re-run tex/generate_paper_assets.py before scripts/build_site.py."
    )


def parse_scalar(value: str) -> str | int | float | bool | None:
    value = value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"nan", "na", "none", "null"}:
        return None
    if INTEGER_RE.fullmatch(value):
        return int(value)
    if not FLOAT_RE.fullmatch(value):
        return value
    number = float(value)
    assert math.isfinite(number), f"Non-finite numeric value: {value}"
    return number


def read_csv(path: Path) -> list[dict[str, object]]:
    assert path.is_file(), f"Missing input: {path}"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames, f"CSV has no header: {path}"
        return [{key: parse_scalar(value) for key, value in row.items()} for row in reader]


def discover_monitoring_exports(monitoring_dir: Path) -> dict[str, Path]:
    """Load the single monitoring bundle produced by the publication rebuild."""
    exports = {
        kind: monitoring_dir / f"tabbench_bio_preliminary_{kind}_current{suffix}"
        for kind, suffix in MONITORING_EXPORTS.items()
    }
    missing = [path for path in exports.values() if not path.is_file()]
    assert not missing, f"Missing monitoring exports: {missing}"
    return exports


def write_json(path: Path, payload: object, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"ensure_ascii": False, "allow_nan": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    path.write_text(json.dumps(payload, **options) + "\n", encoding="utf-8")


def tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements[char] if char in replacements else char for char in str(value))


def write_latex_leaderboard(
    reference: list[dict[str, object]], output: Path
) -> None:
    peers = [row for row in reference if row["model_id"] not in LATEX_EXCLUDED_MODELS]
    assert peers, "Reference leaderboard has no ranked peer models."
    lines = [
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r"Rank & Model & Elo & 95\% CI & Targets & Macro-F1 \\",
        r"\midrule",
    ]
    for rank, row in enumerate(peers, start=1):
        macro_f1 = "--" if row["f1_macro"] is None else f"{float(row['f1_macro']):.3f}"
        lines.append(
            f"{rank} & {tex_escape(row['display'])} & {int(row['Elo'])} & "
            f"[{int(row['Elo_lo'])}, {int(row['Elo_hi'])}] & "
            f"{int(row['n_targets'])} & {macro_f1} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_llms_text(dashboard: dict[str, object], project_url: str) -> str:
    project_url = project_url.rstrip("/")
    meta = dashboard["meta"]
    progress = dashboard["progress"]
    datasets = dashboard["datasets"]
    task_counts = Counter(str(row["task"]).lower() for row in datasets)
    modality_counts = Counter(str(row["modality"]) for row in datasets)
    modality_summary = ", ".join(
        f"{modality}: {count}" for modality, count in sorted(modality_counts.items())
    )
    reference = sorted(dashboard["reference"], key=lambda row: row["Elo"], reverse=True)
    reference_label = str(reference[0]["cell_label"])
    leaders = "\n".join(
        f'{rank}. {row["display"]}: Elo {int(row["Elo"]):,} '
        f'(95% CI {int(row["Elo_lo"]):,}–{int(row["Elo_hi"]):,}; '
        f'{int(row["n_targets"]):,} targets)'
        for rank, row in enumerate(reference[:5], start=1)
    )
    evaluated_models = ", ".join(
        sorted(str(row["display"]) for row in dashboard["models"].values())
    )
    status = progress["status"]
    paper_url = str(meta["paper_url"])
    if not paper_url.startswith(("https://", "http://")):
        paper_url = f"{project_url}/{paper_url.lstrip('/')}"
    sqlite_export = next(
        row for row in dashboard["raw_exports"] if row["format"] == "sqlite3"
    )
    sqlite_line = (
        f'- [Canonical results SQLite]({sqlite_export["path"]}): '
        f'{int(sqlite_export["records"]):,} attempts, '
        f'SHA-256 `{sqlite_export["sha256"]}`'
        if sqlite_export["available"]
        else (
            "- Canonical results SQLite: release upload pending; "
            f'{int(sqlite_export["bytes"]) / 1_000_000:.1f} MB, '
            f'SHA-256 `{sqlite_export["sha256"]}`'
        )
    )
    return f"""# TabBench-Bio

> TabBench-Bio is a living benchmark for tabular learning in high-dimensional biomedical regimes. It compares classical models, neural networks, AutoML, and tabular foundation models across controlled feature and sample budgets.

Use the pages below for context and the canonical SQLite database for exact, structured results. Rankings are operating-point dependent and descriptive rather than universal. The primary charts and rankings use strict nominal-cell results. The interactive sensitivity view may reuse a verified smaller training-sample cell after a training out-of-memory failure and is not treated as performance at the requested larger sample size.

## Current snapshot

- Generated from the published result bundle at {meta["snapshot_utc"]}.
- {len(datasets):,} registered datasets: {task_counts["classification"]:,} classification, {task_counts["regression"]:,} regression.
- Dataset modalities: {modality_summary}.
- {int(meta["configured_model_count"]):,} configured model configurations; {len(dashboard["models"]):,} currently represented in aggregate results.
- {len(dashboard["cell_options"]):,} feature-by-sample operating points currently have aggregate metrics.
- {int(meta["evaluation_points_per_model"]):,} configured dataset-cell-fold evaluation points per model.
- Run monitoring has recorded {int(progress["recorded"]):,} of {int(progress["expected"]):,} planned units ({100 * float(progress["fraction"]):.1f}%): {int(status["pass"]):,} pass, {int(status["skip"]):,} design skip, and {int(status["fail"]):,} fail.

## Current reference results

The predeclared reference operating point is {reference_label}. Bradley–Terry Elo combines the classification and regression target comparisons at this cell and anchors Random Forest at 1,000. Target-bootstrap intervals quantify ranking uncertainty.

{leaders}

The leading intervals overlap, so the displayed order should not be interpreted as a statistically resolved universal ranking. Use the interactive controls or canonical database for other operating points.

## Evaluated models

{evaluated_models}.

## Primary pages

- [Benchmark and interactive results]({project_url}/)
- [Dataset registry]({project_url}/datasets.html)
- [Artifact browser]({project_url}/artifacts.html)
- [Manuscript]({paper_url})

## Machine-readable data

{sqlite_line}

## Citation

If you read or use the benchmark, website, or published result artifacts, cite:

Kreuer, J.; Hellmig, J.; Braitinger, J.; Ouaari, S.; Pfeifer, N. (2026). TabBench-Bio (Version 0.1.0).

- [Citation File Format metadata]({project_url}/CITATION.cff)
- Citation target: [TabBench-Bio repository](https://github.com/not-a-feature/TabBench-Bio); this will move to the arXiv DOI when available
- Authors: Jules Kreuer, Julia Hellmig, Julius Braitinger, Sofiane Ouaari, and Nico Pfeifer
- DOI: not yet available
- arXiv: not yet available

## Optional

- [Changelog]({project_url}/changelog.html): notable benchmark and website updates
- [GitHub repository](https://github.com/not-a-feature/TabBench-Bio): canonical source code and issue tracker
"""


def build_sitemap(project_url: str, last_modified: str) -> str:
    project_url = project_url.rstrip("/")
    urls = "\n".join(
        "  <url>\n"
        f"    <loc>{escape(f'{project_url}/{path}')}</loc>\n"
        f"    <lastmod>{last_modified}</lastmod>\n"
        "  </url>"
        for path in SITEMAP_PATHS
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- Generated by scripts/build_site.py; do not edit manually. -->\n"
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )


def write_agent_metadata(
    site_dir: Path,
    dashboard: dict[str, object],
    project_url: str,
) -> None:
    project_url = project_url.rstrip("/")
    build_date = datetime.now(UTC).date().isoformat()
    (site_dir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {project_url}/sitemap.xml\n",
        encoding="utf-8",
    )
    (site_dir / "sitemap.xml").write_text(
        build_sitemap(project_url, build_date),
        encoding="utf-8",
    )
    (site_dir / "llms.txt").write_text(
        build_llms_text(dashboard, project_url),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_meta(model_id: str) -> dict[str, object]:
    assert model_id in MODEL_DISPLAY, f"Missing display name for {model_id}"
    assert model_id in MODEL_CATEGORY, f"Missing category for {model_id}"
    category = MODEL_CATEGORY[model_id]
    return {
        "id": model_id,
        "display": MODEL_DISPLAY[model_id],
        "category": category,
        "color": CATEGORY_COLORS[category],
        "regular_max_features": REGULAR_MAX_FEATURES.get(model_id),
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


def dataset_metadata() -> list[dict[str, object]]:
    rows = read_csv(GENERATED_DATA / "dataset_metadata.csv")
    assert_dataset_metadata_current(rows)
    registry = json.loads(DATASET_REGISTRY.read_text(encoding="utf-8"))
    specs = {str(spec["bio_id"]): spec for spec in registry}
    missing = sorted(str(row["dataset_id"]) for row in rows if str(row["dataset_id"]) not in specs)
    assert not missing, f"Dataset metadata is absent from the registry: {missing}"
    for row in rows:
        spec = specs[str(row["dataset_id"])]
        assert str(row["source"]) == str(spec["source"]), row["dataset_id"]
        row["source_url"] = dataset_source_url(str(spec["source"]), str(spec["fetch_id"]))
    return rows


def add_dataset_performance(
    datasets: list[dict[str, object]],
    models: dict[str, dict[str, str]],
    reference_cell: str,
    reference_label: str,
    suffix: str = "",
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
    classification = read_csv(GENERATED_DATA / f"sweep_summary{suffix}.csv")
    regression = pd.read_csv(GENERATED_DATA / f"sweep_metrics_regression{suffix}.csv")
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
                values = {
                    key: source_row[f"{key}__mean"]
                    for key, _label, _better in metric_specs
                }
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
                (regression["dataset"] == dataset_id)
                & (regression["cell"] == selected_cell)
            ]
            for model_id, frame in source_rows.groupby("model", sort=False):
                values = {
                    key: float(frame[key].mean())
                    for key, _label, _better in metric_specs
                }
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


def write_dataset_explorer(site_dir: Path, dashboard: dict[str, object]) -> None:
    """Export a compact registry and per-dataset scores for every nominal grid cell."""
    classification = read_csv(GENERATED_DATA / "sweep_summary_strict.csv")
    regression = pd.read_csv(GENERATED_DATA / "sweep_metrics_regression_strict.csv")
    regression_means = regression.groupby(["dataset", "cell", "model"])[
        ["rmse", "mae", "r2"]
    ].mean().reset_index()
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
                rows.append({
                    "cell": row["cell"], "model_id": row["model"],
                    "values": {metric["key"]: row[f'{metric["key"]}__mean'] for metric in metrics},
                })
        else:
            for row in regression_means.loc[
                regression_means["dataset"] == dataset_id
            ].to_dict("records"):
                rows.append({
                    "cell": row["cell"], "model_id": row["model"],
                    "values": {metric["key"]: row[metric["key"]] for metric in metrics},
                })
        for row in rows:
            assert row["model_id"] in dashboard["models"], row
            row["values"] = {
                key: round(float(value), 8) if pd.notna(value) else None
                for key, value in row["values"].items()
            }
        assert len({(row["cell"], row["model_id"]) for row in rows}) == len(rows), dataset_id
        filename = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16] + ".json"
        metadata = {key: value for key, value in dataset.items() if key != "performance"}
        registry.append({
            **metadata, "metrics": metrics,
            "default_cell": dataset["performance"]["cell"], "scores_file": filename,
        })
        write_json(output_dir / filename, {"dataset_id": dataset_id, "scores": rows})
    write_json(output_dir / "index.json", {
        "meta": dashboard["meta"], "models": dashboard["models"],
        "cell_options": dashboard["cell_options"], "datasets": registry,
    })


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


def score_table_sha256(table: pd.DataFrame) -> str:
    ordered = table.sort_index().sort_index(axis=1)
    payload = {
        "models": [str(value) for value in ordered.index],
        "targets": [str(value) for value in ordered.columns],
        "scores": [
            [None if pd.isna(value) else float(value).hex() for value in row]
            for row in ordered.to_numpy()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
        "score_table_sha256": score_table_sha256(table),
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
    suffix: str = "",
) -> tuple[list[str], list[str], list[dict[str, object]], dict[str, object]]:
    assert workers > 0
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    clf = pd.read_csv(
        GENERATED_DATA / f"sweep_metrics_classification{suffix}.csv", low_memory=False
    )
    reg = pd.read_csv(
        GENERATED_DATA / f"sweep_metrics_regression{suffix}.csv", low_memory=False
    )
    metric_datasets = set(clf["dataset"].astype(str)) | set(reg["dataset"].astype(str))
    missing = sorted(metric_datasets - set(domain_of))
    assert not missing, f"Metric datasets have no registered modality: {missing}"
    domains = sorted({domain_of[dataset] for dataset in metric_datasets})

    metrics = [metric for metric in ELO_METRICS if metric in clf.columns]
    assert metrics[0] == "f1_macro", metrics
    rows = [{**row, "domain": "all", "metric": "f1_macro"} for row in elo]
    cell_labels = {str(row["cell"]): str(row["cell_label"]) for row in elo}
    task_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for cell in complete_cells:
        clf_cell = clf[clf["cell"] == cell]
        reg_cell = reg[reg["cell"] == cell]
        for metric in metrics:
            for domain in ["all", *domains]:
                if metric == "f1_macro" and domain == "all":
                    continue
                domain_datasets = (
                    metric_datasets
                    if domain == "all"
                    else {dataset for dataset in metric_datasets if domain_of[dataset] == domain}
                )
                table = score_table(
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
        for cell_rows in results:
            for row in cell_rows:
                row.update(models[str(row["model_id"])])
                rows.append(row)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks))
        ) as pool:
            for cell_rows in pool.map(_domain_elo_task, tasks):
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
                    [
                        row
                        for row in rows
                        if (row["cell"], row["metric"], row["domain"]) == task
                    ]
                ),
            }
            for task in task_tables
        },
    }
    return ["all", *domains], metrics, rows, cache


def build_model_card_coverage(
    datasets: list[dict[str, object]],
    complete_cells: list[str],
    suffix: str = "",
) -> list[dict[str, object]]:
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    scores = pd.read_csv(
        GENERATED_DATA / f"sweep_metrics_classification{suffix}.csv",
        usecols=["cell", "dataset", "model", "imputed"],
    )
    scores = scores[scores["cell"].isin(complete_cells)].copy()
    scores["domain"] = scores["dataset"].astype(str).map(domain_of)
    assert scores["domain"].notna().all(), "Model-card datasets lack modality metadata"
    assert scores["imputed"].dtype == bool, scores["imputed"].dtype

    rows = []
    for (cell, model, domain), frame in scores.groupby(
        ["cell", "model", "domain"], sort=False
    ):
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
    monitoring_exports: dict[str, Path],
    suffix: str = "",
) -> list[dict[str, object]]:
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    scores = pd.read_csv(
        GENERATED_DATA / f"sweep_metrics_classification{suffix}.csv", low_memory=False
    )
    scores = scores[scores["cell"].isin(complete_cells)].copy()
    scores["domain"] = scores["dataset"].astype(str).map(domain_of)
    assert scores["domain"].notna().all(), "Cost-grid datasets lack modality metadata"
    scores["timing_cell"] = scores["cell"]
    if suffix == "_adaptive":
        fallback = scores["fallback"].fillna(False).astype(bool)
        scores.loc[fallback, "timing_cell"] = scores.loc[fallback, "reused_from_cell"]
        assert scores.loc[fallback, "timing_cell"].astype(bool).all(), (
            "Adaptive fallback rows must identify their timing source cell"
        )

    run_stats = pd.read_csv(monitoring_exports["run_stats"])
    run_stats = run_stats[run_stats["status"] == "pass"][
        ["cell", "seed", "dataset", "model", "train_time_s", "inference_time_s"]
    ].rename(
        columns={"cell": "timing_cell", "dataset": "key"}
    )
    timing_columns = ["train_time_s", "inference_time_s"]
    assert run_stats[timing_columns].notna().all().all(), (
        "Successful runs must contain fit and prediction timing."
    )
    run_stats["total_time_s"] = run_stats["train_time_s"] + run_stats["inference_time_s"]
    assert not run_stats.duplicated(["timing_cell", "seed", "key", "model"]).any()
    merged = scores.merge(
        run_stats, on=["timing_cell", "seed", "key", "model"], how="inner"
    )
    assert not merged.empty, "No metric rows match the frozen fitting-time records"

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
    assert rows, "Cost grid is empty"
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


def _sqlite_attempt_count(path: Path) -> int:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"metadata", "cells", "blobs", "attempts"}.issubset(tables), path
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert metadata["schema_version"] == "1", metadata["schema_version"]
        return int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])


def build_artifact_index(
    results_sqlite: Path,
    results_sqlite_url: str,
) -> list[dict[str, object]]:
    results_sqlite = results_sqlite.resolve()
    assert results_sqlite.is_file(), f"Missing canonical result database: {results_sqlite}"
    try:
        sqlite_source = str(results_sqlite.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        sqlite_source = str(results_sqlite).replace("\\", "/")
    return [
        {
            "name": "Canonical Results SQLite",
            "path": results_sqlite_url,
            "records": _sqlite_attempt_count(results_sqlite),
            "bytes": results_sqlite.stat().st_size,
            "sha256": digest(results_sqlite),
            "source": sqlite_source,
            "format": "sqlite3",
            "available": bool(results_sqlite_url),
            "upload_filename": "tabbench-bio-results-v0.1.0.sqlite",
        }
    ]


def write_leaderboard_export(
    dashboard: dict[str, object], output: Path
) -> dict[str, object]:
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


def build_dashboard(
    raw_exports: list[dict[str, object]],
    arxiv_url: str,
    monitoring_exports: dict[str, Path],
    workers: int = 1,
    reuse_dashboards: tuple[dict[str, object], ...] = (),
    suffix: str = "",
) -> dict[str, object]:
    progress_path = monitoring_exports["progress"]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    aggregation_manifest = json.loads(
        (PROJECT_ROOT / "tex" / "generated" / "manifest.json").read_text(encoding="utf-8")
    )
    grid_config = json.loads(GRID_CONFIG.read_text(encoding="utf-8"))
    configured_models = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))

    elo = read_csv(GENERATED_DATA / f"cell_elo{suffix}.csv")
    reference_elo = read_csv(GENERATED_DATA / f"reference_elo{suffix}.csv")
    summaries = {
        str(row["model_id"]): row
        for row in read_csv(GENERATED_DATA / f"reference_model_summary{suffix}.csv")
    }
    costs = {}
    if not suffix:
        costs = {
            str(row["model"]): row
            for row in read_csv(GENERATED_DATA / "performance_vs_cost.csv")
        }

    model_ids = sorted({str(row["model_id"]) for row in elo})
    models = {model_id: model_meta(model_id) for model_id in model_ids}
    for row in elo:
        row.update(models[str(row["model_id"])])

    reference = []
    for row in reference_elo:
        model_id = str(row["model_id"])
        merged = dict(row)
        merged.update(models[model_id])
        merged.update(summaries[model_id])
        if model_id in costs:
            merged.update(costs[model_id])
        reference.append(merged)

    complete_cells = sorted({str(row["cell"]) for row in elo}, key=cell_sort_key)
    cell_options = []
    for cell in complete_cells:
        cap, samples = parse_cell(cell)
        matching = next(row for row in elo if row["cell"] == cell)
        cell_options.append(
            {
                "id": cell,
                "label": matching["cell_label"],
                "feature_cap": cap,
                "n_train": samples,
            }
        )

    rank_rows = read_csv(GENERATED_DATA / f"rank_correlations{suffix}.csv")
    rank_rows_by_cell = {str(row["cell"]): row for row in rank_rows}
    rank_columns = {str(key) for key in rank_rows[0] if key != "cell"}
    rank_cells = [
        cell
        for cell in complete_cells
        if cell in rank_columns and cell in rank_rows_by_cell
    ]
    assert rank_cells, "Rank-correlation data contain no completed cells"
    rank_matrix = [
        [rank_rows_by_cell[row_cell][column_cell] for column_cell in rank_cells]
        for row_cell in rank_cells
    ]
    datasets = dataset_metadata()
    evaluation_points_per_model = (
        int(grid_config["cv_folds"])
        * int(aggregation_manifest["configured_cell_count"])
        * len(datasets)
    )
    reference_label = next(
        str(row["cell_label"]) for row in elo if row["cell"] == aggregation_manifest["reference_cell"]
    )
    add_dataset_performance(
        datasets,
        models,
        str(aggregation_manifest["reference_cell"]),
        reference_label,
        suffix,
    )
    domains, elo_metrics, domain_elo, domain_elo_cache = build_domain_elo(
        elo,
        models,
        datasets,
        complete_cells,
        int(aggregation_manifest["bootstrap_rounds"]),
        workers,
        reuse_dashboards,
        suffix,
    )
    cost_grid = build_cost_grid(
        models,
        datasets,
        complete_cells,
        domains,
        elo_metrics,
        monitoring_exports,
        suffix,
    )
    model_card_coverage = build_model_card_coverage(datasets, complete_cells, suffix)

    return {
        "schema_version": 9,
        "meta": {
            "title": "TabBench Bio",
            "tagline": "A living benchmark for tabular learning in biomedical HDLSS regimes.",
            "snapshot_utc": progress["snapshot_utc"],
            "mode": aggregation_manifest["mode"],
            "reference_cell": aggregation_manifest["reference_cell"],
            "paper_url": arxiv_url,
            "github_url": "https://github.com/not-a-feature/TabBench-Bio",
            "domain": "tabbench-bio.eu",
            "contact_url": "https://github.com/not-a-feature/TabBench-Bio/issues",
            "affiliation": "Methods in Medical Informatics, University of Tübingen",
            "plot_excluded_models": ["DUMMY"],
            "primary_analysis_view": "strict_nominal_cell",
            "adaptive_analysis_role": "sensitivity_only",
            "paper_analysis_view": "strict",
            "configured_model_count": len(configured_models) - 2,
            "evaluation_points_per_model": evaluation_points_per_model,
        },
        "progress": progress,
        "models": models,
        "cell_options": cell_options,
        "cell_elo": elo,
        "domains": domains,
        "elo_metrics": elo_metrics,
        "domain_elo": domain_elo,
        "domain_elo_cache": domain_elo_cache,
        "model_card_coverage": model_card_coverage,
        "cost_grid": cost_grid,
        "reference": reference,
        "datasets": datasets,
        "rank_correlations": {"cells": rank_cells, "matrix": rank_matrix},
        "raw_exports": raw_exports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-url",
        default="https://tabbench-bio.eu",
        help="Canonical public URL used in generated metadata",
    )
    parser.add_argument(
        "--arxiv-url",
        default=DEFAULT_PAPER_URL,
        help="External arXiv abstract URL; defaults to a placeholder until submission",
    )
    parser.add_argument(
        "--results-sqlite",
        type=Path,
        default=DEFAULT_RESULTS_SQLITE,
        help="Canonical result database represented in the artifact manifest",
    )
    parser.add_argument(
        "--results-sqlite-url",
        default="",
        help="Release-asset URL; omit while the large SQLite upload is pending",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=PUBLIC_SITE_DIR,
        help="TabBench-Bio pages checkout that receives generated public assets",
    )
    parser.add_argument(
        "--monitoring-dir",
        type=Path,
        default=PRELIMINARY_DATA,
        help="Directory containing the frozen preliminary status, progress, and run-stat exports",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Worker processes for independent per-cell domain Elo bootstraps",
    )
    parser.add_argument(
        "--full-domain-elo",
        action="store_true",
        help="Ignore the validated per-cell cache and recompute every domain Elo row",
    )
    args = parser.parse_args()
    if not args.arxiv_url.startswith("https://arxiv.org/abs/"):
        parser.error("--arxiv-url must be an external https://arxiv.org/abs/ URL")
    monitoring_dir = args.monitoring_dir.resolve()
    site_dir = args.site_dir.resolve()
    monitoring_exports = discover_monitoring_exports(monitoring_dir)

    assert_generated_inputs_current()
    for required in (
        site_dir / "index.html",
        site_dir / "style.css",
        site_dir / "js" / "app.js",
    ):
        assert required.is_file(), f"Missing site source: {required}"
    assert (site_dir / "assets" / "plotly-cartesian.min.js").is_file(), (
        "Missing vendored Plotly cartesian bundle"
    )
    raw_dir = site_dir / "data" / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_exports = build_artifact_index(
        args.results_sqlite,
        args.results_sqlite_url,
    )
    dashboard_path = site_dir / "data" / "dashboard.json"
    reused_dashboard = None
    if dashboard_path.is_file() and not args.full_domain_elo:
        reused_dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    reused_views = reused_dashboard["analysis_views"] if reused_dashboard else {}
    prior_views = tuple(reused_views.values())
    strict_dashboard = build_dashboard(
        raw_exports,
        args.arxiv_url,
        monitoring_exports,
        args.workers,
        prior_views,
        "_strict",
    )
    adaptive_dashboard = build_dashboard(
        raw_exports,
        args.arxiv_url,
        monitoring_exports,
        args.workers,
        (strict_dashboard, *prior_views),
        "_adaptive",
    )
    view_fields = (
        "cell_options",
        "cell_elo",
        "domains",
        "elo_metrics",
        "domain_elo",
        "domain_elo_cache",
        "model_card_coverage",
        "cost_grid",
        "reference",
        "rank_correlations",
    )
    dashboard = strict_dashboard
    dashboard["analysis_views"] = {
        "strict": {field: strict_dashboard[field] for field in view_fields},
        "adaptive": {field: adaptive_dashboard[field] for field in view_fields},
    }
    leaderboard_entry = write_leaderboard_export(
        dashboard, site_dir / "data" / "leaderboard.json"
    )
    dashboard["raw_exports"] = [leaderboard_entry, *raw_exports]
    raw_exports = dashboard["raw_exports"]
    write_json(dashboard_path, dashboard)
    write_dataset_explorer(site_dir, dashboard)
    write_latex_leaderboard(
        strict_dashboard["reference"], site_dir / "data" / "leaderboard_table.tex"
    )
    write_agent_metadata(site_dir, dashboard, args.project_url)
    social_preview = site_dir / "assets" / "og.png"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_social_preview.py"),
            "--dashboard",
            str(dashboard_path),
            "--output",
            str(social_preview),
        ],
        check=True,
    )
    social_version = dashboard["meta"]["snapshot_utc"].replace(":", "").replace("-", "")
    social_url = f"https://tabbench-bio.eu/assets/og.png?v={social_version}"
    index_path = site_dir / "index.html"
    index = index_path.read_text(encoding="utf-8")
    paper_label = "arXiv (coming soon)" if args.arxiv_url == DEFAULT_PAPER_URL else "Read on arXiv"
    index, paper_replacements = re.subn(
        r'(<a id="paper-link"[^>]*href=")[^"]+("[^>]*>).*?</a>',
        lambda match: (
            match[1] + escape(args.arxiv_url, {'"': "&quot;"}) + match[2]
            + paper_label + ' <span aria-hidden="true">↗</span></a>'
        ),
        index,
    )
    assert paper_replacements == 1
    index, og_replacements = re.subn(
        r'(<meta property="og:image" content=")[^"]+("[^>]*>)',
        rf"\g<1>{social_url}\g<2>",
        index,
    )
    index, twitter_replacements = re.subn(
        r'(<meta name="twitter:image" content=")[^"]+("[^>]*>)',
        rf"\g<1>{social_url}\g<2>",
        index,
    )
    assert og_replacements == twitter_replacements == 1
    index_path.write_text(index, encoding="utf-8")

    (site_dir / "CNAME").write_text("tabbench-bio.eu\n", encoding="ascii")
    (site_dir / ".nojekyll").touch()

    total_bytes = sum(int(entry["bytes"]) for entry in raw_exports)
    try:
        output_dir = site_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        output_dir = site_dir
    print(
        f"Built {output_dir}: "
        f"{len(raw_exports)} artifact exports ({total_bytes / 1_000_000:.1f} MB indexed)"
    )


if __name__ == "__main__":
    main()
