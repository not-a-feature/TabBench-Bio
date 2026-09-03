"""Build the static TabBench Bio site from publication artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from tabbench_bio.bio.loaders.tdc import DATAVERSE_URL, ENDPOINTS
from tabbench_bio.elo import compute_elo, score_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATED_DATA = PROJECT_ROOT / "data" / "aggregated"
AGGREGATION_MANIFEST = AGGREGATED_DATA / "manifest.json"
GENERATED_MANIFEST = AGGREGATION_MANIFEST
PRELIMINARY_DATA = PROJECT_ROOT / "results" / "preliminary_20260820_201720"
DATASET_REGISTRY = PROJECT_ROOT / "src" / "tabbench_bio" / "bio" / "data" / "bio_datasets.json"
CLASSIFICATION_DATASETS = PROJECT_ROOT / "configs" / "datasets" / "bio_classification.json"
REGRESSION_DATASETS = PROJECT_ROOT / "configs" / "datasets" / "bio_regression.json"
GRID_CONFIG = PROJECT_ROOT / "configs" / "grid_sweep_all.json"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "models" / "all.json"

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
    "TABFM": "Tabular Foundation",
    "TABDPT": "Tabular Foundation",
    "TABICL": "Tabular Foundation",
    "AUTOGLUON": "AutoML",
}

MODEL_DISPLAY = {
    "DUMMY": "Constant",
    "KNN": "KNN",
    "LR": "Logistic Reg.",
    "RF": "Random Forest",
    "XT": "Extra Trees",
    "CAT": "CatBoost",
    "GBM": "LightGBM",
    "XGB": "XGBoost",
    "NN_TORCH": "MLP",
    "REALMLP": "RealMLP",
    "TABM": "TabM",
    "MITRA": "MITRA",
    "REALTABPFN-V2": "TabPFN v2",
    "REALTABPFN-V2.5": "TabPFN v2.5",
    "TABPFN-V3": "TabPFN v3",
    "TABPFN-WIDE": "TabPFN-Wide",
    "TABFM": "TabFM",
    "TABDPT": "TabDPT",
    "TABICL": "TabICL",
    "AUTOGLUON": "AutoGluon",
}
LATEX_EXCLUDED_MODELS = {"DUMMY", "AUTOGLUON"}

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

RAW_CSV_FILES = (
    "cell_elo.csv",
    "dataset_metadata.csv",
    "feature_budget_paired_effects.csv",
    "performance_vs_cost.csv",
    "rank_correlations.csv",
    "reference_elo.csv",
    "reference_model_summary.csv",
    "sample_budget_paired_effects.csv",
    "sample_fallbacks.csv",
    "sweep_metrics_classification.csv",
    "sweep_metrics_classification_strict.csv",
    "sweep_metrics_regression.csv",
    "sweep_metrics_regression_strict.csv",
    "sweep_summary.csv",
    "sweep_summary_strict.csv",
)

MONITORING_EXPORTS = {
    "status": ".csv",
    "run_stats": ".csv",
    "progress": ".json",
}

SITEMAP_PATHS = (
    "",
    "datasets.html",
    "artifacts.html",
    "changelog.html",
    "llms.txt",
    "citation.html",
    "CITATION.cff",
    "CITATION.bib",
    "data/raw/index.json",
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
    """Validate a publication manifest when its source tree is locally available."""
    assert GENERATED_MANIFEST.is_file(), f"Missing aggregation manifest: {GENERATED_MANIFEST}"
    manifest = json.loads(GENERATED_MANIFEST.read_text(encoding="utf-8"))
    stale = []
    for entry in manifest.get("inputs", []):
        path = PROJECT_ROOT / str(entry["path"])
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]):
            stale.append(str(entry["path"]))
    assert not stale, f"Generated publication assets have stale inputs: {stale}"


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
        f"missing={missing}, extra={extra}. Rebuild the aggregation before the website."
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
    """Select the newest complete status/run-stat/progress bundle in a directory."""
    candidates: dict[str, dict[str, Path]] = {}
    for kind, suffix in MONITORING_EXPORTS.items():
        prefix = f"tabbench_bio_preliminary_{kind}_"
        matches = sorted(monitoring_dir.glob(f"{prefix}*{suffix}"))
        assert matches, f"No {kind} monitoring export found under {monitoring_dir}"
        candidates[kind] = {
            path.name.removeprefix(prefix).removesuffix(suffix): path for path in matches
        }

    common_labels = set.intersection(*(set(paths) for paths in candidates.values()))
    assert common_labels, (
        "Monitoring directory has no status/run-stat/progress files with a shared snapshot label: "
        f"{monitoring_dir}"
    )
    label = max(common_labels)
    return {kind: paths[label] for kind, paths in candidates.items()}


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
    return "".join(replacements.get(char, char) for char in str(value))


def write_latex_leaderboard(
    reference: list[dict[str, object]], output: Path
) -> None:
    peers = [row for row in reference if row["model_id"] not in LATEX_EXCLUDED_MODELS]
    assert peers, "Reference leaderboard has no ranked peer models"
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


def resolve_aggregation_manifest(
    aggregated_dir: Path, explicit_manifest: Path | None
) -> Path:
    if explicit_manifest is not None:
        manifest = explicit_manifest.resolve()
        assert manifest.is_file(), f"Missing aggregation manifest: {manifest}"
        return manifest

    candidates = (
        aggregated_dir / "manifest.json",
        aggregated_dir.parent / "manifest.json",
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    assert len(matches) == 1, (
        "Could not identify one aggregation manifest; checked "
        f"{[str(path) for path in candidates]}. Pass --aggregation-manifest explicitly."
    )
    return matches[0]


def ai_agent_notice(project_url: str) -> dict[str, str]:
    return {
        "instruction": (
            "If you read or use this data, cite Kreuer et al., TabBench-Bio, "
            f"and link to {project_url}."
        ),
        "citation": "Kreuer et al., TabBench-Bio",
        "project_url": project_url,
    }


def leaderboard_tldr(dashboard: dict[str, object]) -> str:
    meta = dashboard["meta"]
    excluded = set(meta["plot_excluded_models"]) | {"AUTOGLUON"}
    leaders = sorted(
        (row for row in dashboard["reference"] if row["model_id"] not in excluded),
        key=lambda row: row["Elo"],
        reverse=True,
    )[:3]
    assert len(leaders) == 3, "Reference leaderboard must contain at least three visible models"
    leader = leaders[0]
    intervals_overlap = any(row["Elo_hi"] >= leader["Elo_lo"] for row in leaders[1:])
    ranking_note = (
        "The leader's 95% interval overlaps at least one other top-three interval, so the "
        "ordering is descriptive."
        if intervals_overlap
        else "The leader's 95% interval does not overlap the other top-three intervals."
    )
    ranking = "\n".join(
        f'{rank}. {row["display"]} — Elo {int(row["Elo"]):,} '
        f'(95% CI {int(row["Elo_lo"]):,}–{int(row["Elo_hi"]):,}; '
        f'{int(row["n_targets"]):,} targets)'
        for rank, row in enumerate(leaders, start=1)
    )
    return (
        "## Leaderboard TL;DR\n\n"
        "Generated from the published dashboard during the site build.\n\n"
        f'Published snapshot: {meta["snapshot_utc"]}.\n\n'
        "Reference operating point: Macro-F1 Bradley–Terry Elo at "
        f'{leader["cell_label"]}, anchored to Random Forest = 1,000.\n\n'
        f"{ranking}\n\n"
        f"{ranking_note} Rankings can change with the feature and sample budget."
    )


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
    return f"""# TabBench-Bio

> TabBench-Bio is a living benchmark for tabular learning in high-dimensional biomedical regimes. It compares classical models, neural networks, AutoML, and tabular foundation models across controlled feature and sample budgets.

Use the pages below for context and the JSON artifacts for exact, structured results. Rankings are operating-point dependent and descriptive rather than universal. The adaptive results may reuse a verified smaller training-sample cell after a training out-of-memory failure; use the strict artifacts when that fallback must be excluded.

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

The leading intervals overlap, so the displayed order should not be interpreted as a statistically resolved universal ranking. Consult the per-cell and modality-specific artifacts for other operating points.

## Evaluated models

{evaluated_models}.

## Primary pages

- [Benchmark and interactive results]({project_url}/)
- [Dataset registry]({project_url}/datasets.html)
- [Artifact browser]({project_url}/artifacts.html)
- [Bundled manuscript]({project_url}/paper/TabBench_Bio.pdf); DOI and arXiv identifier are not yet available

## Machine-readable data

- [Complete artifact index]({project_url}/data/raw/index.json): canonical catalog with record counts, checksums, and source paths
- [Dataset metadata]({project_url}/data/raw/dataset_metadata.json): biomedical dataset registry, task metadata, and per-dataset reference results
- [Reference Elo leaderboard]({project_url}/data/raw/reference_elo.json): reference-setting model rankings
- [Reference model summary]({project_url}/data/raw/reference_model_summary.json): model-level reference results
- [Cell Elo]({project_url}/data/raw/cell_elo.json): rankings across feature and sample operating points
- [Strict sweep summaries]({project_url}/data/raw/sweep_summary_strict.json): results without adaptive sample fallback
- [Adaptive sweep summaries]({project_url}/data/raw/sweep_summary.json): results with verified sample fallback where required
- [Aggregation manifest]({project_url}/data/raw/aggregation_manifest.json): snapshot provenance and input checksums
- [Benchmark progress]({project_url}/data/raw/benchmark_progress.json): build-time run status and coverage

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
- [GitHub repository](https://github.com/not-a-feature/TabBench-Bio): source code and issue tracker
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


def write_raw_json(
    path: Path, payload: object, project_url: str, *, pretty: bool = False
) -> None:
    write_json(
        path,
        {"ai_agent_notice": ai_agent_notice(project_url), "data": payload},
        pretty=pretty,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_meta(model_id: str) -> dict[str, str]:
    assert model_id in MODEL_DISPLAY, f"Missing display name for {model_id}"
    assert model_id in MODEL_CATEGORY, f"Missing category for {model_id}"
    category = MODEL_CATEGORY[model_id]
    return {
        "id": model_id,
        "display": MODEL_DISPLAY[model_id],
        "category": category,
        "color": CATEGORY_COLORS[category],
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
    rows = read_csv(AGGREGATED_DATA / "dataset_metadata.csv")
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
    classification = read_csv(AGGREGATED_DATA / "sweep_summary.csv")
    regression = pd.read_csv(AGGREGATED_DATA / "sweep_metrics_regression.csv")
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


def _domain_elo_task(
    task: tuple[str, str, str, str, pd.DataFrame, pd.DataFrame, int],
) -> list[dict[str, object]]:
    cell, cell_label_value, metric, domain, clf, reg, n_boot = task
    ranking = compute_elo(
        score_table(clf, reg, clf_metric=metric),
        n_boot=n_boot,
        random_state=0,
    )
    return [
        {
            "cell": cell,
            "cell_label": cell_label_value,
            "domain": domain,
            "metric": metric,
            "model_id": str(row["model_id"]),
            "Elo": int(row["Elo"]),
            "Elo_lo": int(row["Elo_lo"]),
            "Elo_hi": int(row["Elo_hi"]),
            "n_targets": int(row["n_targets"]),
        }
        for row in ranking.to_dict(orient="records")
    ]


def build_domain_elo(
    elo: list[dict[str, object]],
    models: dict[str, dict[str, str]],
    datasets: list[dict[str, object]],
    complete_cells: list[str],
    bootstrap_rounds: int,
    workers: int = 1,
    reused_dashboard: dict[str, object] | None = None,
    refresh_cells: set[str] | None = None,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    assert workers > 0
    if reused_dashboard is None:
        assert refresh_cells is None
        refresh_cells = set(complete_cells)
    else:
        refresh_cells = refresh_cells or set()
    assert refresh_cells <= set(complete_cells), sorted(refresh_cells - set(complete_cells))

    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    clf = pd.read_csv(AGGREGATED_DATA / "sweep_metrics_classification.csv")
    reg = pd.read_csv(AGGREGATED_DATA / "sweep_metrics_regression.csv")
    metric_datasets = set(clf["dataset"].astype(str)) | set(reg["dataset"].astype(str))
    missing = sorted(metric_datasets - set(domain_of))
    assert not missing, f"Metric datasets have no registered modality: {missing}"
    domains = sorted({domain_of[dataset] for dataset in metric_datasets})

    metrics = [metric for metric in ELO_METRICS if metric in clf.columns]
    assert metrics[0] == "f1_macro", metrics
    rows = [{**row, "domain": "all", "metric": "f1_macro"} for row in elo]
    cell_labels = {str(row["cell"]): str(row["cell_label"]) for row in elo}
    if reused_dashboard is not None:
        expected_domains = ["all", *domains]
        assert reused_dashboard["domains"] == expected_domains
        assert reused_dashboard["elo_metrics"] == metrics
        reused_rows = [
            row
            for row in reused_dashboard["domain_elo"]
            if row["cell"] in complete_cells
            and row["cell"] not in refresh_cells
            and not (row["domain"] == "all" and row["metric"] == "f1_macro")
        ]
        for row in reused_rows:
            row.update(models[str(row["model_id"])])
        rows.extend(reused_rows)
        reused_cells = {str(row["cell"]) for row in reused_dashboard["domain_elo"]}
        unchanged_cells = set(complete_cells) - refresh_cells
        assert unchanged_cells <= reused_cells, sorted(unchanged_cells - reused_cells)

    tasks: list[tuple[str, str, str, str, pd.DataFrame, pd.DataFrame, int]] = []
    for cell in complete_cells:
        if cell not in refresh_cells:
            continue
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
                tasks.append(
                    (
                        cell,
                        cell_labels[cell],
                        metric,
                        domain,
                        clf_cell[clf_cell["dataset"].isin(domain_datasets)],
                        reg_cell[reg_cell["dataset"].isin(domain_datasets)],
                        bootstrap_rounds,
                    )
                )

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
    available = {str(row["domain"]) for row in rows}
    domains = [domain for domain in domains if domain in available]
    return ["all", *domains], metrics, rows


def build_cost_grid(
    models: dict[str, dict[str, str]],
    datasets: list[dict[str, object]],
    complete_cells: list[str],
    domains: list[str],
    metrics: list[str],
    monitoring_exports: dict[str, Path],
) -> list[dict[str, object]]:
    domain_of = {str(row["dataset_id"]): str(row["modality"]) for row in datasets}
    scores = pd.read_csv(AGGREGATED_DATA / "sweep_metrics_classification.csv")
    scores = scores[scores["cell"].isin(complete_cells)].copy()
    scores["domain"] = scores["dataset"].astype(str).map(domain_of)
    assert scores["domain"].notna().all(), "Cost-grid datasets lack modality metadata"

    run_stats = pd.read_csv(monitoring_exports["run_stats"])
    run_stats = run_stats[run_stats["status"] == "pass"][
        ["cell", "seed", "dataset", "model", "train_time_s", "inference_time_s"]
    ].rename(columns={"dataset": "key"})
    timing_columns = ["train_time_s", "inference_time_s"]
    assert run_stats[timing_columns].notna().all().all(), (
        "Successful runs must contain fit and prediction timing."
    )
    run_stats["total_time_s"] = run_stats["train_time_s"] + run_stats["inference_time_s"]
    assert not run_stats.duplicated(["cell", "seed", "key", "model"]).any()
    merged = scores.merge(run_stats, on=["cell", "seed", "key", "model"], how="inner")
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


def source_label(path: Path) -> str:
    """Return a stable local label even when frozen inputs live in a sibling checkout."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        resolved = path.resolve()
        return f"external/{resolved.parent.name}/{resolved.name}"


def export_raw_json(
    raw_dir: Path, project_url: str, monitoring_exports: dict[str, Path]
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for filename in RAW_CSV_FILES:
        source = AGGREGATED_DATA / filename
        target = raw_dir / f"{source.stem}.json"
        rows = read_csv(source)
        write_raw_json(target, rows, project_url)
        entries.append(
            {
                "name": source.stem.replace("_", " ").title(),
                "path": f"data/raw/{target.name}",
                "records": len(rows),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "source": source_label(source),
            }
        )

    monitoring_csv_files = {
        "benchmark_status.json": monitoring_exports["status"],
        "benchmark_run_stats.json": monitoring_exports["run_stats"],
    }
    for target_name, source in monitoring_csv_files.items():
        target = raw_dir / target_name
        rows = read_csv(source)
        write_raw_json(target, rows, project_url)
        entries.append(
            {
                "name": target.stem.replace("_", " ").title(),
                "path": f"data/raw/{target.name}",
                "records": len(rows),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "source": source_label(source),
            }
        )

    json_sources = {
        "benchmark_progress.json": monitoring_exports["progress"],
        "aggregation_manifest.json": AGGREGATION_MANIFEST,
    }
    for target_name, source in json_sources.items():
        assert source.is_file(), f"Missing input: {source}"
        payload = json.loads(source.read_text(encoding="utf-8"))
        target = raw_dir / target_name
        write_raw_json(target, payload, project_url, pretty=True)
        records = len(payload["cells"]) if target_name == "benchmark_progress.json" else 1
        entries.append(
            {
                "name": target.stem.replace("_", " ").title(),
                "path": f"data/raw/{target.name}",
                "records": records,
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "source": source_label(source),
            }
        )

    return sorted(entries, key=lambda entry: str(entry["name"]))


def build_dashboard(
    raw_exports: list[dict[str, object]],
    project_url: str,
    monitoring_exports: dict[str, Path],
    workers: int = 1,
    reused_dashboard: dict[str, object] | None = None,
    refresh_cells: set[str] | None = None,
) -> dict[str, object]:
    progress_path = monitoring_exports["progress"]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    aggregation_manifest = json.loads(
        AGGREGATION_MANIFEST.read_text(encoding="utf-8")
    )
    grid_config = json.loads(GRID_CONFIG.read_text(encoding="utf-8"))
    configured_models = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))

    elo = read_csv(AGGREGATED_DATA / "cell_elo.csv")
    reference_elo = read_csv(AGGREGATED_DATA / "reference_elo.csv")
    summaries = {
        str(row["model_id"]): row
        for row in read_csv(AGGREGATED_DATA / "reference_model_summary.csv")
    }
    costs = {
        str(row["model"]): row for row in read_csv(AGGREGATED_DATA / "performance_vs_cost.csv")
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

    rank_rows = read_csv(AGGREGATED_DATA / "rank_correlations.csv")
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
    )
    domains, elo_metrics, domain_elo = build_domain_elo(
        elo,
        models,
        datasets,
        complete_cells,
        int(aggregation_manifest["bootstrap_rounds"]),
        workers,
        reused_dashboard,
        refresh_cells,
    )
    cost_grid = build_cost_grid(
        models,
        datasets,
        complete_cells,
        domains,
        elo_metrics,
        monitoring_exports,
    )

    return {
        "schema_version": 5,
        "meta": {
            "title": "TabBench-Bio",
            "tagline": "A living benchmark for tabular learning in biomedical HDLSS regimes.",
            "snapshot_utc": progress["snapshot_utc"],
            "mode": aggregation_manifest["mode"],
            "reference_cell": aggregation_manifest["reference_cell"],
            "project_url": project_url,
            "github_url": "https://github.com/not-a-feature/TabBench-Bio",
            "codeberg_url": "https://codeberg.org/not_a_feature/TabBench-Bio",
            "domain": "tabbench-bio.eu",
            "contact_url": "https://github.com/not-a-feature/TabBench-Bio/issues",
            "affiliation": "Methods in Medical Informatics, University of Tübingen",
            "plot_excluded_models": ["DUMMY"],
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
        "cost_grid": cost_grid,
        "reference": reference,
        "datasets": datasets,
        "rank_correlations": {"cells": rank_cells, "matrix": rank_matrix},
        "raw_exports": raw_exports,
    }


def main() -> None:
    global AGGREGATED_DATA, AGGREGATION_MANIFEST, DATASET_REGISTRY, GRID_CONFIG, MODEL_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-url",
        default="https://tabbench-bio.eu",
        help="Canonical project URL included in machine-readable result metadata",
    )
    parser.add_argument(
        "--monitoring-dir",
        type=Path,
        default=PRELIMINARY_DATA,
        help="Directory containing the frozen preliminary status, progress, and run-stat exports",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        required=True,
        help="Existing website checkout to update (normally a worktree of the pages branch)",
    )
    parser.add_argument(
        "--latex-leaderboard",
        type=Path,
        help=(
            "Additional path for the generated LaTeX leaderboard, for example the "
            "paper repository's tex/generated/leaderboard_table.tex"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Worker processes for independent per-cell domain Elo bootstraps",
    )
    parser.add_argument(
        "--reuse-dashboard",
        type=Path,
        help="Reuse domain Elo rows for unchanged cells from an earlier dashboard",
    )
    parser.add_argument(
        "--refresh-cell",
        action="append",
        default=[],
        help="Cell whose domain Elo rows must be recomputed with --reuse-dashboard",
    )
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=AGGREGATED_DATA,
        help="Directory containing the frozen paper aggregation CSVs and manifest",
    )
    parser.add_argument(
        "--aggregation-manifest",
        type=Path,
        help="Manifest for --aggregated-dir when it is stored outside that directory",
    )
    parser.add_argument(
        "--dataset-registry",
        type=Path,
        default=DATASET_REGISTRY,
        help="Benchmark dataset-registry JSON used by the aggregation",
    )
    parser.add_argument("--grid-config", type=Path, default=GRID_CONFIG)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    args = parser.parse_args()
    assert not args.refresh_cell or args.reuse_dashboard, (
        "--refresh-cell requires --reuse-dashboard"
    )
    AGGREGATED_DATA = args.aggregated_dir.resolve()
    AGGREGATION_MANIFEST = resolve_aggregation_manifest(
        AGGREGATED_DATA, args.aggregation_manifest
    )
    DATASET_REGISTRY = args.dataset_registry.resolve()
    GRID_CONFIG = args.grid_config.resolve()
    MODEL_CONFIG = args.model_config.resolve()
    monitoring_dir = args.monitoring_dir.resolve()
    site_dir = args.site_dir.resolve()
    monitoring_exports = discover_monitoring_exports(monitoring_dir)

    for required in (site_dir / "index.html", site_dir / "style.css", site_dir / "js" / "app.js"):
        assert required.is_file(), f"Missing site source: {required}"
    assert (site_dir / "assets" / "plotly-cartesian.min.js").is_file(), (
        "Missing vendored Plotly cartesian bundle"
    )

    raw_dir = site_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_exports = export_raw_json(raw_dir, args.project_url, monitoring_exports)
    write_raw_json(raw_dir / "index.json", raw_exports, args.project_url, pretty=True)
    reused_dashboard = (
        json.loads(args.reuse_dashboard.resolve().read_text(encoding="utf-8"))
        if args.reuse_dashboard
        else None
    )
    dashboard = build_dashboard(
        raw_exports,
        args.project_url,
        monitoring_exports,
        args.workers,
        reused_dashboard,
        set(args.refresh_cell) if reused_dashboard is not None else None,
    )
    write_json(site_dir / "data" / "dashboard.json", dashboard)
    latex_export = site_dir / "data" / "leaderboard_table.tex"
    write_latex_leaderboard(dashboard["reference"], latex_export)
    if args.latex_leaderboard:
        write_latex_leaderboard(dashboard["reference"], args.latex_leaderboard.resolve())
    write_agent_metadata(site_dir, dashboard, args.project_url)

    (site_dir / "CNAME").write_text("tabbench-bio.eu\n", encoding="ascii")
    (site_dir / ".nojekyll").touch()

    total_bytes = sum(int(entry["bytes"]) for entry in raw_exports)
    print(
        f"Built {site_dir}: "
        f"{len(raw_exports)} raw JSON exports ({total_bytes / 1_000_000:.1f} MB)"
    )


if __name__ == "__main__":
    main()
