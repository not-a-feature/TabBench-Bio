"""Generate the complete public agent guide and search metadata from current results."""

from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

SITEMAP_PATHS = (
    "",
    "datasets.html",
    "artifacts.html",
    "changelog.html",
    "llms.txt",
    "skill.md",
    "citation.html",
    "CITATION.cff",
    "CITATION.bib",
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
    reference_label = next(
        cell["label"] for cell in dashboard["cell_options"] if cell["id"] == meta["reference_cell"]
    )
    leaders = (
        "\n".join(
            f"{rank}. {row['display']}: Elo {int(row['Elo']):,} "
            f"(95% CI {int(row['Elo_lo']):,}–{int(row['Elo_hi']):,}; "
            f"{int(row['n_targets']):,} targets)"
            for rank, row in enumerate(reference[:5], start=1)
        )
        or "No comparable completed folds with Random Forest yet."
    )
    evaluated_models = ", ".join(
        sorted(str(row["display"]) for row in dashboard["models"].values())
    )
    flagged_models = sorted(
        str(model["display"])
        for model in dashboard["models"].values()
        if model["training_data_overlap"]
    )
    overlap_note = ""
    if flagged_models:
        disclaimers = "\n".join(
            f"- † {name}: Part of the benchmark training data was used in the training process of this model."
            for name in flagged_models
        )
        overlap_note = (
            "## Training-data overlap\n\n"
            + disclaimers
            + "\n\nThese models remain in the benchmark results but are excluded from the Reference leaders podium and social card.\n\n"
        )
    status = progress["status"]
    paper_url = str(meta["paper_url"])
    if not paper_url.startswith(("https://", "http://")):
        paper_url = f"{project_url}/{paper_url.lstrip('/')}"
    sqlite_export = next(row for row in dashboard["raw_exports"] if row["format"] == "sqlite3")
    sqlite_line = (
        f"- [Canonical results SQLite]({sqlite_export['path']}): "
        f"{int(sqlite_export['records']):,} attempts, "
        f"SHA-256 `{sqlite_export['sha256']}`"
        if sqlite_export["available"]
        else (
            "- Results SQLite: download link not configured; "
            f"{int(sqlite_export['bytes']) / 1_000_000:.1f} MB, "
            f"SHA-256 `{sqlite_export['sha256']}`"
        )
    )
    return f"""# TabBench-Bio

> TabBench-Bio is a living benchmark for tabular learning in high-dimensional biomedical regimes. It compares classical models, neural networks, AutoML, and tabular foundation models across controlled feature and sample budgets.

Use the pages below for context and the canonical SQLite database for exact, structured results. Rankings are operating-point dependent and descriptive rather than universal. The website opens in Strict (primary) mode by default, matching the paper's primary analysis and using nominal-cell results without substitution. Dashboard and leaderboard top-level rankings follow this Strict default. Adaptive sensitivity results may reuse a verified smaller training-sample cell after a memory failure and are not measurements at the requested larger sample size. All explicit analysis_views remain available. The Failures-omitted view drops failed runs instead of scoring them at chance, which removes a model from those targets, so its ranking is computed on differing target pools and reads as an optimistic bound. Dataset-specific score exports use Strict results.

## Results

- Generated from the result bundle at {meta["snapshot_utc"]}.
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

{overlap_note}## Agent skill

Here is a skill for your agent: [TabBench-Bio model selection]({project_url}/skill.md).
Use it to match task, modality, budgets and compute constraints, and interpret uncertainty and training-data overlap.
Download the Markdown file directly; no repository clone is needed.

## Primary pages

- [Benchmark and interactive results]({project_url}/)
- [Dataset registry]({project_url}/datasets.html)
- [Artifact browser]({project_url}/artifacts.html)
- [Manuscript]({paper_url})

## Machine-readable data

- [Dashboard JSON]({project_url}/data/dashboard.json): modality-specific rankings, costs, coverage, analysis views and model flags.
- [Leaderboard JSON]({project_url}/data/leaderboard.json): aggregate rankings.
- [Dataset index JSON]({project_url}/data/datasets/index.json): task-specific metrics and links to per-dataset scores.

{sqlite_line}

## Citation

If you read or use the benchmark, website, or published result artifacts, cite:

Kreuer, J.; Ouaari, S.; Hellmig, J.; Braitinger, J.; Pfeifer, N. (2026). TabBench-Bio: A Living Benchmark for Machine Learning on High-Dimensional Biomedical Tables. arXiv:2609.07441. https://doi.org/10.48550/arXiv.2609.07441

- [Citation File Format metadata]({project_url}/CITATION.cff)
- Citation target: [TabBench-Bio paper](https://doi.org/10.48550/arXiv.2609.07441)
- Authors: Jules Kreuer, Sofiane Ouaari, Julia Hellmig, Julius Braitinger, and Nico Pfeifer
- DOI: [10.48550/arXiv.2609.07441](https://doi.org/10.48550/arXiv.2609.07441)
- arXiv: [2609.07441](https://arxiv.org/abs/2609.07441)

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
        "<!-- Generated by tabbench-bio leaderboard; do not edit manually. -->\n"
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )


def write_agent_metadata(site_dir: Path, dashboard: dict) -> None:
    project_url = "https://" + dashboard["meta"]["domain"]
    (site_dir / "llms.txt").write_text(build_llms_text(dashboard, project_url), encoding="utf-8")
    (site_dir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {project_url}/sitemap.xml\n", encoding="utf-8"
    )
    (site_dir / "sitemap.xml").write_text(
        build_sitemap(project_url, dashboard["meta"]["snapshot_utc"][:10]), encoding="utf-8"
    )
