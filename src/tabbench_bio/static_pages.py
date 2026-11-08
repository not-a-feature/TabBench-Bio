"""Build crawlable entity pages while retaining the interactive chart templates."""

import json
import math
import re
from html import escape
from pathlib import Path


def entity_slug(identifier: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")
    if not slug:
        raise ValueError(f"Identifier has no URL-safe characters: {identifier!r}")
    return slug


def detail_url(kind: str, identifier: str) -> str:
    return f"/{kind}s/{entity_slug(identifier)}/"


def replace_element(html: str, element_id: str, content: str) -> str:
    pattern = rf'(<(?P<tag>\w+)\b[^>]*\bid="{element_id}"[^>]*>).*?(</(?P=tag)>)'
    html, count = re.subn(pattern, lambda match: match[1] + content + match[3], html, flags=re.S)
    assert count == 1, (element_id, count)
    return html


def rooted_html(html: str) -> str:
    """Resolve template assets/navigation from the site root at every folder depth."""

    def root(match):
        value = match[2]
        if value.startswith(("/", "#")) or re.match(r"[a-zA-Z][\w+.-]*:", value):
            return match[0]
        return match[1] + "/" + value + '"'

    return re.sub(r'((?:href|src)=")([^"]+)"', root, html)


def page_metadata(html: str, title: str, description: str, canonical: str) -> str:
    html = re.sub(r"<title>.*?</title>", lambda _: f"<title>{escape(title)}</title>", html)
    html = re.sub(
        r'<meta name="description"\s+content="[^"]*">',
        lambda _: f'<meta name="description" content="{escape(description, quote=True)}">',
        html,
    )
    canonical_tag = f'<link rel="canonical" href="{escape(canonical, quote=True)}">'
    if re.search(r'<link rel="canonical"[^>]*>', html):
        html = re.sub(r'<link rel="canonical"[^>]*>', lambda _: canonical_tag, html)
    else:
        html = html.replace("</head>", canonical_tag + "\n</head>")
    return html


def format_score(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def model_label(model: dict) -> str:
    return (
        model["display"]
        + (" †" if model.get("training_data_overlap") else "")
        + (" ‡" if model.get("classification_only") else "")
    )


def model_page(template: str, identifier: str, dashboard: dict, project_url: str) -> str:
    model = dashboard["models"][identifier]
    title = f"{model['display']} on biomedical data · TabBench-Bio"
    description = model.get("description") or (
        f"Compare {model['display']} ({model['category']}) on biomedical datasets: "
        "benchmark results, uncertainty and failure rates across feature and sample budgets."
    )
    html = page_metadata(
        template, title, description, project_url + detail_url("model", identifier)
    )
    html = html.replace(
        'data-page="model"', f'data-page="model" data-entity-id="{escape(identifier, quote=True)}"'
    )
    for element_id, content in (
        ("model-page-title", model_label(model)),
        ("model-card-name", model_label(model)),
        ("model-card-family", model["category"]),
        ("model-description", description),
    ):
        html = replace_element(html, element_id, escape(content))
    if model.get("classification_only"):
        html = replace_element(
            html,
            "model-card-classification-warning",
            "‡ Classification only; regression tasks are not supported.",
        )
        html = html.replace(
            'id="model-card-classification-warning" class="model-card-limit-warning" hidden',
            'id="model-card-classification-warning" class="model-card-limit-warning"',
        )
    limit = model.get("regular_max_features")
    html = replace_element(
        html,
        "model-card-limit",
        f"Regular feature limit: {limit:,}" if limit else "No regular feature limit specified",
    )
    source = model.get("source_url")
    if source:
        html = re.sub(
            r'(<a id="model-card-source"[^>]*?)\s+hidden>',
            lambda m: m[1] + f' href="{escape(source, quote=True)}">',
            html,
            flags=re.S,
        )
    view = dashboard["analysis_views"]["strict"]
    rows = sorted(
        (
            row
            for row in view["domain_elo"]
            if row["model_id"] == identifier
            and row["cell"] == dashboard["meta"]["reference_cell"]
            and row["metric"] == "f1_macro"
        ),
        key=lambda row: (row["domain"] != "all", row["domain"]),
    )
    summary = (
        '<details class="dataset-exact-scores"><summary id="static-results-title">'
        "Reference benchmark results</summary>"
    )
    if model.get("training_data_overlap"):
        summary += "<p>Training-data overlap: part of this benchmark was used in this model's training.</p>"
    if rows:
        summary += (
            f"<p>Strict Macro-F1 Elo at {escape(rows[0]['cell_label'])}. "
            "Random Forest is anchored at 1,000; intervals are target-bootstrap 95% intervals. "
            "These reference results stay fixed while you explore other budgets in the charts. "
            "Overlapping intervals do not establish a clear winner.</p>"
            '<div class="table-wrap"><table><thead><tr><th>Dataset modality</th><th>Elo</th>'
            "<th>95% interval</th><th>Targets</th></tr></thead><tbody>"
        )
        summary += (
            "".join(
                f"<tr><td>{escape('All modalities' if row['domain'] == 'all' else row['domain'])}</td>"
                f"<td>{row['Elo']:,}</td><td>{row['Elo_lo']:,}–{row['Elo_hi']:,}</td>"
                f"<td>{row['n_targets']}</td></tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )
    else:
        summary += "<p>No Macro-F1 Elo estimate is available for this model at the reference operating point.</p>"
    summary += "</details>"
    html = html.replace("  </main>", summary + "\n  </main>", 1)
    return rooted_html(html)


def dataset_page(
    template: str, dataset: dict, registry: dict, scores: list[dict], project_url: str
) -> str:
    identifier = dataset["dataset_id"]
    description = dataset.get("description") or (
        f"{dataset['display_name']}: {dataset['modality']} {dataset['task'].lower()} "
        f"for {dataset['target']}. Compare machine learning models across feature and sample budgets."
    )
    html = page_metadata(
        template,
        f"{dataset['display_name']} model comparison · TabBench-Bio",
        description,
        project_url + detail_url("dataset", identifier),
    )
    html = html.replace(
        'data-page="dataset"',
        f'data-page="dataset" data-entity-id="{escape(identifier, quote=True)}"',
    )
    for element_id, value in (
        ("detail-title", dataset["display_name"]),
        ("detail-modality", dataset["modality"]),
        (
            "detail-task",
            "Binary classification" if dataset.get("problem_type") == "binary" else dataset["task"],
        ),
        ("detail-license", "Licence: " + (dataset.get("license") or "Not specified")),
        ("detail-id", identifier),
        ("detail-target", dataset["target"]),
        ("dataset-description", description),
    ):
        html = replace_element(html, element_id, escape(value))
    source = dataset.get("source_url")
    html = replace_element(html, "detail-source", escape(dataset["source"]) + " ↗")
    html = re.sub(
        r'(<a id="detail-source"[^>]*)(>)',
        lambda m: m[1] + (f' href="{escape(source, quote=True)}"' if source else " hidden") + m[2],
        html,
    )
    cell = dataset["default_cell"]
    option = next(row for row in registry["cell_options"] if row["id"] == cell)
    metric = dataset["metrics"][0]
    models = registry["models"]
    excluded = set(registry["meta"]["plot_excluded_models"])
    rows = [
        row
        for row in scores
        if row["cell"] == cell
        and row["model_id"] not in excluded
        and isinstance(row["values"].get(metric["key"]), (int, float))
        and math.isfinite(row["values"][metric["key"]])
    ]
    rows.sort(
        key=lambda row: (
            (1 if metric["better"] == "low" else -1) * row["values"][metric["key"]],
            row["model_id"],
        )
    )
    point = f"{option['label']} · {'Lower' if metric['better'] == 'low' else 'Higher'} {metric['label']} is better"
    html = replace_element(html, "detail-point", escape(point))
    html = replace_element(
        html,
        "dataset-score-head",
        "<tr><th>Model</th>"
        + "".join(f"<th>{escape(item['label'])}</th>" for item in dataset["metrics"])
        + "</tr>",
    )
    table = "".join(
        '<tr><td><a href="'
        + detail_url("model", row["model_id"])
        + '">'
        + escape(model_label(models[row["model_id"]]))
        + "</a></td>"
        + "".join(
            f"<td>{format_score(row['values'].get(item['key']))}</td>"
            for item in dataset["metrics"]
        )
        + "</tr>"
        for row in rows
    )
    html = replace_element(html, "dataset-score-body", table)
    html = html.replace(
        '<details class="dataset-exact-scores">', '<details class="dataset-exact-scores" open>'
    )
    html = html.replace(
        'id="detail-loading" role="status"', 'id="detail-loading" role="status" hidden'
    )
    if not rows:
        html = html.replace(
            'id="detail-point-empty" class="dataset-empty" hidden',
            'id="detail-point-empty" class="dataset-empty"',
        )
    return rooted_html(html)


def write_static_pages(site_dir: Path, dashboard: dict, project_url: str) -> None:
    registry = json.loads((site_dir / "data/datasets/index.json").read_text(encoding="utf-8"))
    for kind, identifiers in (
        ("model", list(dashboard["models"])),
        ("dataset", [row["dataset_id"] for row in registry["datasets"]]),
    ):
        slugs = [entity_slug(identifier) for identifier in identifiers]
        if len(set(slugs)) != len(slugs):
            raise ValueError(f"Duplicate {kind} URL slugs")
        directory = site_dir / f"{kind}s"
        directory.mkdir(exist_ok=True)
        # Remove only pages listed in our previous generated manifest.
        manifest = directory / ".generated-pages.json"
        previous = json.loads(manifest.read_text()) if manifest.exists() else []
        for slug in set(previous) - set(slugs):
            if slug != entity_slug(slug):
                raise ValueError(f"Invalid generated page slug: {slug!r}")
            (directory / slug / "index.html").unlink(missing_ok=True)
        manifest.write_text(json.dumps(slugs), encoding="utf-8")
        overview = (site_dir / f"{kind}s.html").read_text(encoding="utf-8")
        overview = re.sub(
            r'<link rel="canonical"[^>]*>',
            f'<link rel="canonical" href="{project_url}/{kind}s/">',
            overview,
        )
        (directory / "index.html").write_text(rooted_html(overview), encoding="utf-8")
    model_template = (site_dir / "model.html").read_text(encoding="utf-8")
    dataset_template = (site_dir / "dataset.html").read_text(encoding="utf-8")
    for identifier in dashboard["models"]:
        path = site_dir / detail_url("model", identifier).strip("/") / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            model_page(model_template, identifier, dashboard, project_url), encoding="utf-8"
        )
    for dataset in registry["datasets"]:
        payload = json.loads(
            (site_dir / "data/datasets" / dataset["scores_file"]).read_text(encoding="utf-8")
        )
        assert payload["dataset_id"] == dataset["dataset_id"]
        path = site_dir / detail_url("dataset", dataset["dataset_id"]).strip("/") / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            dataset_page(dataset_template, dataset, registry, payload["scores"], project_url),
            encoding="utf-8",
        )
