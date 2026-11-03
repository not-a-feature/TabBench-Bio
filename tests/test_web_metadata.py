"""Search metadata covers real entity pages, consistently across sitemap formats."""

import json
import shutil
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tabbench_bio import web_metadata
from tabbench_bio.static_pages import detail_url, write_static_pages


def test_registry_links_and_sitemaps_are_consistent(tmp_path):
    source = Path(web_metadata.__file__).parent / "web"
    for name in ("models.html", "model.html", "dataset.html", "datasets.html", "index.html"):
        shutil.copyfile(source / name, tmp_path / name)
    data = tmp_path / "data/datasets"
    data.mkdir(parents=True)
    dataset = {
        "dataset_id": "GEO/example & test",
        "display_name": "Example <dataset>",
        "modality": "Gene expression",
        "task": "Classification",
        "target": "Case & control",
        "source": "geo",
        "description": "Gene expression measurements used to distinguish case & control samples.",
        "source_url": "https://example.org/source?x=1&y=2",
        "default_cell": "cap_100_n20",
        "metrics": [{"key": "f1_macro", "label": "Macro-F1", "better": "high"}],
        "scores_file": "example.json",
    }
    models = {
        "RF": {
            "display": "Random Forest",
            "category": "Tree-based",
            "description": "Random Forest averages predictions from many trees & their random feature subsets.",
            "source_url": "https://github.com/scikit-learn/scikit-learn",
        },
        "custom/α": {"display": "Custom <model>", "category": "Custom", "source_url": None},
    }
    cell = {"id": "cap_100_n20", "label": "p=100, n=20", "feature_cap": 100, "n_train": 20}
    registry = {
        "datasets": [dataset],
        "models": models,
        "cell_options": [cell],
        "meta": {"plot_excluded_models": []},
    }
    (data / "index.json").write_text(json.dumps(registry), encoding="utf-8")
    (data / "example.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset["dataset_id"],
                "scores": [
                    {"model_id": "RF", "cell": cell["id"], "values": {"f1_macro": 0.8}},
                    {"model_id": "custom/α", "cell": cell["id"], "values": {"f1_macro": 0.9}},
                ],
            }
        ),
        encoding="utf-8",
    )
    dashboard = {
        "models": models,
        "meta": {"reference_cell": cell["id"]},
        "analysis_views": {
            "strict": {
                "domain_elo": [
                    {
                        "model_id": "RF",
                        "cell": cell["id"],
                        "cell_label": cell["label"],
                        "metric": "f1_macro",
                        "domain": "all",
                        "Elo": 1000,
                        "Elo_lo": 950,
                        "Elo_hi": 1050,
                        "n_targets": 1,
                    }
                ]
            }
        },
    }
    domain = "https://example.org"
    web_metadata.write_dataset_registry_summary(tmp_path)
    web_metadata.write_model_registry_summary(tmp_path, {"models": models})
    write_static_pages(tmp_path, dashboard, domain)
    web_metadata.write_sitemaps(tmp_path, domain, "2026-11-02")
    text_urls = (tmp_path / "sitemap.txt").read_text().splitlines()
    xml_urls = [
        node.text
        for node in ET.fromstring((tmp_path / "sitemap.xml").read_bytes()).findall("{*}url/{*}loc")
    ]
    assert text_urls == xml_urls
    assert len(text_urls) == len(set(text_urls))
    assert domain + "/model.html" not in text_urls
    assert domain + "/dataset.html" not in text_urls
    relative = detail_url("dataset", dataset["dataset_id"])
    assert f"{domain}{relative}" in text_urls
    assert f'href="{relative}"' in (tmp_path / "datasets.html").read_text(encoding="utf-8")
    assert 'href="https://example.org/source?x=1&amp;y=2"' in (
        tmp_path / "datasets.html"
    ).read_text(encoding="utf-8")
    assert not list(tmp_path.glob("dataset-*.html"))
    overview = (tmp_path / "models.html").read_text(encoding="utf-8")
    assert (
        'href="https://github.com/scikit-learn/scikit-learn" target="_blank" rel="noreferrer">Source ↗</a>'
        in overview
    )
    assert overview.count(">Source ↗</a>") == 1
    for identifier in models:
        relative = detail_url("model", identifier)
        assert f"{domain}{relative}" in text_urls
        assert f'href="{relative}"' in overview
    assert not list(tmp_path.glob("model-*.html"))
    robots = (tmp_path / "robots.txt").read_text()
    assert f"Sitemap: {domain}/sitemap.txt\n" in robots
    assert f"Sitemap: {domain}/sitemap.xml\n" in robots
    dataset_html = (tmp_path / "datasets/geo-example-test/index.html").read_text(encoding="utf-8")
    model_html = (tmp_path / "models/rf/index.html").read_text(encoding="utf-8")
    assert 'data-entity-id="GEO/example &amp; test"' in dataset_html
    assert '<h1 id="detail-title">Example &lt;dataset&gt;</h1>' in dataset_html
    assert (
        'id="dataset-description">Gene expression measurements used to distinguish case &amp; control samples.</p>'
        in dataset_html
    )
    assert f'href="{domain}/datasets/geo-example-test/"' in dataset_html
    assert "<td>0.9</td>" in dataset_html and "<td>0.8</td>" in dataset_html
    assert dataset_html.index("<td>0.9</td>") < dataset_html.index("<td>0.8</td>")
    assert 'href="/models/custom/"' in dataset_html
    assert 'src="/js/app.js?' in dataset_html
    assert 'href="/style.min.css?' in dataset_html
    assert "Loading dataset" not in dataset_html
    assert '<h1 id="model-page-title">Random Forest</h1>' in model_html
    assert (
        'id="model-description">Random Forest averages predictions from many trees &amp; their random feature subsets.</p>'
        in model_html
    )
    assert "<td>1,000</td><td>950–1,050</td>" in model_html
    assert f'href="{domain}/models/rf/"' in model_html
    assert '<section class="section"' not in model_html
    assert model_html.index('class="dataset-plots"') < model_html.index('id="static-results-title"')
    assert '<details class="dataset-exact-scores"><summary id="static-results-title">' in model_html
    assert 'href="https://github.com/scikit-learn/scikit-learn"' in model_html
    assert "Custom &lt;model&gt;" in (tmp_path / "models/custom/index.html").read_text(
        encoding="utf-8"
    )
    for kind in ("models", "datasets"):
        folder_index = (tmp_path / kind / "index.html").read_text(encoding="utf-8")
        assert f'href="{domain}/{kind}/"' in folder_index
        assert f"{domain}/{kind}/" in text_urls

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"href", "src"}:
                    assert value.startswith(("/", "https:", "#")), value

    for path in [*tmp_path.glob("models/**/index.html"), *tmp_path.glob("datasets/**/index.html")]:
        Links().feed(path.read_text(encoding="utf-8"))

    before = (tmp_path / "models/rf/index.html").read_bytes()
    write_static_pages(tmp_path, dashboard, domain)
    assert (tmp_path / "models/rf/index.html").read_bytes() == before
    del models["custom/α"]
    web_metadata.write_model_registry_summary(tmp_path, {"models": models})
    write_static_pages(tmp_path, dashboard, domain)
    web_metadata.write_sitemaps(tmp_path, domain, "2026-11-02")
    assert not list(tmp_path.glob("model-*.html"))
    assert domain + detail_url("model", "custom/α") not in (tmp_path / "sitemap.txt").read_text()
    assert not (tmp_path / "models/custom/index.html").exists()


def test_entity_url_collisions_fail_instead_of_overwriting(tmp_path):
    source = Path(web_metadata.__file__).parent / "web"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    data = tmp_path / "data/datasets"
    data.mkdir(parents=True)
    (data / "index.json").write_text('{"datasets": []}')
    with pytest.raises(ValueError, match="Duplicate model URL slugs"):
        write_static_pages(tmp_path, {"models": {"A/B": {}, "A-B": {}}}, "https://example.org")
