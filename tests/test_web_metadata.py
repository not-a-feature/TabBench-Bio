"""Search metadata covers real entity pages, consistently across sitemap formats."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

from tabbench_bio import web_metadata


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
    }
    (data / "index.json").write_text(json.dumps({"datasets": [dataset]}), encoding="utf-8")
    models = {
        "RF": {
            "display": "Random Forest",
            "category": "Tree-based",
            "source_url": "https://github.com/scikit-learn/scikit-learn",
        },
        "custom/α": {"display": "Custom <model>", "category": "Custom", "source_url": None},
    }
    domain = "https://example.org"
    web_metadata.write_dataset_registry_summary(tmp_path)
    web_metadata.write_model_registry_summary(tmp_path, {"models": models})
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
    relative = "dataset.html?" + urlencode({"id": dataset["dataset_id"]})
    assert f"{domain}/{relative}" in text_urls
    assert f'href="{relative}"' in (tmp_path / "datasets.html").read_text(encoding="utf-8")
    assert not list(tmp_path.glob("dataset-*.html"))
    overview = (tmp_path / "models.html").read_text(encoding="utf-8")
    assert (
        'href="https://github.com/scikit-learn/scikit-learn" target="_blank" rel="noreferrer">Source ↗</a>'
        in overview
    )
    assert overview.count(">Source ↗</a>") == 1
    for identifier in models:
        relative = "model.html?" + urlencode({"model": identifier})
        assert f"{domain}/{relative}" in text_urls
        assert f'href="{relative}"' in overview
    assert not list(tmp_path.glob("model-*.html"))
    robots = (tmp_path / "robots.txt").read_text()
    assert f"Sitemap: {domain}/sitemap.txt\n" in robots
    assert f"Sitemap: {domain}/sitemap.xml\n" in robots
    del models["custom/α"]
    web_metadata.write_model_registry_summary(tmp_path, {"models": models})
    web_metadata.write_sitemaps(tmp_path, domain, "2026-11-02")
    assert not list(tmp_path.glob("model-*.html"))
    assert (
        domain + "/model.html?" + urlencode({"model": "custom/α"})
        not in (tmp_path / "sitemap.txt").read_text()
    )
