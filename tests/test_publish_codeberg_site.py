"""Publishing includes every generated asset and excludes source repositories."""

import pytest

from scripts import publish_codeberg_site as publisher


def test_publish_copies_the_complete_snapshot(tmp_path, monkeypatch):
    site = tmp_path / "site"
    files = {
        "index.html": "index",
        ".nojekyll": "",
        "CNAME": "tabbench-bio.eu",
        "data/dashboard.json": "{}",
        "data/leaderboard.json": "rankings",
        "skill.md": "skill",
        "skills/biomedical-tabular-model-selection/references/data.md": "reference",
        "js/app.js": "app",
        "assets/og.png": "image",
    }
    for relative, content in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(publisher.tempfile, "tempdir", str(tmp_path))
    calls = []

    def fake_run(*command, cwd=None):
        calls.append(command)
        if command[:3] == ("git", "add", "--all"):
            for relative, content in files.items():
                assert (cwd / relative).read_text(encoding="utf-8") == content
        if command[:3] == ("git", "diff", "--cached"):
            return "data/leaderboard.json\nskill.md"
        return "commit"

    monkeypatch.setattr(publisher, "run", fake_run)
    publisher.validate_site(site)
    assert publisher.publish(site, publisher.DEFAULT_REMOTE) == "commit"
    assert calls[-1] == ("git", "push", publisher.DEFAULT_REMOTE, "HEAD:refs/heads/pages")
    assert (site / "data/leaderboard.json").read_text(encoding="utf-8") == "rankings"


def test_publisher_rejects_source_checkout(tmp_path):
    for name in publisher.REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "src").mkdir()
    with pytest.raises(AssertionError, match="source directories"):
        publisher.validate_site(tmp_path)
