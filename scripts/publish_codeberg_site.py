"""Publish a built TabBench-Bio site to the Codeberg Pages branch."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_REMOTE = "git@codeberg.org:not_a_feature/TabBench-Bio.git"
REQUIRED_FILES = ("index.html", ".nojekyll", "CNAME", "data/dashboard.json")
FORBIDDEN_DIRECTORIES = (".github", "configs", "scripts", "src", "tests")


def run(*command: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_site(site: Path) -> None:
    assert site.is_dir(), f"Site directory does not exist: {site}"
    for relative in REQUIRED_FILES:
        assert (site / relative).is_file(), f"Missing generated site file: {relative}"
    forbidden = [name for name in FORBIDDEN_DIRECTORIES if (site / name).exists()]
    assert not forbidden, "Refusing to publish source directories: " + ", ".join(forbidden)


def publish(site: Path, remote: str) -> str:
    with tempfile.TemporaryDirectory(prefix="tabbench-bio-pages-") as temporary:
        checkout = Path(temporary)
        run("git", "init", cwd=checkout)
        run("git", "config", "user.name", "TabBench Bio deployment", cwd=checkout)
        run("git", "config", "user.email", "deploy@tabbench-bio.eu", cwd=checkout)
        run("git", "fetch", remote, "refs/heads/pages", cwd=checkout)
        run("git", "switch", "--create", "pages", "FETCH_HEAD", cwd=checkout)
        previous = run("git", "rev-parse", "HEAD", cwd=checkout)
        for path in checkout.iterdir():
            if path.name == ".git":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        shutil.copytree(
            site,
            checkout,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".gitignore", "__pycache__", "*.pyc"),
        )
        run("git", "add", "--all", cwd=checkout)
        if not run("git", "diff", "--cached", "--name-only", cwd=checkout):
            return previous
        run("git", "commit", "-m", "Deploy TabBench Bio website", cwd=checkout)
        commit = run("git", "rev-parse", "HEAD", cwd=checkout)
        run("git", "push", remote, "HEAD:refs/heads/pages", cwd=checkout)
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    args = parser.parse_args()

    site = args.site_dir.resolve()
    validate_site(site)
    commit = publish(site, args.remote)
    print(f"Published Codeberg Pages commit {commit}")


if __name__ == "__main__":
    main()
