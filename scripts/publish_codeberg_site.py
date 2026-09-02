"""Publish a built TabBench-Bio site as one orphan Codeberg Pages commit."""

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


def remote_head(remote: str) -> str:
    output = run("git", "ls-remote", remote, "refs/heads/pages")
    return output.split()[0] if output else ""


def publish(site: Path, remote: str) -> str:
    expected = remote_head(remote)
    with tempfile.TemporaryDirectory(prefix="tabbench-bio-pages-") as temporary:
        checkout = Path(temporary)
        shutil.copytree(
            site,
            checkout,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".gitignore", "__pycache__", "*.pyc"),
        )
        run("git", "init", "--initial-branch=pages", cwd=checkout)
        run("git", "config", "user.name", "TabBench Bio deployment", cwd=checkout)
        run("git", "config", "user.email", "deploy@tabbench-bio.eu", cwd=checkout)
        run("git", "add", "--all", cwd=checkout)
        run("git", "commit", "-m", "Deploy TabBench Bio website", cwd=checkout)
        commit = run("git", "rev-parse", "HEAD", cwd=checkout)
        lease = f"--force-with-lease=refs/heads/pages:{expected}"
        run("git", "push", lease, remote, "HEAD:refs/heads/pages", cwd=checkout)
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
