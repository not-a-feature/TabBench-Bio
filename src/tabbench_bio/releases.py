"""Resolve and cache a published benchmark reference database."""

import json
import re
from itertools import count
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from tabbench_bio import __version__
from tabbench_bio.io_utils import download_file, sha256_file

_REPOSITORY = "not-a-feature/TabBench-Bio"
_RELEASES = f"https://api.github.com/repos/{_REPOSITORY}/releases"
_LATEST = f"{_RELEASES}/latest"
LATEST_RESULTS_URL = f"https://github.com/{_REPOSITORY}/releases/latest/download/results.sqlite"
EXTERNAL_RESULTS_URL = (
    f"https://github.com/{_REPOSITORY}/releases/download/v{__version__}/external-results.sqlite"
)


def parse_release_version(value: str) -> tuple[int, ...] | None:
    """Parse an X.Y.Z release version for numeric comparison."""
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        return tuple(int(part) for part in value.split("."))
    return None


def _github_json(url):
    request = Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "tabbench-bio"}
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _results_assets(release):
    return [asset for asset in release["assets"] if asset["name"] == "results.sqlite"]


def _previous_release(version: str):
    installed = parse_release_version(version)
    candidates = []
    if installed is not None:
        for page in count(1):
            releases = _github_json(f"{_RELEASES}?per_page=100&page={page}")
            for release in releases:
                if release["draft"] or release["prerelease"]:
                    continue
                parts = parse_release_version(release["tag_name"].removeprefix("v"))
                if parts is not None and parts < installed and _results_assets(release):
                    candidates.append((parts, release))
            if len(releases) < 100:
                break
    if not candidates:
        raise ValueError(f"No results.sqlite on release v{version} or any earlier release")
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _versioned_release(selection: str | None):
    version = __version__ if selection is None else selection
    tag = f"v{version}"
    try:
        release = _github_json(f"{_RELEASES}/tags/{quote(tag, safe='')}")
    except HTTPError as error:
        if error.code != 404:
            raise
        if selection is not None:
            raise ValueError(f"GitHub release {tag} was not found") from error
    else:
        assert release["tag_name"] == tag, (tag, release["tag_name"])
        if selection is not None or _results_assets(release):
            return release
    release = _previous_release(version)
    print(f"No results.sqlite for {tag}. Falling back to {release['tag_name']}.", flush=True)
    return release


def release_reference(cache_dir: Path, selection: str | None = None) -> tuple[Path, dict]:
    """Choose a release and reuse its verified, version-specific asset."""
    release = _github_json(_LATEST) if selection == "latest" else _versioned_release(selection)
    tag = release["tag_name"]
    assert re.fullmatch(r"[A-Za-z0-9._+-]+", tag) and tag not in (".", ".."), tag
    assets = _results_assets(release)
    assert len(assets) == 1, (
        f"Release {tag} must have one results.sqlite asset (found {len(assets)})"
    )
    (asset,) = assets
    digest = asset["digest"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), "Release asset needs a SHA-256 digest"
    expected = digest.removeprefix("sha256:")
    url = asset["browser_download_url"]
    assert url.startswith(f"https://github.com/{_REPOSITORY}/releases/download/"), url
    destination = cache_dir / "releases" / tag / str(int(asset["id"])) / "results.sqlite"
    print(f"Published baseline: {tag} ({release['published_at']})", flush=True)
    if destination.is_file():
        assert (
            destination.stat().st_size == asset["size"] and sha256_file(destination) == expected
        ), f"Corrupt reference cache: {destination}; remove it and retry"
        print(f"Reference download cache reused: {destination}", flush=True)
    else:
        print(f"Downloading reference: {url} ({asset['size']:,} bytes)", flush=True)
        download_file(url, destination, checksum=expected)
        assert destination.stat().st_size == asset["size"], "Reference download size mismatch"
    return destination, {
        "tag": tag,
        "published_at": release["published_at"],
        "release_url": release["html_url"],
        "asset_url": url,
        "sha256": expected,
        "path": str(destination),
    }
