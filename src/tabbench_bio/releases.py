"""Resolve and cache the latest published benchmark reference database."""

import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from tabbench_bio.io_utils import download_file, sha256_file

_REPOSITORY = "not-a-feature/TabBench-Bio"
_LATEST = f"https://api.github.com/repos/{_REPOSITORY}/releases/latest"
LATEST_RESULTS_URL = f"https://github.com/{_REPOSITORY}/releases/latest/download/results.sqlite"


def latest_reference(cache_dir: Path) -> tuple[Path, dict]:
    """Check the latest release and reuse its verified, version-specific asset."""
    request = Request(
        _LATEST, headers={"Accept": "application/vnd.github+json", "User-Agent": "tabbench-bio"}
    )
    with urlopen(request, timeout=60) as response:
        release = json.load(response)
    tag = release["tag_name"]
    assert re.fullmatch(r"[A-Za-z0-9._-]+", tag) and tag not in (".", ".."), tag
    (asset,) = [asset for asset in release["assets"] if asset["name"] == "results.sqlite"]
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
