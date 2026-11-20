"""Versioned reference selection and verified release caching."""

import hashlib
import io
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from tabbench_bio import io_utils, releases

RELEASES_URL = "https://api.github.com/repos/not-a-feature/TabBench-Bio/releases"


def release_record(tag, data, asset_id):
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "published_at": "2026-10-15T08:23:53Z",
        "html_url": f"https://github.com/not-a-feature/TabBench-Bio/releases/tag/{tag}",
        "assets": [
            {
                "id": asset_id,
                "name": "results.sqlite",
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "browser_download_url": f"https://github.com/not-a-feature/TabBench-Bio/releases/download/{tag}/results.sqlite",
            }
        ],
    }


def stub_github(monkeypatch, responses):
    requests = []

    def open_url(request, timeout):
        url = request.full_url if isinstance(request, Request) else request
        requests.append(url)
        response = responses[url]
        if isinstance(response, HTTPError):
            raise response
        return io.BytesIO(
            response if isinstance(response, bytes) else json.dumps(response).encode()
        )

    monkeypatch.setattr(releases, "urlopen", open_url)
    monkeypatch.setattr(io_utils, "urlopen", open_url)
    return requests


def test_latest_is_checked_again_and_unchanged_asset_is_not_downloaded(tmp_path, monkeypatch):
    data = b"first reference"
    release = release_record("v0.3.0", data, 30)
    requests = []

    def open_url(request, timeout):
        url = request.full_url if isinstance(request, Request) else request
        requests.append(url)
        return io.BytesIO(json.dumps(release).encode() if url == releases._LATEST else data)

    monkeypatch.setattr(releases, "urlopen", open_url)
    monkeypatch.setattr(io_utils, "urlopen", open_url)
    first, metadata = releases.release_reference(tmp_path, "latest")
    assert first.read_bytes() == data and metadata["tag"] == "v0.3.0"
    assert releases.release_reference(tmp_path, "latest")[0] == first
    assert requests.count(releases._LATEST) == 2 and len(requests) == 3
    data = b"new reference"
    release = release_record("v0.4.0", data, 40)
    newest, metadata = releases.release_reference(tmp_path, "latest")
    assert newest != first and newest.read_bytes() == data
    assert metadata["tag"] == "v0.4.0" and first.is_file()
    previous = newest
    data = b"replacement asset in the latest release"
    release = release_record("v0.4.0", data, 41)
    newest, metadata = releases.release_reference(tmp_path, "latest")
    assert newest != previous and newest.read_bytes() == data
    assert previous.is_file() and metadata["sha256"] == hashlib.sha256(data).hexdigest()
    newest.write_bytes(b"corrupted")
    with pytest.raises(AssertionError, match="Corrupt reference cache"):
        releases.release_reference(tmp_path, "latest")


@pytest.mark.parametrize(("selection", "tag"), [(None, "v1.4.0"), ("1.3.2", "v1.3.2")])
def test_default_and_explicit_versions_request_the_exact_release(
    tmp_path, monkeypatch, selection, tag
):
    monkeypatch.setattr(releases, "__version__", "1.4.0")
    data = b"versioned reference"
    release = release_record(tag, data, 100)
    api = f"{RELEASES_URL}/tags/{tag}"
    download = release["assets"][0]["browser_download_url"]
    requests = stub_github(monkeypatch, {api: release, download: data})

    path, metadata = releases.release_reference(tmp_path, selection)

    assert requests == [api, download]
    assert path.read_bytes() == data
    assert metadata["tag"] == tag


@pytest.mark.parametrize("missing_release", [False, True])
def test_default_walks_back_to_nearest_older_release_with_database(
    tmp_path, monkeypatch, capsys, missing_release
):
    monkeypatch.setattr(releases, "__version__", "1.14.0")
    data = b"nearest older reference"
    release = release_record("v1.10.0", data, 110)
    api = f"{RELEASES_URL}/tags/v1.14.0"
    page1 = f"{RELEASES_URL}?per_page=100&page=1"
    page2 = f"{RELEASES_URL}?per_page=100&page=2"
    download = release["assets"][0]["browser_download_url"]
    requests = stub_github(
        monkeypatch,
        {
            api: HTTPError(api, 404, "Not Found", {}, None)
            if missing_release
            else {"tag_name": "v1.14.0", "assets": []},
            page1: [
                release_record("v2.0.0", b"newer", 200),
                {**release_record("v1.13.0", b"prerelease", 113), "prerelease": True},
                {**release_record("v1.12.0", b"draft", 112), "draft": True},
                {**release_record("v1.11.0", b"", 111), "assets": []},
                release_record("v1.9.0", b"older", 109),
                *(release_record(f"v0.1.{i}", b"older", 300 + i) for i in range(95)),
            ],
            page2: [release],
            download: data,
        },
    )

    path, metadata = releases.release_reference(tmp_path)

    assert requests == [api, page1, page2, download]
    assert path.read_bytes() == data
    assert metadata["tag"] == "v1.10.0"
    assert "Falling back to v1.10.0" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("selection", "endpoint"), [("latest", "latest"), ("1.4.0", "tags/v1.4.0")]
)
def test_explicit_selection_does_not_fall_back_when_database_is_missing(
    tmp_path, monkeypatch, selection, endpoint
):
    api = f"{RELEASES_URL}/{endpoint}"
    requests = stub_github(monkeypatch, {api: {"tag_name": "v1.4.0", "assets": []}})

    with pytest.raises(AssertionError, match=r"Release v1\.4\.0 .* results\.sqlite"):
        releases.release_reference(tmp_path, selection)

    assert requests == [api]


def test_default_rejects_bad_release_metadata_without_falling_back(tmp_path, monkeypatch):
    monkeypatch.setattr(releases, "__version__", "1.4.0")
    api = f"{RELEASES_URL}/tags/v1.4.0"
    release = release_record("v1.4.0", b"invalid reference", 140)
    release["assets"][0]["digest"] = "missing"
    requests = stub_github(monkeypatch, {api: release})

    with pytest.raises(AssertionError, match="Release asset needs a SHA-256 digest"):
        releases.release_reference(tmp_path)

    assert requests == [api]
