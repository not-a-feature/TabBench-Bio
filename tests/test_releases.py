"""The default reference follows the latest release and verifies cached assets."""

import hashlib
import io
import json
from urllib.request import Request

import pytest

from tabbench_bio import io_utils, releases


def release_record(tag, data, asset_id):
    return {
        "tag_name": tag,
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
    first, metadata = releases.latest_reference(tmp_path)
    assert first.read_bytes() == data and metadata["tag"] == "v0.3.0"
    assert releases.latest_reference(tmp_path)[0] == first
    assert requests.count(releases._LATEST) == 2 and len(requests) == 3
    data = b"new reference"
    release = release_record("v0.4.0", data, 40)
    newest, metadata = releases.latest_reference(tmp_path)
    assert newest != first and newest.read_bytes() == data
    assert metadata["tag"] == "v0.4.0" and first.is_file()
    newest.write_bytes(b"corrupted")
    with pytest.raises(AssertionError, match="Corrupt reference cache"):
        releases.latest_reference(tmp_path)
