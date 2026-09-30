"""Release downloads must preserve offline reuse and reject incomplete or changed data."""

import hashlib
import io
from dataclasses import replace
from urllib.error import URLError

import pytest

from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.loaders import get_loader
from tabbench_bio.bio.loaders import local as loader

TABLE = b"embedding_0,group_id,y\n1,locus-a,A\n2,locus-b,B\n"


@pytest.fixture
def spec(monkeypatch, tmp_path):
    monkeypatch.delenv("TABBENCH_BIO_LOCAL_DIR", raising=False)
    monkeypatch.setattr(loader, "_BUNDLED_LOCAL", tmp_path / "bundled")
    return BioDatasetSpec(
        bio_id="local-release",
        source="local",
        fetch_id="table.csv",
        target="y",
        problem_type="binary",
        group_column="group_id",
        download_url="https://github.com/example/benchmark/releases/download/datasets-v1/table.csv",
        download_sha256=hashlib.sha256(TABLE).hexdigest(),
    )


def test_download_then_reuse_without_network(spec, monkeypatch, tmp_path):
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(TABLE)

    monkeypatch.setattr(loader, "urlopen", download)
    source = get_loader(spec, cache_dir=str(tmp_path / "cache"))
    first = source.fetch(spec)
    second = source.fetch(spec)
    assert calls == [spec.download_url]
    assert (tmp_path / "cache/local_raw/table.csv").read_bytes() == TABLE
    assert list(first.X) == ["embedding_0"]
    assert first.groups.tolist() == ["locus-a", "locus-b"]
    assert first.y.equals(second.y)
    assert first.source_url == spec.download_url


def test_checksum_failure_never_publishes_partial_file(spec, monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad content"))
    source = loader.LocalLoader(cache_dir=tmp_path / "cache")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        source.fetch(spec)
    assert list((tmp_path / "cache").iterdir()) == []


def test_interrupted_download_cleans_partial_file(spec, monkeypatch, tmp_path):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(10)

    monkeypatch.setattr(loader, "urlopen", lambda *args, **kwargs: Interrupted(TABLE))
    with pytest.raises(OSError, match="connection interrupted"):
        loader.LocalLoader(cache_dir=tmp_path / "cache").fetch(spec)
    assert list((tmp_path / "cache").iterdir()) == []


def test_unavailable_release_cleans_partial_file(spec, monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise URLError("release unavailable")

    monkeypatch.setattr(loader, "urlopen", unavailable)
    with pytest.raises(URLError, match="release unavailable"):
        loader.LocalLoader(cache_dir=tmp_path / "cache").fetch(spec)
    assert list((tmp_path / "cache").iterdir()) == []


def test_corrupt_cached_file_is_not_silently_reused(spec, tmp_path):
    path = tmp_path / "table.csv"
    path.write_bytes(b"corrupt")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        loader.LocalLoader(cache_dir=tmp_path).fetch(spec)
    assert path.read_bytes() == b"corrupt"


def test_explicit_local_directory_receives_download(spec, monkeypatch, tmp_path):
    local = tmp_path / "custom"
    monkeypatch.setenv("TABBENCH_BIO_LOCAL_DIR", str(local))
    monkeypatch.setattr(loader, "urlopen", lambda *args, **kwargs: io.BytesIO(TABLE))
    loader.LocalLoader(cache_dir=tmp_path / "unused").fetch(spec)
    assert (local / "table.csv").read_bytes() == TABLE
    assert not (tmp_path / "unused").exists()


def test_existing_local_table_takes_precedence(spec, monkeypatch, tmp_path):
    path = tmp_path / "bundled/table.csv"
    path.parent.mkdir()
    path.write_bytes(TABLE)
    monkeypatch.setattr(loader, "urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    loader.LocalLoader(cache_dir=tmp_path / "unused").fetch(spec)
    assert not (tmp_path / "unused").exists()


def test_missing_unregistered_file_stays_local(spec, tmp_path):
    spec = replace(spec, download_url=None, download_sha256=None)
    with pytest.raises(FileNotFoundError, match="local file"):
        loader.LocalLoader(cache_dir=tmp_path).fetch(spec)


@pytest.mark.parametrize(
    "changes",
    [
        {"download_sha256": None},
        {"download_url": None},
        {"download_sha256": "bad"},
        {"download_url": "http://example.com/table.csv"},
    ],
)
def test_download_requires_https_and_checksum(spec, changes):
    with pytest.raises(AssertionError):
        replace(spec, **changes)
