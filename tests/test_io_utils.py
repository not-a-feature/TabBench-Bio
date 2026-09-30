import hashlib
import json

import pandas as pd
import pytest

from tabbench_bio import io_utils


@pytest.mark.parametrize("kind", ["json", "pickle", "csv", "parquet"])
@pytest.mark.parametrize("failure", [None, "fsync", "replace"])
def test_atomic_writers_preserve_data_and_remove_temporary_files(
    tmp_path, monkeypatch, kind, failure
):
    if kind == "parquet":
        pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"target": [1, 2]})
    writers = {
        "json": lambda path: io_utils.atomic_write_json(path, frame.to_dict("list")),
        "pickle": lambda path: io_utils.atomic_to_pickle(frame, path),
        "csv": lambda path: io_utils.atomic_to_csv(frame, path, index=False),
        "parquet": lambda path: io_utils.atomic_to_parquet(frame, path, index=False),
    }
    path = tmp_path / f"artifact.{kind}"
    path.write_bytes(b"previous artifact")

    def interrupted(*args):
        raise OSError("interrupted publication")

    if failure:
        monkeypatch.setattr(io_utils.os, failure, interrupted)
        with pytest.raises(OSError, match="interrupted publication"):
            writers[kind](path)
        assert path.read_bytes() == b"previous artifact"
    else:
        writers[kind](path)
        if kind == "json":
            assert json.loads(path.read_text(encoding="utf-8")) == frame.to_dict("list")
        else:
            reader = {"pickle": pd.read_pickle, "csv": pd.read_csv, "parquet": pd.read_parquet}[
                kind
            ]
            pd.testing.assert_frame_equal(reader(path), frame)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "payload", [b"", b"abc", b"0123456789" * 100000], ids=["empty", "small", "large"]
)
def test_file_checksum(tmp_path, payload):
    path = tmp_path / "artifact"
    path.write_bytes(payload)
    assert io_utils.sha256_file(path) == hashlib.sha256(payload).hexdigest()
