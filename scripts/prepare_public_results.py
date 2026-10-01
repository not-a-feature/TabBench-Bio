"""Create a compact public SQLite copy without machine identifiers or training logs.

Usage: python scripts/prepare_public_results.py INPUT.sqlite OUTPUT.sqlite
The source is read-only. Existing output files are never overwritten.
Prediction, probability and ground-truth payloads remain byte-identical.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path, PurePosixPath

PRIVATE_KEYS = {"host", "hostname", "writer_id", "job_id", "slurm_job_id", "pid"}
PRIVATE_METADATA = {"host", "hostname", "platform", "merge_sources"}
PATH_PATTERN = re.compile(
    r"(?<![\w:/])(?:[A-Za-z]:[\\/]|/(?:home|mnt|ceph|tmp|scratch|var|Users)/)[^\s\"'<>]+"
)


def clean(value):
    if isinstance(value, dict):
        result = {
            key: clean(item)
            for key, item in value.items()
            if key not in PRIVATE_KEYS and key != "cpu_affinity"
        }
        if "cpu_affinity" in value and isinstance(value["cpu_affinity"], list):
            result["cpu_affinity_count"] = len(value["cpu_affinity"])
        return result
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        return PATH_PATTERN.sub(
            lambda match: "[path]/" + PurePosixPath(match[0].replace("\\", "/")).name, value
        )
    return value


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    assert source.is_file(), source
    assert source != output and not output.exists(), "Choose a new output file"
    output.parent.mkdir(parents=True, exist_ok=True)
    original_hash = sha256(source)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".public-results-", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        assert original.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        before = original.execute(
            "SELECT cell,model,status,COUNT(*) FROM attempts GROUP BY cell,model,status"
        ).fetchall()
        with sqlite3.connect(temporary, uri=True) as public:
            print("Copying database...", file=sys.stderr, flush=True)
            original.backup(public)
            public.execute("PRAGMA foreign_keys=ON")
            for key in PRIVATE_METADATA:
                public.execute("DELETE FROM metadata WHERE key=?", (key,))
            public.execute("UPDATE metadata SET value='public-export' WHERE key='writer_id'")
            public.execute("UPDATE metadata SET value='tabbench-bio' WHERE key='experiment_id'")
            public.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                [
                    ("source_database_sha256", original_hash),
                    (
                        "public_export_policy",
                        "Public benchmark snapshot. Machine identifiers and paths redacted; logs and unreferenced blobs omitted. Measurements, hardware specifications and tuning audits retained.",
                    ),
                ],
            )
            for cell, text in public.execute("SELECT cell,config_json FROM cells").fetchall():
                encoded = encode(clean(json.loads(text)))
                public.execute(
                    "UPDATE cells SET config_json=?,config_sha256=? WHERE cell=?",
                    (encoded, hashlib.sha256(encoded.encode()).hexdigest(), cell),
                )
            cursor = original.execute("SELECT attempt_id,record_json FROM attempts")
            while batch := cursor.fetchmany(1000):
                public.executemany(
                    "UPDATE attempts SET record_json=?,log_sha256=NULL WHERE attempt_id=?",
                    [(encode(clean(json.loads(text))), key) for key, text in batch],
                )
            print("Removing unused blobs...", file=sys.stderr, flush=True)
            public.execute("CREATE TEMP TABLE referenced_blobs (sha256 TEXT PRIMARY KEY)")
            for column in ("prediction_sha256", "probability_sha256", "ground_truth_sha256"):
                public.execute(
                    f"INSERT OR IGNORE INTO referenced_blobs SELECT {column} FROM attempts WHERE {column} IS NOT NULL"
                )
            # Foreign-key deletes otherwise rescan every attempt for each unused blob.
            reference_columns = (
                "prediction_sha256",
                "probability_sha256",
                "ground_truth_sha256",
                "log_sha256",
            )
            for column in reference_columns:
                public.execute(f"CREATE INDEX public_export_{column} ON attempts({column})")
            deleted_blobs = public.execute(
                "DELETE FROM blobs WHERE sha256 NOT IN (SELECT sha256 FROM referenced_blobs)"
            ).rowcount
            for column in reference_columns:
                public.execute(f"DROP INDEX public_export_{column}")
            public.execute("DROP TABLE referenced_blobs")
            public.commit()
            print("Compacting public database...", file=sys.stderr, flush=True)
            public.execute("VACUUM")
            print("Verifying unchanged results...", file=sys.stderr, flush=True)
            assert public.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert public.execute("PRAGMA foreign_key_check").fetchall() == []
            assert (
                public.execute(
                    "SELECT cell,model,status,COUNT(*) FROM attempts GROUP BY cell,model,status"
                ).fetchall()
                == before
            )
            assert public.execute("SELECT COUNT(*) FROM blobs WHERE kind='log'").fetchone() == (0,)
            for cell, text, digest in public.execute(
                "SELECT cell,config_json,config_sha256 FROM cells"
            ):
                assert hashlib.sha256(text.encode()).hexdigest() == digest, cell
            public.execute("ATTACH DATABASE ? AS original", (source.as_uri() + "?mode=ro",))
            columns = "attempt_id,cell,seed,dataset,model,status,reason,timestamp,prediction_sha256,probability_sha256,ground_truth_sha256"
            assert (
                public.execute(
                    f"SELECT {columns} FROM attempts EXCEPT SELECT {columns} FROM original.attempts"
                ).fetchone()
                is None
            )
            assert (
                public.execute("SELECT * FROM blobs EXCEPT SELECT * FROM original.blobs").fetchone()
                is None
            )
            original_rows = original.execute(
                "SELECT attempt_id,record_json FROM attempts ORDER BY attempt_id"
            )
            public_rows = public.execute(
                "SELECT attempt_id,record_json FROM attempts ORDER BY attempt_id"
            )
            for (original_id, original_text), (public_id, public_text) in zip(
                original_rows, public_rows, strict=True
            ):
                assert original_id == public_id
                assert clean(json.loads(original_text)) == json.loads(public_text)
                assert clean(json.loads(public_text)) == json.loads(public_text)
            records = sum(row[3] for row in before)
        public.close()
    original.close()
    assert sha256(source) == original_hash, "Source changed during export"
    temporary.rename(output)
    report = {
        "output": str(output),
        "records": records,
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "removed_blobs": deleted_blobs,
        "source_sha256": original_hash,
        "output_sha256": sha256(output),
        "results_and_prediction_payloads_unchanged": True,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
