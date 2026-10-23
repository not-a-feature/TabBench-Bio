"""Read-only checks for an installed benchmark and its caches."""

import importlib.util
import json
import os
import pickle
import sqlite3
import subprocess
from collections import Counter
from contextlib import closing
from itertools import product
from pathlib import Path

from tabbench_bio.bio.cache import load_cached_raw
from tabbench_bio.bio.datasets import load_specs
from tabbench_bio.bio.fingerprint import spec_fingerprint, validate_cached_spec
from tabbench_bio.model_registry import MODEL_REGISTRY, canonical_model_key
from tabbench_bio.model_run import model_python
from tabbench_bio.resources import resource_root
from tabbench_bio.seeds import get_seeds
from tabbench_bio.web_metadata import SKILL_NAME

CHECKOUT = resource_root()


def check_metadata(root: Path) -> list[str]:
    problems = []
    for key, spec in MODEL_REGISTRY.items():
        profile = root / "environments" / f"{spec.environment}.txt"
        if not profile.is_file():
            problems.append(f"{key}: missing environment profile {profile}")
        if spec.adapter is not None:
            module, _ = spec.adapter.split(":")
            if importlib.util.find_spec(module) is None:
                problems.append(f"{key}: missing adapter module {module}")
    return problems


def check_skill(root: Path, site_dir: Path | None = None) -> list[str]:
    canonical = root / "skills" / SKILL_NAME
    source = canonical / "SKILL.md"
    if not source.is_file():
        return [f"Missing canonical skill: {source}"]
    problems = []
    if "license: EUPL-1.2" not in source.read_text(encoding="utf-8").splitlines():
        problems.append(f"{source}: expected license: EUPL-1.2")
    web = root.parent / "web" if root.name == "_resources" else root / "src/tabbench_bio/web"
    destinations = [(web, False)]
    if site_dir is not None:
        destinations.append((site_dir, True))
    for destination, required in destinations:
        copy = destination / "skill.md"
        if required and not copy.is_file():
            problems.append(f"Missing generated skill asset: {copy}")
        elif copy.is_file() and copy.read_bytes() != source.read_bytes():
            problems.append(f"Stale skill copy: {copy}; rebuild the site")
    return problems


def check_cache(cache_dir: Path) -> list[str]:
    specs = load_specs()
    problems = []
    raw_root = cache_dir / "bio"
    for path in sorted((raw_root / "datasets").glob("*.pkl")):
        try:
            raw = load_cached_raw(raw_root, path.stem)
            assert raw.bio_id in specs, f"{path}: dataset no longer registered"
            validate_cached_spec(raw, specs[raw.bio_id])
        except (AssertionError, OSError, EOFError, pickle.UnpicklingError) as exc:
            problems.append(str(exc))
    for directory in sorted((cache_dir / "datasets_processed").glob("seed_*")):
        if not directory.is_dir():
            continue
        metadata = directory / "split_params.json"
        if not metadata.is_file() and "_frozen_" in directory.name:
            metadata = (
                directory.with_name(directory.name.split("_frozen_", 1)[0]) / "split_params.json"
            )
        if not metadata.is_file():
            problems.append(f"{directory}: missing split fingerprints; rebuild this cache")
            continue
        params = json.loads(metadata.read_text(encoding="utf-8"))
        if "dataset_specs" in params:
            for name, fingerprint in params["dataset_specs"].items():
                if name not in specs or spec_fingerprint(specs[name]) != fingerprint:
                    problems.append(f"{directory}: stale split specification for {name}")
        for path in directory.glob("*/dataset_spec.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            name = record["bio_id"]
            if name not in specs or spec_fingerprint(specs[name]) != record["spec_sha256"]:
                problems.append(f"{path.parent}: stale split specification for {name}")
        for frame in directory.rglob("*_train.pkl"):
            if "dataset_specs" not in params and not (frame.parent / "dataset_spec.json").is_file():
                problems.append(f"{frame}: missing dataset fingerprint; rebuild this cache")
    return problems


def check_adapter(key: str) -> list[str]:
    key = canonical_model_key(key)
    if key not in MODEL_REGISTRY:
        return [f"Unknown model: {key}"]
    try:
        python = model_python(MODEL_REGISTRY[key].environment)
    except AssertionError as exc:
        return [str(exc)]
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    result = subprocess.run(
        [
            python,
            "-c",
            "import sys; from tabbench_bio.model import _resolve_hyperparameters; "
            "_resolve_hyperparameters([sys.argv[1]], 0) if sys.argv[1] != 'AUTOGLUON' else None",
            key,
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    return (
        [] if result.returncode == 0 else [f"{key}: adapter import failed\n{result.stderr.strip()}"]
    )


def check_results(path: Path) -> list[str]:
    """Check outcome presence against the datasets, folds and models stored per cell."""
    problems = []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        dataset_keys = {
            row[0] for row in connection.execute("SELECT DISTINCT dataset FROM attempts")
        }
        for cell, config_json in connection.execute("SELECT cell, config_json FROM cells"):
            rows = connection.execute(
                "SELECT seed, dataset, model, status, reason FROM attempts WHERE cell = ? "
                "ORDER BY CASE WHEN status='skip' AND reason IN ('benchmark_exclusion','dummy_unavailable') "
                "THEN 2 WHEN status='pass' THEN 1 ELSE 0 END DESC, timestamp DESC, attempt_id DESC",
                (cell,),
            ).fetchall()
            counts = Counter(row[:3] for row in rows)
            present = {row[:3] for row in rows if row[3] in {"pass", "fail", "skip"}}
            current = {}
            for row in rows:
                current.setdefault(row[:3], row[3:])
            config = json.loads(config_json)
            models = {"DUMMY", *config["models"]} | {model for _, _, model in counts}
            datasets = {dataset for _, dataset, _ in counts}
            for name in config["datasets_classification"] + config["datasets_regression"]:
                name = name.replace("/", "_").replace("\\", "_")
                # Biological tasks have one curated target; retain additional recorded targets.
                datasets.add(f"{name}_0")
                datasets.update(key for key in dataset_keys if key.rsplit("_", 1)[0] == name)
            seeds = set(get_seeds(config)) | {seed for seed, _, _ in counts}
            missing = sorted(
                (seed, dataset, model)
                for seed, dataset, model in product(seeds, datasets, models)
                if (seed, dataset, model) not in present
            )
            duplicates = sorted((key, n) for key, n in counts.items() if n > 1)
            gate_violations = sorted(
                (seed, dataset, model, status)
                for (seed, dataset, model), (status, _) in current.items()
                if model != "DUMMY"
                and (seed, dataset, "DUMMY") in current
                and current[(seed, dataset, "DUMMY")][0] != "pass"
                and status != "skip"
            )
            if gate_violations:
                problems.append(
                    f"{cell}: {len(gate_violations)} DUMMY eligibility violations; "
                    f"other models must skip when DUMMY fails or skips; first 10: {gate_violations[:10]}"
                )
            if missing:
                problems.append(
                    f"{cell}: {len(missing)} missing split coordinates "
                    f"(seed, dataset, model); first 10: {missing[:10]}"
                )
            if duplicates:
                problems.append(
                    f"{cell}: {len(duplicates)} duplicate split coordinates "
                    f"({sum(n - 1 for _, n in duplicates)} extra attempts); "
                    f"first 10 (coordinate, count): {duplicates[:10]}"
                )
    return problems


def cmd_doctor(args) -> None:
    checks = [
        ("Registry and environment files", check_metadata(CHECKOUT)),
        ("Skill source and copies", check_skill(CHECKOUT, args.site_dir)),
        ("Dataset caches", check_cache(args.cache_dir)),
    ]
    if args.results is not None:
        checks.append(("Result split coverage", check_results(args.results)))
    checks.extend((f"Adapter {key}", check_adapter(key)) for key in args.model)
    for name, problems in checks:
        print(f"{'FAIL' if problems else 'OK'} {name}")
        for problem in problems:
            print(f"  {problem}")
    if not args.model:
        print("Adapter imports not tested; use --model KEY to check its installed environment.")
    raise SystemExit(1 if any(problems for _, problems in checks) else 0)
