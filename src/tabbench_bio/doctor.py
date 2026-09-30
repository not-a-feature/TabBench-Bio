"""Read-only checks for a benchmark checkout and its caches."""

import importlib.util
import json
import os
import pickle
import subprocess
from pathlib import Path

from tabbench_bio.bio.cache import load_cached_raw
from tabbench_bio.bio.datasets import load_specs
from tabbench_bio.bio.fingerprint import spec_fingerprint, validate_cached_spec
from tabbench_bio.model_registry import MODEL_REGISTRY, canonical_model_key
from tabbench_bio.model_run import model_python

CHECKOUT = Path(__file__).resolve().parents[2]
SKILL_NAME = "biomedical-tabular-model-selection"


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
    destinations = [(root / "src/tabbench_bio/web", False)]
    if site_dir is not None:
        destinations.append((site_dir, True))
    for destination, required in destinations:
        for asset in canonical.rglob("*"):
            if not asset.is_file():
                continue
            copies = [destination / "skills" / SKILL_NAME / asset.relative_to(canonical)]
            if asset == source:
                copies.append(destination / "skill.md")
            for copy in copies:
                if required and not copy.is_file():
                    problems.append(f"Missing generated skill asset: {copy}")
                elif copy.is_file() and copy.read_bytes() != asset.read_bytes():
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
    environment["PYTHONPATH"] = str(CHECKOUT / "src")
    result = subprocess.run(
        [
            python,
            "-c",
            "import sys; from tabbench_bio.model import _resolve_hyperparameters; "
            "_resolve_hyperparameters([sys.argv[1]], 0) if sys.argv[1] != 'AUTOGLUON' else None",
            key,
        ],
        cwd=CHECKOUT,
        env=environment,
        capture_output=True,
        text=True,
    )
    return (
        [] if result.returncode == 0 else [f"{key}: adapter import failed\n{result.stderr.strip()}"]
    )


def cmd_doctor(args) -> None:
    assert (CHECKOUT / "configs/models").is_dir(), "Run doctor from an editable source checkout"
    checks = [
        ("Registry and environment files", check_metadata(CHECKOUT)),
        ("Skill source and copies", check_skill(CHECKOUT, args.site_dir)),
        ("Dataset caches", check_cache(args.cache_dir)),
    ]
    checks.extend((f"Adapter {key}", check_adapter(key)) for key in args.model)
    for name, problems in checks:
        print(f"{'FAIL' if problems else 'OK'} {name}")
        for problem in problems:
            print(f"  {problem}")
    if not args.model:
        print("Adapter imports not tested; use --model KEY to check its installed environment.")
    raise SystemExit(1 if any(problems for _, problems in checks) else 0)
