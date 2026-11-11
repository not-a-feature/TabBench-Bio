"""Validated public model metadata from the bundled JSON catalog."""

import json
import re
from pathlib import Path

from tabbench_bio.resources import resource_root

PUBLIC_METADATA_FIELDS = frozenset(
    {
        "display",
        "category",
        "description",
        "source_url",
        "hidden",
        "training_data_overlap",
        "supported_tasks",
        "max_features",
        "max_classes",
    }
)


def load_model_metadata(path: Path | None = None) -> dict[str, dict]:
    """Load complete metadata, resolving explicitly declared parent inheritance.

    Base entries declare every public field. Variants may inherit those fields
    through ``base_model`` and override individual values in the same catalog.
    Execution settings are deliberately rejected here.
    """
    path = Path(path) if path is not None else resource_root() / "configs/model_metadata.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: model metadata must be a JSON array")
    entries = {}
    for entry in rows:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each model metadata entry must be an object")
        key = entry.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]*", key):
            raise ValueError(f"{path}: invalid model key {key!r}")
        if key in entries:
            raise ValueError(f"{path}: duplicate model key {key}")
        unknown = entry.keys() - PUBLIC_METADATA_FIELDS - {"key", "base_model"}
        if unknown:
            raise ValueError(f"{path}: {key}: unknown public metadata fields {sorted(unknown)}")
        entries[key] = entry

    resolved = {}
    visiting = set()

    def resolve(key):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError(f"{path}: cyclic model metadata inheritance at {key}")
        if key not in entries:
            raise ValueError(f"{path}: unknown metadata parent {key!r}")
        visiting.add(key)
        entry = entries[key]
        parent = entry.get("base_model")
        inherited = {}
        if "base_model" in entry:
            if not isinstance(parent, str):
                raise ValueError(f"{path}: {key}.base_model must be a model key")
            inherited = {k: v for k, v in resolve(parent).items() if k in PUBLIC_METADATA_FIELDS}
        item = {**inherited, **entry}
        missing = PUBLIC_METADATA_FIELDS - item.keys()
        if missing:
            raise ValueError(f"{path}: {key}: missing public metadata fields {sorted(missing)}")
        for field in ("display", "category", "description"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"{path}: {key}.{field} must be a non-empty string")
        if item["source_url"] is not None and (
            not isinstance(item["source_url"], str) or not item["source_url"].strip()
        ):
            raise ValueError(f"{path}: {key}.source_url must be a URL string or null")
        for field in ("hidden", "training_data_overlap"):
            if type(item[field]) is not bool:
                raise ValueError(f"{path}: {key}.{field} must be a boolean")
        tasks = item["supported_tasks"]
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(task not in ("classification", "regression") for task in tasks)
            or len(set(tasks)) != len(tasks)
        ):
            raise ValueError(f"{path}: {key}.supported_tasks must list unique supported tasks")
        for field, minimum in (("max_features", 1), ("max_classes", 2)):
            if item[field] is not None and (type(item[field]) is not int or item[field] < minimum):
                raise ValueError(f"{path}: {key}.{field} must be null or an integer >= {minimum}")
        visiting.remove(key)
        resolved[key] = item
        return item

    for key in entries:
        resolve(key)
    return resolved


def hidden_model_ids(registry: dict[str, dict] | None = None) -> list[str]:
    """Return default-hidden IDs, including inherited variant visibility."""
    registry = load_model_metadata() if registry is None else registry
    return [
        key
        for key, entry in registry.items()
        if entry.get("hidden", registry.get(entry.get("base_model"), {}).get("hidden", False))
    ]


def load_model_catalog(root: Path | None = None) -> dict[str, dict]:
    """Join public metadata with runnable rosters without duplicating public fields."""
    root = resource_root() if root is None else Path(root)
    registry = load_model_metadata(root / "configs/model_metadata.json")
    paths = sorted((root / "configs/models").glob("*.json"))
    if not paths:
        raise ValueError(f"Model rosters missing: {root / 'configs/models'}")
    seen = {}
    for path in paths:
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{path}: model roster must be a JSON array")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                raise ValueError(f"{path}: roster entries require a model key")
            key = entry["key"]
            if key not in registry:
                raise ValueError(f"{path}: missing public metadata for {key}")
            misplaced = entry.keys() & PUBLIC_METADATA_FIELDS
            if misplaced:
                raise ValueError(
                    f"{path}: move {key} metadata {sorted(misplaced)} to configs/model_metadata.json"
                )
            if key in seen and seen[key] != entry:
                raise ValueError(f"{path}: conflicting roster model {key}")
            if entry.get("base_model") != registry[key].get("base_model"):
                raise ValueError(f"{path}: conflicting metadata parent for {key}")
            seen[key] = entry
            registry[key] = {**registry[key], **entry}
    return registry
