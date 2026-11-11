"""Python execution defaults composed with the public JSON metadata catalog."""

from dataclasses import asdict, dataclass
from pathlib import Path

from tabbench_bio.model_metadata import PUBLIC_METADATA_FIELDS, load_model_metadata


@dataclass(frozen=True)
class ExecutionSpec:
    device: str
    environment: str = "standard"
    adapter: str | None = None
    aliases: tuple[str, ...] = ()
    hyperparameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    """Resolved compatibility view; public values are required JSON fields."""

    display: str
    category: str
    description: str
    source_url: str | None
    hidden: bool
    training_data_overlap: bool
    supported_tasks: tuple[str, ...]
    max_features: int | None
    max_classes: int | None
    device: str
    environment: str = "standard"
    adapter: str | None = None
    aliases: tuple[str, ...] = ()
    hyperparameters: tuple[tuple[str, object], ...] = ()


EXECUTION_REGISTRY = {
    "DUMMY": ExecutionSpec(device="cpu"),
    "KNN": ExecutionSpec(device="cpu"),
    "LR": ExecutionSpec(device="cpu"),
    "RF": ExecutionSpec(device="cpu"),
    "XT": ExecutionSpec(device="cpu"),
    "CAT": ExecutionSpec(device="cpu"),
    "GBM": ExecutionSpec(device="cpu"),
    "XGB": ExecutionSpec(device="gpu"),
    "NN_TORCH": ExecutionSpec(device="gpu"),
    "REALMLP": ExecutionSpec(device="gpu"),
    "TABM": ExecutionSpec(device="gpu"),
    "MITRA": ExecutionSpec(device="gpu"),
    "FTFM": ExecutionSpec(
        device="gpu", environment="ftfm", adapter="tabbench_bio.models.ftfm:FTFMModel"
    ),
    "CAUSILO": ExecutionSpec(
        device="gpu", environment="causilo", adapter="tabbench_bio.models.causilo:CausiloModel"
    ),
    "GENEICL": ExecutionSpec(
        device="gpu", environment="geneicl", adapter="tabbench_bio.models.geneicl:GeneICLModel"
    ),
    "LIGHTPFN": ExecutionSpec(
        device="gpu", environment="lightpfn", adapter="tabbench_bio.models.lightpfn:LightPFNModel"
    ),
    "KUMO-TABULAR-MEDIUM": ExecutionSpec(
        device="gpu",
        environment="kumo",
        adapter="tabbench_bio.models.kumo_tabular:KumoTabularMediumModel",
    ),
    "KUMO-TABULAR-LARGE": ExecutionSpec(
        device="gpu",
        environment="kumo",
        adapter="tabbench_bio.models.kumo_tabular:KumoTabularLargeModel",
    ),
    "SELDON": ExecutionSpec(
        device="gpu", environment="seldon", adapter="tabbench_bio.models.seldon:SeldonModel"
    ),
    "SELDON-MINI": ExecutionSpec(
        device="gpu", environment="seldon", adapter="tabbench_bio.models.seldon:SeldonMiniModel"
    ),
    "LIMIX-2": ExecutionSpec(
        device="gpu", environment="limix2", adapter="tabbench_bio.models.limix_2:LimiX2Model"
    ),
    "REALTABPFN-V2": ExecutionSpec(device="gpu"),
    "REALTABPFN-V2.5": ExecutionSpec(device="gpu"),
    "TABPFN-V3": ExecutionSpec(
        device="gpu", adapter="tabbench_bio.models.tabpfn_v3:TabPFNV3Model", aliases=("TABPFNV3",)
    ),
    "TABPFN-V3.5": ExecutionSpec(
        device="gpu",
        environment="tabpfn35",
        adapter="tabbench_bio.models.tabpfn_v3_5:TabPFNV35Model",
        aliases=("TABPFNV35",),
    ),
    "TABPFN-WIDE": ExecutionSpec(
        device="gpu",
        adapter="tabbench_bio.models.tabpfn_wide:TabPFNWideModel",
        aliases=("TABPFNWIDE",),
    ),
    "TABPFN-WIDE-5K-NE3": ExecutionSpec(
        device="gpu",
        adapter="tabbench_bio.models.tabpfn_wide:TabPFNWideModel",
        hyperparameters=(("model_name", "wide-v2-5k"), ("n_estimators", 3)),
    ),
    "TABFM": ExecutionSpec(device="gpu", adapter="tabbench_bio.models.tabfm:TabFMModel"),
    "TABDPT": ExecutionSpec(device="gpu"),
    "TABICL": ExecutionSpec(device="gpu"),
    "AUTOGLUON": ExecutionSpec(device="gpu"),
    "FASTAI": ExecutionSpec(device="gpu"),
    "TABPFN": ExecutionSpec(device="gpu"),
}


def load_registry(metadata_path: Path | None = None) -> dict[str, ModelSpec]:
    """Join validated JSON metadata with Python adapter/execution definitions."""
    metadata = load_model_metadata(metadata_path)
    missing = EXECUTION_REGISTRY.keys() - metadata.keys()
    if missing:
        raise ValueError(f"Missing public metadata for execution models: {sorted(missing)}")
    registry = {}
    for key, execution in EXECUTION_REGISTRY.items():
        public = {field: metadata[key][field] for field in PUBLIC_METADATA_FIELDS}
        public["supported_tasks"] = tuple(public["supported_tasks"])
        registry[key] = ModelSpec(**public, **asdict(execution))
    return registry


MODEL_REGISTRY = load_registry()
MODEL_ALIASES = {alias: key for key, spec in EXECUTION_REGISTRY.items() for alias in spec.aliases}
MODEL_CATEGORY = {key: spec.category for key, spec in MODEL_REGISTRY.items()}
MODEL_DISPLAY = {key: spec.display for key, spec in MODEL_REGISTRY.items()}
REGULAR_MAX_FEATURES = {
    key: spec.max_features for key, spec in MODEL_REGISTRY.items() if spec.max_features is not None
}


def canonical_model_key(key: str) -> str:
    key = key.upper()
    return MODEL_ALIASES[key] if key in MODEL_ALIASES else key


def model_entry(entry: dict) -> dict:
    key = canonical_model_key(entry["key"])
    parent = canonical_model_key(entry["base_model"]) if "base_model" in entry else key
    defaults = {}
    if parent in EXECUTION_REGISTRY:
        spec = EXECUTION_REGISTRY[parent]
        defaults = {"environment": spec.environment, "device": spec.device}
    return {**defaults, **entry, "key": key}
