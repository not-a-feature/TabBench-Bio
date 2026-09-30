"""Model adapters, execution defaults and public metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    display: str
    category: str
    device: str
    environment: str = "standard"
    adapter: str | None = None
    max_features: int | None = None
    supported_tasks: tuple[str, ...] = ("classification", "regression")
    max_classes: int | None = None
    training_data_overlap: bool = False
    aliases: tuple[str, ...] = ()
    hyperparameters: tuple[tuple[str, object], ...] = ()


MODEL_REGISTRY = {
    "DUMMY": ModelSpec(display="Constant", category="Baseline", device="cpu"),
    "KNN": ModelSpec(display="KNN", category="Traditional ML", device="cpu"),
    "LR": ModelSpec(display="Logistic Regression", category="Traditional ML", device="cpu"),
    "RF": ModelSpec(display="Random Forest", category="Tree-based", device="cpu"),
    "XT": ModelSpec(display="Extra Trees", category="Tree-based", device="cpu"),
    "CAT": ModelSpec(display="CatBoost", category="Gradient Boosting", device="cpu"),
    "GBM": ModelSpec(display="LightGBM", category="Gradient Boosting", device="cpu"),
    "XGB": ModelSpec(display="XGBoost", category="Gradient Boosting", device="gpu"),
    "NN_TORCH": ModelSpec(display="MLP", category="Deep Learning", device="gpu"),
    "REALMLP": ModelSpec(display="RealMLP", category="Deep Learning", device="gpu"),
    "TABM": ModelSpec(display="TabM", category="Deep Learning", device="gpu"),
    "MITRA": ModelSpec(
        display="MITRA", category="Tabular Foundation", device="gpu", max_features=500
    ),
    "CAUSILO": ModelSpec(
        display="Causilo",
        category="Tabular Foundation",
        device="gpu",
        environment="causilo",
        adapter="tabbench_bio.models.causilo:CausiloModel",
    ),
    "KUMO-TABULAR-MEDIUM": ModelSpec(
        display="Kumo Tabular Medium",
        category="Tabular Foundation",
        device="gpu",
        environment="kumo",
        adapter="tabbench_bio.models.kumo_tabular:KumoTabularMediumModel",
    ),
    "LIMIX-2": ModelSpec(
        display="LimiX2",
        category="Tabular Foundation",
        device="gpu",
        environment="limix2",
        max_classes=10,
        adapter="tabbench_bio.models.limix_2:LimiX2Model",
    ),
    "REALTABPFN-V2": ModelSpec(
        display="RealTabPFN v2", category="Tabular Foundation", device="gpu", max_features=500
    ),
    "REALTABPFN-V2.5": ModelSpec(
        display="RealTabPFN 2.5", category="Tabular Foundation", device="gpu", max_features=2000
    ),
    "TABPFN-V3": ModelSpec(
        display="TabPFN 3",
        category="Tabular Foundation",
        device="gpu",
        adapter="tabbench_bio.models.tabpfn_v3:TabPFNV3Model",
        max_features=10000,
        aliases=("TABPFNV3",),
    ),
    "TABPFN-V3.5": ModelSpec(
        display="TabPFN 3.5",
        category="Tabular Foundation",
        device="gpu",
        environment="tabpfn35",
        adapter="tabbench_bio.models.tabpfn_v3_5:TabPFNV35Model",
        max_features=20000,
        aliases=("TABPFNV35",),
    ),
    "TABPFN-WIDE": ModelSpec(
        display="TabPFN Wide (8k)",
        category="Tabular Foundation",
        device="gpu",
        adapter="tabbench_bio.models.tabpfn_wide:TabPFNWideModel",
        supported_tasks=("classification",),
        aliases=("TABPFNWIDE",),
    ),
    "TABPFN-WIDE-5K-NE3": ModelSpec(
        display="TabPFN Wide 5k (ne3)",
        category="Tabular Foundation",
        device="gpu",
        adapter="tabbench_bio.models.tabpfn_wide:TabPFNWideModel",
        supported_tasks=("classification",),
        hyperparameters=(("model_name", "wide-v2-5k"), ("n_estimators", 3)),
    ),
    "TABFM": ModelSpec(
        display="TabFM",
        category="Tabular Foundation",
        device="gpu",
        adapter="tabbench_bio.models.tabfm:TabFMModel",
        max_features=2000,
    ),
    "TABDPT": ModelSpec(
        display="TabDPT",
        category="Tabular Foundation",
        device="gpu",
        max_features=2500,
        training_data_overlap=True,
    ),
    "TABICL": ModelSpec(
        display="TabICL", category="Tabular Foundation", device="gpu", max_features=2000
    ),
    "AUTOGLUON": ModelSpec(display="AutoGluon", category="AutoML", device="gpu"),
    "FASTAI": ModelSpec(display="FastAI", category="Deep Learning", device="gpu"),
    "TABPFN": ModelSpec(display="TabPFN", category="Tabular Foundation", device="gpu"),
}

MODEL_ALIASES = {alias: key for key, spec in MODEL_REGISTRY.items() for alias in spec.aliases}
MODEL_CATEGORY = {key: spec.category for key, spec in MODEL_REGISTRY.items()}
MODEL_DISPLAY = {key: spec.display for key, spec in MODEL_REGISTRY.items()}


def canonical_model_key(key: str) -> str:
    key = key.upper()
    return MODEL_ALIASES[key] if key in MODEL_ALIASES else key


def model_entry(entry: dict) -> dict:
    key = canonical_model_key(entry["key"])
    parent = canonical_model_key(entry["base_model"]) if "base_model" in entry else key
    defaults = {}
    if parent in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[parent]
        defaults = {"environment": spec.environment, "device": spec.device}
    return {**defaults, **entry, "key": key}
