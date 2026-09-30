"""Custom adapters available to the standard benchmark commands."""

CUSTOM_MODELS = {
    "KUMO-TABULAR-MEDIUM": {
        "adapter": "tabbench_bio.models.kumo_tabular:KumoTabularMediumModel",
        "environment": "kumo",
        "device": "gpu",
        "max_features": None,
        "classification_only": False,
    },
    "CAUSILO": {
        "adapter": "tabbench_bio.models.causilo:CausiloModel",
        "environment": "causilo",
        "device": "gpu",
        "max_features": None,
        "classification_only": False,
    },
    "LIMIX-2": {
        "adapter": "tabbench_bio.models.limix_2:LimiX2Model",
        "environment": "limix2",
        "device": "gpu",
        "max_features": None,
        "classification_only": False,
    },
    "TABPFN-V3.5": {
        "adapter": "tabbench_bio.models.tabpfn_v3_5:TabPFNV35Model",
        "environment": "tabpfn35",
        "device": "gpu",
        "max_features": 20_000,
        "classification_only": False,
    },
}
