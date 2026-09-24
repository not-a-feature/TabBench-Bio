"""Custom adapters available to the standard benchmark commands."""

CUSTOM_MODELS = {
    "TABPFN-V3.5": {
        "adapter": "tabbench_bio.models.tabpfn_v3_5:TabPFNV35Model",
        "environment": "tabpfn35",
        "device": "gpu",
        "max_features": 20_000,
        "classification_only": False,
    },
}
