"""Published regular feature limits declared by the model adapters."""

from tabbench_bio.models.custom import CUSTOM_MODELS

REGULAR_MAX_FEATURES = {
    "MITRA": 500,
    "REALTABPFN-V2": 500,
    "REALTABPFN-V2.5": 2_000,
    "TABDPT": 2_500,
    "TABFM": 2_000,
    "TABICL": 2_000,
    "TABPFN-V3": 10_000,
}
REGULAR_MAX_FEATURES.update(
    {
        key: entry["max_features"]
        for key, entry in CUSTOM_MODELS.items()
        if entry["max_features"] is not None
    }
)
