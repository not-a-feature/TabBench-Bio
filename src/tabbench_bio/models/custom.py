"""Deprecated compatibility view; use model_registry.MODEL_REGISTRY instead."""

from tabbench_bio.model_registry import MODEL_REGISTRY

CUSTOM_MODELS = {
    key: {
        "adapter": spec.adapter,
        "environment": spec.environment,
        "device": spec.device,
        "max_features": spec.max_features,
        "classification_only": "regression" not in spec.supported_tasks,
    }
    for key, spec in MODEL_REGISTRY.items()
    if spec.adapter is not None
}
