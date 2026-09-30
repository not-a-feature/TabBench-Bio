"""Published regular feature limits declared in the model registry."""

from tabbench_bio.model_registry import MODEL_REGISTRY

REGULAR_MAX_FEATURES = {
    key: spec.max_features for key, spec in MODEL_REGISTRY.items() if spec.max_features is not None
}
