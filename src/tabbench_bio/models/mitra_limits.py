"""Use MITRA's actual fine-tuning split sizes as its initial memory limits."""

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _train_with_actual_sample_limits(
    original_train,
    estimator,
    X_train,
    y_train,
    X_valid,
    y_valid,
    task,
    dim_output,
    n_classes=0,
    time_limit=None,
):
    """Call MITRA's trainer with support/query limits clamped to its actual splits."""
    n_support = len(X_train)
    n_query = len(X_valid)
    assert n_support > 0, "MITRA received an empty support split."
    assert n_query > 0, "MITRA received an empty query split."

    original_create_config = estimator._create_config
    had_instance_override = "_create_config" in estimator.__dict__
    if had_instance_override:
        instance_override = estimator.__dict__["_create_config"]

    def create_config_with_actual_limits(task, dim_output, time_limit=None):
        cfg, model_cls = original_create_config(task, dim_output, time_limit)
        hyperparams = cfg.hyperparams
        initial_support = hyperparams["max_samples_support"]
        initial_query = hyperparams["max_samples_query"]
        hyperparams["max_samples_support"] = min(initial_support, n_support)
        hyperparams["max_samples_query"] = min(initial_query, n_query)
        logger.info(
            "MITRA initial sample limits: support=%d, query=%d (actual splits: %d/%d).",
            hyperparams["max_samples_support"],
            hyperparams["max_samples_query"],
            n_support,
            n_query,
        )
        return cfg, model_cls

    estimator._create_config = create_config_with_actual_limits
    try:
        return original_train(
            estimator,
            X_train,
            y_train,
            X_valid,
            y_valid,
            task,
            dim_output,
            n_classes=n_classes,
            time_limit=time_limit,
        )
    finally:
        if had_instance_override:
            estimator._create_config = instance_override
        else:
            del estimator._create_config


@contextmanager
def mitra_actual_sample_limits():
    """Temporarily clamp MITRA's hard-coded limits to each fine-tuning split."""
    from autogluon.tabular.models.mitra.sklearn_interface import MitraBase

    original_train = MitraBase._train_ensemble

    def train_with_actual_sample_limits(
        estimator,
        X_train,
        y_train,
        X_valid,
        y_valid,
        task,
        dim_output,
        n_classes=0,
        time_limit=None,
    ):
        return _train_with_actual_sample_limits(
            original_train,
            estimator,
            X_train,
            y_train,
            X_valid,
            y_valid,
            task,
            dim_output,
            n_classes=n_classes,
            time_limit=time_limit,
        )

    MitraBase._train_ensemble = train_with_actual_sample_limits
    try:
        yield
    finally:
        assert MitraBase._train_ensemble is train_with_actual_sample_limits
        MitraBase._train_ensemble = original_train
