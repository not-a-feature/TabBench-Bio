"""AutoGluon model wrapper for the default TabPFN-3 checkpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel
from sklearn.base import BaseEstimator, ClassifierMixin

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
_NATIVE_MAX_CLASSES = 10


def _classifier(device: str, categorical_features_indices: list[int] | None):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device=device,
        n_estimators=8,
        categorical_features_indices=categorical_features_indices,
        ignore_pretraining_limits=True,
        random_state=0,
    )


def _regressor(device: str, categorical_features_indices: list[int] | None):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V3,
        device=device,
        n_estimators=8,
        categorical_features_indices=categorical_features_indices,
        ignore_pretraining_limits=True,
        random_state=0,
    )


class _CloneableTabPFNV3(BaseEstimator, ClassifierMixin):
    """Build TabPFN-3 lazily so sklearn ECOC cloning never copies loaded weights."""

    def __init__(
        self,
        device: str = "cuda",
        categorical_features_indices: list[int] | None = None,
    ) -> None:
        self.device = device
        self.categorical_features_indices = categorical_features_indices

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = _classifier(self.device, self.categorical_features_indices)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def devices_(self):
        return self.model_.devices_

    def to(self, device):
        self.model_.to(device)
        self.device = str(device)
        return self


class TabPFNV3Model(TabPFNModel):
    """Expose explicit TabPFN-3 classifier and regressor defaults as ``TABPFN-V3``."""

    ag_key = "TABPFN-V3"
    ag_name = "TabPFNV3"
    _default_auxiliary_params_extra = {"max_features": 10_000}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        try:
            import tabpfn
        except ImportError as exc:
            raise ImportError(
                "TabPFNV3Model requires tabpfn>=8.0.3. Install with: pip install 'tabpfn>=8.0.3'",
            ) from exc

        del tabpfn
        import torch

        device = "cuda" if num_gpus and torch.cuda.is_available() else "cpu"
        if num_gpus and device == "cpu":
            logger.warning("TabPFN-3: GPU requested but CUDA unavailable; running on CPU.")

        X = self.preprocess(X, y=y, is_train=True)

        if self.problem_type == "regression":
            self.model = _regressor(device, self._cat_indices)
            self.model.fit(X, y)
            return

        base_model = _CloneableTabPFNV3(
            device=device,
            categorical_features_indices=self._cat_indices,
        )
        many_class_threshold = _NATIVE_MAX_CLASSES
        if self.num_classes is not None and self.num_classes > many_class_threshold:
            try:
                from tabpfn_extensions.many_class import ManyClassClassifier
            except ImportError as exc:
                raise ImportError(
                    f"TabPFN-3: {self.num_classes} classes exceeds the native limit "
                    f"({many_class_threshold}); install tabpfn-extensions for ECOC.",
                ) from exc
            self.model = ManyClassClassifier(
                estimator=base_model,
                alphabet_size=many_class_threshold,
            )
        else:
            self.model = base_model

        self.model.fit(X, y)

    def _set_default_params(self) -> None:
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        return []

    def _log_license(self, device: str) -> None:
        logger.log(20, "\tBuilt with TabPFN-3 (TabPFN-3 License v1.0, non-commercial)")
