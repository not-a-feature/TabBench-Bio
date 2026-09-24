"""AutoGluon adapter for the standard TabPFN-3.5 checkpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _estimator(problem_type: str, device: str, categorical_features_indices: list[int] | None):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    estimator = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
    return estimator.create_default_for_version(
        ModelVersion.V3_5,
        device=device,
        n_estimators=8,
        categorical_features_indices=categorical_features_indices,
        ignore_pretraining_limits=True,
        random_state=0,
    )


class TabPFNV35Model(TabPFNModel):
    """Select TabPFN-3.5 explicitly for classification and regression."""

    ag_key = "TABPFN-V3.5"
    ag_name = "TabPFNV35"
    _default_auxiliary_params_extra = {"max_features": 20_000, "max_classes": None}

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
        import torch

        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("TabPFN-3.5 requested a GPU, but CUDA is unavailable.")
        self._fit_device = "cuda" if num_gpus else "cpu"
        X = self.preprocess(X, y=y, is_train=True)
        self.model = _estimator(self.problem_type, self._fit_device, self._cat_indices)
        if self.problem_type != "regression" and self.num_classes > 10:
            from tabpfn_extensions.many_class import ManyClassClassifier

            self.model = ManyClassClassifier(estimator=self.model, alphabet_size=10, random_state=0)
        self.model.fit(X, y)

    def get_device(self) -> str:
        return self._fit_device

    def _set_device(self, device: str) -> None:
        if self.problem_type != "regression" and self.num_classes > 10:
            # ECOC fits fresh clones during prediction rather than retaining GPU models.
            self.model.estimator.set_params(device=device)
        else:
            self.model.to(device)
        self._fit_device = device

    def _set_default_params(self) -> None:
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        return []

    def _log_license(self, device: str) -> None:
        logger.log(20, "\tBuilt with TabPFN-3.5 (TabPFN-3.5 License v1.0, non-commercial)")
