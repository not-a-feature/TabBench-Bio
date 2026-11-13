"""AutoGluon adapter for the released LightPFN 1.5.1 estimators."""

from importlib.metadata import version

import torch
from autogluon.core.models import AbstractModel
from lightpfn import LightPFNClassifier, LightPFNRegressor

CHECKPOINT_REVISION = "00a2536cfd76dbac121e63b1da99b580c2be67c3"


class LightPFNModel(AbstractModel):
    ag_key = "LIGHTPFN"
    ag_name = "LightPFN"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        if version("lightpfn") != "1.5.1":
            raise RuntimeError("LIGHTPFN requires lightpfn==1.5.1; install the lightpfn profile.")
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("LightPFN requested a GPU, but CUDA is unavailable.")
        if self.problem_type != "regression" and not 2 <= self.num_classes <= 10:
            raise ValueError("LightPFN supports 2-10 classes.")
        self._fit_device = "cuda:0" if num_gpus else "cpu"
        estimator = LightPFNRegressor if self.problem_type == "regression" else LightPFNClassifier
        self.model = estimator(
            device=self._fit_device,
            n_threads=num_cpus,
            random_state=0,
            repo_id="ueuegio/LightPFN",
            revision=CHECKPOINT_REVISION,
        )
        # Let LightPFN handle categoricals and missing values with its released defaults.
        self.model.fit(self.preprocess(X), y)

    def _predict_proba(self, X, **kwargs):
        predict = (
            self.model.predict if self.problem_type == "regression" else self.model.predict_proba
        )
        return self._convert_proba_to_unified_form(predict(self.preprocess(X)))

    def get_device(self):
        return self._fit_device

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass", "regression"]
