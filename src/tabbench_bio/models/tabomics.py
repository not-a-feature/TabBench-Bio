"""AutoGluon adapter for the released TabOmics v2.5 classifier."""

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from tabomics import Classifier


class TabOmicsModel(AbstractModel):
    ag_key = "TABOMICS"
    ag_name = "TabOmics"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert self.problem_type in self.supported_problem_types()
        assert 2 <= self.num_classes <= 10, "TabOmics supports 2-10 classes."
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("TabOmics requested a GPU, but CUDA is unavailable.")
        torch.set_num_threads(num_cpus)
        self._fit_device = "cuda:0" if num_gpus else "cpu"
        self.model = Classifier(
            model="v2.5", device=self._fit_device, n_estimators=self.params["n_estimators"]
        ).fit(self.preprocess(X), y)
        assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
            "TabOmics probabilities must follow AutoGluon's class order."
        )

    def _predict_proba(self, X, **kwargs):
        prediction = np.asarray(self.model.predict_proba(self.preprocess(X)))
        assert prediction.shape == (
            len(X),
            self.num_classes,
        ), "Unexpected TabOmics prediction shape."
        assert np.isfinite(prediction).all(), "TabOmics returned non-finite predictions."
        assert (prediction >= 0).all() and np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
        return self._convert_proba_to_unified_form(prediction)

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 1)

    def get_device(self):
        return self._fit_device

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass"]
