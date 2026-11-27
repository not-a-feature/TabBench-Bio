"""AutoGluon adapters for Causilo 1.0 and 1.5."""

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from causilo import CausiloClassifier, CausiloRegressor

CHECKPOINT_REVISION = "94f2bd91db0737d4da59f347910662905ecb5a09"


class CausiloModel(AbstractModel):
    ag_key = "CAUSILO"
    ag_name = "Causilo"

    def _estimator_options(self):
        from causilo.checkpoints import RELEASE_COMMIT

        assert RELEASE_COMMIT == CHECKPOINT_REVISION, (
            "Unexpected Causilo checkpoint revision; reinstall the causilo profile."
        )
        return {"n_estimators": 8}

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        estimator_options = self._estimator_options()
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("Causilo requested a GPU, but CUDA is unavailable.")
        torch.set_num_threads(num_cpus)
        estimator = CausiloRegressor if self.problem_type == "regression" else CausiloClassifier
        self.model = estimator(
            device="cuda:0" if num_gpus else "cpu",
            random_state=42,
            use_kv_cache=False,
            retain_preprocessing=True,
            **estimator_options,
        )
        self.model.fit(self.preprocess(X), y)
        if self.problem_type != "regression":
            assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
                "Causilo probabilities must follow AutoGluon's class order."
            )

    def _predict_proba(self, X, **kwargs):
        regression = self.problem_type == "regression"
        X = self.preprocess(X)
        prediction = np.asarray(
            self.model.predict(X, output_type="mean") if regression else self.model.predict_proba(X)
        )
        expected = (len(X),) if regression else (len(X), self.num_classes)
        assert prediction.shape == expected, "Unexpected Causilo prediction shape."
        assert np.isfinite(prediction).all(), "Causilo returned non-finite predictions."
        if not regression:
            assert (prediction >= 0).all() and np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
        return self._convert_proba_to_unified_form(prediction)

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass", "regression"]


class Causilo15Model(CausiloModel):
    ag_key = "CAUSILO-1.5"
    ag_name = "Causilo15"

    def _more_tags(self):
        return {"can_refit_full": True}

    def _estimator_options(self):
        return {"model_version": "1.5", "max_context": None}
