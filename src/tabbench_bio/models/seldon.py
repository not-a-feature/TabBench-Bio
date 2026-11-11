"""Released Seldon models with benchmark-controlled feature and row budgets."""

import numpy as np
import sdm.processing as sp
import torch
from autogluon.core.models import AbstractModel
from neuralk_oss import SeldonClassifier, SeldonRecipe, SeldonRegressor

from tabbench_bio.models.seldon_cache import BoundedSeldonICL

CHECKPOINT_REVISIONS = {
    "seldon": "54479903c8863a22789a9bc7a71dea7bb43dcb06",
    "seldon-mini": "9a7e0cbbe4b42dd49a3a99ef954467a9c789d722",
}
REGRESSION_CHECKPOINT_REVISIONS = {
    "seldon": "290b8e3e6f3d5748c02d2414742cc4b4f9ac01d3",
    "seldon-mini": "67d1672fa193a82d9bf5a3b8f921fa5c9587c280",
}


class _FullFeatureRecipe(SeldonRecipe):
    def features(self):
        return [step for step in super().features() if not isinstance(step, sp.SelectColumns)]


class SeldonModel(AbstractModel):
    ag_key = "SELDON"
    ag_name = "Seldon"
    checkpoint = "seldon"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert self.problem_type in self.supported_problem_types()
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("Seldon requested a GPU, but CUDA is unavailable.")
        torch.set_num_threads(num_cpus)
        options = {
            "device": "cuda:0" if num_gpus else "cpu",
            "max_context_size": max(200_000, len(X)),
        }
        if self.problem_type == "regression":
            self.model = SeldonRegressor.from_pretrained(
                self.checkpoint,
                revision=REGRESSION_CHECKPOINT_REVISIONS[self.checkpoint],
                **options,
            )
        else:
            # Preserve the original classification weights and packaged recipe.
            # Their older repository revision contains a single unnamed checkpoint.
            self.model = SeldonClassifier.from_config(
                self.checkpoint,
                checkpoint=f"Neuralk-AI/{self.checkpoint}",
                revision=CHECKPOINT_REVISIONS[self.checkpoint],
                **options,
            )
        # Keep the packaged variant's recipe settings and ensemble size.
        recipe = self.model.estimator.recipe
        self.model.estimator.recipe = _FullFeatureRecipe(**vars(recipe))
        original = self.model.estimator
        # Sequential members bound GPU memory; CPU encodings use an evictable
        # budget rather than SDM's unbounded pinned context cache.
        self.model.estimator = BoundedSeldonICL(
            original.model, original.recipe, query_chunk_size=16
        ).to(self.model.device)
        self.model.estimator_batch_size = 1
        self.model.fit(self.preprocess(X), y)
        if self.problem_type != "regression":
            assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
                "Seldon probabilities must follow AutoGluon's class order."
            )

    def _predict_proba(self, X, **kwargs):
        regression = self.problem_type == "regression"
        predict = self.model.predict if regression else self.model.predict_proba
        prediction = np.asarray(predict(self.preprocess(X)))
        expected = (len(X),) if regression else (len(X), self.num_classes)
        assert prediction.shape == expected, "Unexpected Seldon prediction shape."
        assert np.isfinite(prediction).all(), "Seldon returned non-finite predictions."
        if not regression:
            assert (prediction >= 0).all() and np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
        return self._convert_proba_to_unified_form(prediction)

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass", "regression"]


class SeldonMiniModel(SeldonModel):
    ag_key = "SELDON-MINI"
    ag_name = "SeldonMini"
    checkpoint = "seldon-mini"
