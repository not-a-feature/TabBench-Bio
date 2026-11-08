"""Released Seldon classifiers with benchmark-controlled feature and row budgets."""

import numpy as np
import sdm.processing as sp
import torch
from autogluon.core.models import AbstractModel
from neuralk_oss import SeldonClassifier, SeldonRecipe

from tabbench_bio.models.seldon_cache import BoundedSeldonICL

CHECKPOINT_REVISIONS = {
    "seldon": "54479903c8863a22789a9bc7a71dea7bb43dcb06",
    "seldon-mini": "9a7e0cbbe4b42dd49a3a99ef954467a9c789d722",
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
        self.model = SeldonClassifier.from_pretrained(
            self.checkpoint,
            revision=CHECKPOINT_REVISIONS[self.checkpoint],
            device="cuda:0" if num_gpus else "cpu",
            max_context_size=max(200_000, len(X)),
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
        assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
            "Seldon probabilities must follow AutoGluon's class order."
        )

    def _predict_proba(self, X, **kwargs):
        prediction = np.asarray(self.model.predict_proba(self.preprocess(X)))
        assert prediction.shape == (len(X), self.num_classes), "Unexpected Seldon prediction shape."
        assert np.isfinite(prediction).all(), "Seldon returned non-finite predictions."
        assert (prediction >= 0).all() and np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
        return self._convert_proba_to_unified_form(prediction)

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass"]


class SeldonMiniModel(SeldonModel):
    ag_key = "SELDON-MINI"
    ag_name = "SeldonMini"
    checkpoint = "seldon-mini"
