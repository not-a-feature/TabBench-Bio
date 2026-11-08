"""Released FTFM-1 classification ensemble, unchanged from the reference run."""

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from ftfm import FTFMClassifier
from huggingface_hub import snapshot_download

CHECKPOINT_REVISION = "a8362b12392a80d2e4618e91e6ce02b0bed2781d"


class FTFMModel(AbstractModel):
    ag_key = "FTFM"
    ag_name = "FTFM"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert self.problem_type in self.supported_problem_types()
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("FTFM requested a GPU, but CUDA is unavailable.")
        torch.set_num_threads(num_cpus)
        checkpoint = snapshot_download(
            repo_id="josifgrabocka/ftfm",
            revision=CHECKPOINT_REVISION,
            allow_patterns=["config.json", "model.safetensors"],
        )
        self.model = FTFMClassifier(
            n_estimators=8,
            random_state=42,
            device="cuda:0" if num_gpus else "cpu",
            view_batch_size=1,
            model_path=checkpoint,
            allow_auto_download=False,
        ).fit(self.preprocess(X), y)
        assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
            "FTFM probabilities must follow AutoGluon's class order."
        )

    def _predict_proba(self, X, **kwargs):
        prediction = np.asarray(self.model.predict_proba(self.preprocess(X)))
        assert prediction.shape == (len(X), self.num_classes), "Unexpected FTFM prediction shape."
        assert np.isfinite(prediction).all(), "FTFM returned non-finite predictions."
        assert (prediction >= 0).all() and np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
        return self._convert_proba_to_unified_form(prediction)

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    def _more_tags(self):
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass"]
