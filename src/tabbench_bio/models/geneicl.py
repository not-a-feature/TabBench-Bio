"""AutoGluon adapter for GeneICL's released seed-0 checkpoint."""

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from geneicl import CKPT, GeneICL
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder

from tabbench_bio.io_utils import sha256_file

CHECKPOINT_SHA256 = "52609d58f5bb71850998bce8623404166cd651fcac1d450516cbd4eda7c858da"


class GeneICLModel(AbstractModel):
    ag_key = "GENEICL"
    ag_name = "GeneICL"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("GeneICL requested a GPU, but CUDA is unavailable.")
        regression = self.problem_type == "regression"
        if not regression:
            classes = np.unique(y)
            if not 2 <= len(classes) <= 10:
                raise ValueError(
                    "GeneICL supports 2-10 classes; it does not merge or drop classes."
                )
            assert np.array_equal(classes, np.arange(self.num_classes)), (
                "GeneICL requires AutoGluon's consecutive class labels."
            )
        assert sha256_file(CKPT) == CHECKPOINT_SHA256, "Unexpected GeneICL seed-0 checkpoint."
        torch.set_num_threads(num_cpus)
        self._encoder = ColumnTransformer(
            [
                (
                    "category",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                    make_column_selector(dtype_include=["category", "object", "string", "bool"]),
                )
            ],
            remainder="passthrough",
        )
        train = self._encoder.fit_transform(self.preprocess(X)).astype(np.float32)
        self.model = GeneICL(
            mode="regression" if regression else "classification",
            ckpt=CKPT,
            device="cuda:0" if num_gpus else "cpu",
            ensemble_seed=None,
        )
        self.model.fit(train, y)
        if not regression:
            assert np.array_equal(self.model.classes_, np.arange(self.num_classes)), (
                "GeneICL probabilities must follow AutoGluon's class order."
            )

    def _predict_proba(self, X, **kwargs):
        regression = self.problem_type == "regression"
        query = self._encoder.transform(self.preprocess(X)).astype(np.float32)
        prediction = np.asarray(
            self.model.predict(query) if regression else self.model.predict_proba(query)
        )
        expected = (len(X),) if regression else (len(X), self.num_classes)
        assert prediction.shape == expected, "Unexpected GeneICL prediction shape."
        assert np.isfinite(prediction).all(), "GeneICL returned non-finite predictions."
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
