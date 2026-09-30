"""LimiX-2 with training-only preprocessing and its released inference ensembles."""

import json
from importlib.resources import files

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from inference.v2_0.predictor import LimiXPredictor
from inference.v2_0.preprocess import (
    CategoricalFeatureEncoder,
    FilterValidFeatures,
    PolynomialInteractionGenerator,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder

CHECKPOINT_REVISION = "de07b679e74a41b50b9de18251a8fa245e537440"


class _TrainOnlyFilter(FilterValidFeatures):
    def fit(self, x, categorical_features, seed, y, **kwargs):
        train = x[: len(y)]
        self.valid_features = (~np.all(train == train[:1], axis=0)) & (~np.isnan(train).all(axis=0))
        self.invalid_indices = ~self.valid_features
        assert self.valid_features.any(), "LimiX-2: no non-constant training features."
        self.categorical_idx = [
            new
            for new, old in enumerate(np.flatnonzero(self.valid_features))
            if old in categorical_features
        ]
        return self.categorical_idx


class _TrainOnlyEncoder(CategoricalFeatureEncoder):
    def fit_transform(self, x, categorical_features, seed, *, y, **kwargs):
        train, indices = super().fit_transform(x[: len(y)], categorical_features, seed)
        query = x[len(y) :]
        if self.transformer is not None:
            query = self.transformer.transform(query)
            if self.encoding_strategy.endswith("_shuffled"):
                for column, permutation in self.category_mappings.items():
                    known = ~np.isnan(query[:, column])
                    query[known, column] = permutation[query[known, column].astype(int)]
        return np.concatenate([train, query]), indices


class _TrainOnlyInteractions(PolynomialInteractionGenerator):
    def fit(self, x, categorical_features, seed, *, y, **kwargs):
        return super().fit(x[: len(y)], categorical_features, seed)


class _Predictor(LimiXPredictor):
    class CacheManager(LimiXPredictor.CacheManager):
        def __init__(self, cache_dir):
            # Upstream creates its hard-coded directory even with use_data_cache=False.
            super().__init__(cache_dir=HF_HUB_CACHE + "/limix-preprocess")

    def predict(self, x_train, y_train, x_test, **kwargs):
        self._train_rows = len(y_train)
        return super().predict(x_train, y_train, x_test, **kwargs)

    def get_categorical_features_indices(self, x):
        return super().get_categorical_features_indices(x[: self._train_rows])

    def convert_category2num(self, x, **kwargs):
        encoder = ColumnTransformer(
            [
                (
                    "category",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                    make_column_selector(dtype_include=["category", "object", "string", "bool"]),
                )
            ],
            remainder="passthrough",
        )
        encoder.fit(x.iloc[: self._train_rows])
        return encoder.transform(x).astype(np.float64)

    def build_preprocess_pipeline(self):
        super().build_preprocess_pipeline()
        for pipeline in self.preprocess_pipelines:
            for index, step in enumerate(pipeline):
                if isinstance(step, FilterValidFeatures):
                    pipeline[index] = _TrainOnlyFilter()
                elif isinstance(step, CategoricalFeatureEncoder):
                    pipeline[index] = _TrainOnlyEncoder(
                        encoding_strategy=step.encoding_strategy,
                        onehot_size_fallback=step.onehot_size_fallback,
                    )
                elif isinstance(step, PolynomialInteractionGenerator):
                    pipeline[index] = _TrainOnlyInteractions(
                        max_interaction_features=step.max_interactions,
                    )


class LimiX2Model(AbstractModel):
    """Adapt the frozen LimiX-2 predictor to AutoGluon's fit/predict interface."""

    ag_key = "LIMIX-2"
    ag_name = "LimiX2"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("LimiX-2 requested a GPU, but CUDA is unavailable.")
        self._fit_device = "cuda:0" if num_gpus else "cpu"
        torch.set_num_threads(num_cpus)
        self._train_y = np.asarray(y).copy()
        if self.problem_type != "regression":
            classes = np.unique(self._train_y)
            if not 2 <= len(classes) <= 10:
                raise ValueError(
                    "LimiX-2 supports 2–10 classes; it does not merge or drop classes."
                )
            assert np.array_equal(classes, np.arange(self.num_classes)), (
                "LimiX-2 requires AutoGluon's consecutive class labels."
            )
        self._train_X = self.preprocess(X).copy()
        task = "reg" if self.problem_type == "regression" else "cls"
        config = json.loads(
            files("config").joinpath(f"{task}_default_noretrieval_v2.json").read_text()
        )
        # An OOM must not silently shrink SVD or discard ensemble members.
        config["adaptive_svd"]["retry_on_cuda_resource_error"] = False
        checkpoint = hf_hub_download(
            repo_id="stable-ai/LimiX-2",
            filename="LimiX-2.ckpt",
            revision=CHECKPOINT_REVISION,
        )
        self.model = _Predictor(
            device=torch.device(self._fit_device),
            model_path=checkpoint,
            inference_config=config,
            inference_with_DDP=False,
            gpu_ids=[0] if num_gpus else None,
            use_data_cache=False,
            preprocess_num_jobs=num_cpus,
            seed=0,
        )

    def _predict_proba(self, X, **kwargs):
        regression = self.problem_type == "regression"
        prediction = np.asarray(
            self.model.predict(
                self._train_X,
                self._train_y,
                self.preprocess(X),
                task_type="Regression" if regression else "Classification",
            )
        )
        expected = (len(X),) if regression else (len(X), self.num_classes)
        assert prediction.shape == expected, "Unexpected LimiX-2 prediction shape."
        assert np.isfinite(prediction).all(), "LimiX-2 returned non-finite predictions."
        if not regression:
            assert (prediction >= 0).all() and np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
            assert np.array_equal(self.model.classes, np.arange(self.num_classes))
        return self._convert_proba_to_unified_form(prediction)

    def _get_default_resources(self):
        return torch.get_num_threads(), int(torch.cuda.is_available())

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass", "regression"]
