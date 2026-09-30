"""Kumo Tabular Medium with its released ensemble and no internal column cap."""

import numpy as np
import sdm
import sdm.processing as sp
import torch
from autogluon.core.models import AbstractModel
from huggingface_hub import hf_hub_download
from sdm.processing.execution import RecipeExecution

CHECKPOINT_REVISION = "bd7fa122b516c7355583873ffcf38f5e79403ebd"


class KumoTabularMediumModel(AbstractModel):
    ag_key = "KUMO-TABULAR-MEDIUM"
    ag_name = "KumoTabularMedium"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        assert num_cpus >= 1
        if num_gpus and not torch.cuda.is_available():
            raise RuntimeError("Kumo requested a GPU, but CUDA is unavailable.")
        torch.set_num_threads(num_cpus)
        self._fit_device = "cuda:0" if num_gpus else "cpu"
        regression = self.problem_type == "regression"
        task = "regression" if regression else "classification"
        self.model = sdm.models.KumoTabular(
            task=task, size="medium", pretrained=False, device="meta"
        )
        checkpoint = hf_hub_download(
            repo_id="nvidia/Kumo-Tabular",
            filename=f"medium/{'regressor' if regression else 'classifier'}.pt",
            revision=CHECKPOINT_REVISION,
        )
        self.model.models[task].load_state_dict(
            torch.load(checkpoint, map_location=self._fit_device, weights_only=True), assign=True
        )
        X = self.preprocess(X)
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        x = sdm.TableTensor.from_pandas(X, stypes=self._stypes, device=self._fit_device)
        target = sdm.TableTensor.from_pandas(
            y.rename("target").to_frame(),
            stypes={"target": "numerical" if regression else "categorical"},
            device=self._fit_device,
        )
        recipe = self.model.default_recipe()
        recipe.features = sp.Sequential(
            *(step for step in recipe.features if not isinstance(step, sp.SelectColumns))
        )
        self._recipe_execution = RecipeExecution(recipe)
        generator = torch.Generator(self._fit_device).manual_seed(42)
        with torch.inference_mode():
            contexts = self._recipe_execution.fit_transform(
                x=x,
                y=target,
                related_tables=None,
                num_members=8,
                generator=generator,
            )
        self._contexts = tuple(
            context._replace(x=context.x.cpu(), y=context.y.cpu()) for context in contexts
        )
        self._rng_state = generator.get_state()

    def _predict_proba(self, X, **kwargs):
        x = sdm.TableTensor.from_pandas(
            self.preprocess(X), stypes=self._stypes, device=self._fit_device
        )
        regression = self.problem_type == "regression"
        with torch.inference_mode():
            queries = self._recipe_execution.transform(x=x, related_tables=None)
            generator = torch.Generator(self._fit_device).set_state(self._rng_state)
            outputs = []
            for context, query in zip(self._contexts, queries, strict=True):
                context = context._replace(
                    x=context.x.to(self._fit_device), y=context.y.to(self._fit_device)
                )
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=self._fit_device.startswith("cuda")
                ):
                    outputs += self.model._forward_members(
                        contexts=[context],
                        queries=[query],
                        estimator_batch_size=None,
                        generator=generator,
                    )
            if regression:
                outputs = list(self._recipe_execution.inverse_transform_target(outputs))
            output = self._recipe_execution.transform_output(outputs)
        if regression:
            prediction = output.numerical.float().mean(dim=-1).cpu().numpy()
        else:
            columns = output.columns[sdm.Stype.numerical]
            order = [columns.index(str(i)) for i in range(self.num_classes)]
            prediction = output.numerical[..., order].float().cpu().numpy()
        expected = (len(X),) if regression else (len(X), self.num_classes)
        assert prediction.shape == expected, "Unexpected Kumo prediction shape."
        assert np.isfinite(prediction).all(), "Kumo returned non-finite predictions."
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
