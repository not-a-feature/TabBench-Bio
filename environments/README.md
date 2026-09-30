# Model environments

Each registered model selects an `environment` profile. The profile's requirements
live in `environments/<profile>.txt`, and its installed Python environment lives in
`.venvs/<profile>/`. Compatible models can share one profile.

| Profile | Models |
|---|---|
| `standard` | Existing model roster except TabPFN 3.5, using the established `full` extra |
| `tabpfn35` | TabPFN 3.5, using its separate extra |
| `causilo` | Causilo, with its pinned source revision |
| `limix2` | LimiX2, with its pinned source revision and PyTorch version |
| `kumo` | Kumo Tabular Medium, with pinned SDM source and checkpoint revisions |

The model registry in `src/tabbench_bio/model_registry.py` declares these assignments.
The roster in `configs/models/all.json` selects benchmark models; explicit per-run
environment and device overrides remain supported. TabPFN-Wide and TabPFN 3.5 require
different versions of the `tabpfn` package, so they use separate profiles.

Install a profile from the repository root, replacing `standard` with its name:

```bash
uv venv .venvs/standard --python 3.12
uv pip install --python .venvs/standard/bin/python -r environments/standard.txt
```

On Windows, use `.venvs/standard/Scripts/python.exe` in the install command.
Create the environment on each machine where you run models. Virtual environments
are ignored by Git and should not be copied between machines.

Keep the core `.venv` active and run `tabbench-bio MODELKEY`. The command selects the
model's environment automatically. The generic Slurm launcher uses the same command.
Missing profiles or environments stop the run before fitting. No packages are
installed automatically.

Requirements files are not complete dependency locks. Keep installed environments
unchanged during a run and its resumes. The runner records the profile and interpreter
path, but does not compare installed package versions when resuming.

The lower-level `tabbench-bio run`, feature-sweep script and standalone scikit-learn
helper use the interpreter that launches them. They do not switch profiles.
Database merging and leaderboard generation need only the core environment.

To add a model, follow the [integration guide](../benchmark_my_model/INTEGRATION.md).

`KUMO-TABULAR-MEDIUM` is a tabular foundation model using the Medium checkpoint,
eight sequential ensemble members, and uncached attention using training-only processed context. Its internal 500-column
selection is removed; the benchmark's feature cells still apply. Other released
preprocessing remains unchanged. Code is Apache-2.0; weights are OpenMDW-1.1.
Run with `tabbench-bio KUMO-TABULAR-MEDIUM` after installing the `kumo` profile.
