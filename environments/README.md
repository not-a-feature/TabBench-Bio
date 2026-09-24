# Model environments

Each registered model selects an `environment` profile. The profile's requirements
live in `environments/<profile>.txt`, and its installed Python environment lives in
`.venvs/<profile>/`. Compatible models can share one profile.

| Profile | Models |
|---|---|
| `standard` | Existing model roster except TabPFN 3.5, using the established `full` extra |
| `tabpfn35` | TabPFN 3.5, using its separate extra |

The model roster in `configs/models/all.json` declares these assignments. Entries in
`src/tabbench_bio/models/custom.py` take precedence. TabPFN-Wide and TabPFN 3.5 require
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
