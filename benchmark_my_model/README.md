# Benchmark your model

Compare your scikit-learn estimator with the published benchmark and get an HTML
report with rankings, plots and per-dataset results. For resumable runs and SQLite
results you can merge into the benchmark, use the [integration guide](INTEGRATION.md).

## Quickstart

From the repository root, create a Python 3.13 environment:

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[bio]"
```

On Windows, activate it with `.venv\Scripts\Activate.ps1`. Install your model's
own dependencies in the same environment.

Edit [my_model.py](my_model.py), then start with two datasets:

```sh
tabbench-bio evaluate --model benchmark_my_model/my_model.py \
  --name SVM --dataset OpenML-1083 --dataset OpenML-1088
```

Open `my_model_results/report/report.html`. The helper uses the latest published
reference by default and downloads it when needed. Only your model is fitted,
on the reference's frozen folds where DUMMY passed.

## Supply your estimator

Define `create_model(task)` in a Python file. It receives `"classification"` or
`"regression"` and must return a cloneable estimator with `fit(X, y)` and `predict(X)`.
For example:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(task):
    model = (
        LogisticRegression(max_iter=1000, random_state=0) if task == "classification" else Ridge()
    )
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        model,
    )
```

Features arrive as NumPy arrays. Put imputation, scaling and feature selection
inside the pipeline so they are fitted only on training rows. Each fold gets a
fresh clone. For custom estimator classes, inherit the appropriate scikit-learn
classifier or regressor mixin before `BaseEstimator` and expose constructor parameters.

## Choose a comparison

The default is classification in `cap_10000_n100`: up to 10,000 features and 100
training rows. Repeat `--dataset ID` to select tasks; omit it to use all datasets
of the selected task type.

To evaluate both classification and regression:

```sh
tabbench-bio evaluate --model benchmark_my_model/my_model.py \
  --name SVM --task overall --output results/svm-overall
```

Use `--task regression` for regression alone. Other useful options are:

- `--baseline results.sqlite` to use a particular local reference.
- `--cell cap_2000_n100` to select another published feature/sample budget.
- `--threads 4` to limit CPU threads during fitting.
- `--output my_results` to choose a new output directory.

Missing datasets download automatically. Some sources require credentials; start
with the public datasets in the quickstart before running the full comparison.
The helper uses the active Python environment and does not select model profiles.

## Read and regenerate results

The report shows Elo rankings, uncertainty, dataset scores, paired comparisons and
coverage. RF anchors Elo at 1000. Classification uses macro-F1; regression uses
RMSE. A dataset subset produces a ranking for that subset.

| Output | Contents |
|---|---|
| `report/report.html` | The comparison report |
| `report/` | Ranking tables, plots and LaTeX exports |
| `fold_metrics.csv` | Your model's scores and timings |
| `metrics/` | Scored folds for your model and the reference models |
| `attempts.csv` | Passed, failed and skipped outcomes |

Regenerate a report from saved metrics without fitting again:

```sh
tabbench-bio evaluate --report-only --name SVM --output my_model_results
```

Use the same `--name`, `--task` and `--output` as the original run. Training runs
require a new output directory and cannot resume. The [registered-model workflow](INTEGRATION.md)
supports resumption and produces mergeable SQLite results.
