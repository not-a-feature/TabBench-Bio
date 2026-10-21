"""SVM estimators for the clone-free TabBench-Bio model interface."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def create_model(task: str):
    if task == "classification":
        estimator = SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=1024, random_state=0)
    elif task == "regression":
        estimator = SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="scale", cache_size=1024)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        estimator,
    )
