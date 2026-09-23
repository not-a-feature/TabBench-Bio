"""Replace the estimator returned here with your own scikit-learn-compatible model."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(task: str):
    estimator = (
        LogisticRegression(max_iter=1000, random_state=0) if task == "classification" else Ridge()
    )
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
