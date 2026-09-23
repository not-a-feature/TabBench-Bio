import json

import pandas as pd

from scripts import build_site


def test_dataset_explorer_exports_each_cell_and_preserves_metric_means(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pd.DataFrame(
        [
            {"dataset": "classification", "cell": "small", "model": "RF", "f1_macro__mean": 0.25},
            {"dataset": "classification", "cell": "large", "model": "RF", "f1_macro__mean": 0.75},
            {"dataset": "classification", "cell": "large", "model": "ALT", "f1_macro__mean": None},
        ]
    ).to_csv(inputs / "sweep_summary_strict.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "regression",
                "cell": "small",
                "model": "RF",
                "rmse": 1.0,
                "mae": 0.5,
                "r2": 0.2,
            },
            {
                "dataset": "regression",
                "cell": "small",
                "model": "RF",
                "rmse": 3.0,
                "mae": 1.5,
                "r2": 0.6,
            },
            {
                "dataset": "regression",
                "cell": "large",
                "model": "RF",
                "rmse": 0.5,
                "mae": 0.25,
                "r2": 0.8,
            },
        ]
    ).to_csv(inputs / "sweep_metrics_regression_strict.csv", index=False)
    monkeypatch.setattr(build_site, "GENERATED_DATA", inputs)
    dashboard = {
        "meta": {"reference_cell": "small"},
        "models": {"RF": {}, "ALT": {}},
        "cell_options": [{"id": "small"}, {"id": "large"}],
        "datasets": [
            {
                "dataset_id": "classification",
                "task": "Classification",
                "performance": {
                    "cell": "small",
                    "metrics": [{"key": "f1_macro", "better": "high"}],
                },
            },
            {
                "dataset_id": "regression",
                "task": "Regression",
                "performance": {
                    "cell": "large",
                    "metrics": [{"key": "rmse", "better": "low"}, {"key": "r2", "better": "high"}],
                },
            },
        ],
    }
    build_site.write_dataset_explorer(tmp_path / "site", dashboard)
    output = tmp_path / "site" / "data" / "datasets"
    index = json.loads((output / "index.json").read_text("utf-8"))
    assert len(index["datasets"]) == 2
    exported = {
        dataset["dataset_id"]: json.loads((output / dataset["scores_file"]).read_text("utf-8"))[
            "scores"
        ]
        for dataset in index["datasets"]
    }
    assert exported["classification"] == [
        {"cell": "small", "model_id": "RF", "values": {"f1_macro": 0.25}},
        {"cell": "large", "model_id": "RF", "values": {"f1_macro": 0.75}},
        {"cell": "large", "model_id": "ALT", "values": {"f1_macro": None}},
    ]
    regression = {row["cell"]: row["values"] for row in exported["regression"]}
    assert regression == {"small": {"rmse": 2.0, "r2": 0.4}, "large": {"rmse": 0.5, "r2": 0.8}}
    assert index["datasets"][1]["default_cell"] == "large"
    assert "performance" not in index["datasets"][0]
