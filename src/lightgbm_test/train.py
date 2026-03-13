from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def build_dummy_table(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=1.0,
        random_state=random_state,
    )

    columns = [f"feature_{i:02d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dummy_table()
    dataset_path = DATA_DIR / "dummy_tabular_data.csv"
    df.to_csv(dataset_path, index=False)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, pred_proba)), 4),
    }

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metrics_path = OUTPUT_DIR / "metrics.json"
    importance_path = OUTPUT_DIR / "feature_importance.csv"
    report_path = OUTPUT_DIR / "classification_report.txt"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_df.to_csv(importance_path, index=False)
    report_path.write_text(classification_report(y_test, pred), encoding="utf-8")

    print("Saved dataset:", dataset_path)
    print("Saved metrics:", metrics_path)
    print("Saved feature importance:", importance_path)
    print("Saved classification report:", report_path)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
