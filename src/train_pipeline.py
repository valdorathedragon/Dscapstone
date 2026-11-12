import glob
import json
import os
import pickle
from datetime import date
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_latest_cleaned_file(processed_dir: str) -> str:
    """Finds the most recent cleaned dataset from Block 2."""
    files = sorted(glob.glob(os.path.join(processed_dir, "ecommerce_cleaned_*.csv")))
    if not files:
        raise FileNotFoundError("Cleaned CSV not found. Run src/clean_validate.py first.")
    return files[-1]


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sessions 25, 28, 29 & 30: Date extraction, rolling features, and log transform."""
    df = df.copy()

    # Session 29: Convert text to datetime and extract date signals
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    df["order_month"] = df["order_date"].dt.month
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Session 30: Rolling window feature (3-order moving average of net_revenue)
    df["rolling_3_rev"] = df["net_revenue"].rolling(window=3, min_periods=1).mean().round(2)

    # Session 28: Log transform to tame right-skewed revenue
    df["log_net_revenue"] = np.log1p(df["net_revenue"]).round(4)

    # Binary imbalanced target: 1 if order was Returned (~20%), 0 otherwise (~80%)
    df["is_returned"] = (df["order_status"] == "Returned").astype(int)
    return df


def build_preprocessor(numeric_cols: list, categorical_cols: list, text_col: str) -> ColumnTransformer:
    """Sessions 26, 27, 31 & 33: Reusable ColumnTransformer preventing data leakage."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("txt", TfidfVectorizer(max_features=20, stop_words="english"), text_col),
        ],
        remainder="drop",
    )
    return preprocessor


def run_ml_pipeline(processed_dir: str, models_dir: str, reports_dir: str) -> None:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    today_str = str(date.today())

    # 1. Load cleaned data & engineer features
    raw_clean_path = get_latest_cleaned_file(processed_dir)
    df = pd.read_csv(raw_clean_path)
    df_feat = engineer_domain_features(df)

    numeric_cols = [
        "quantity", "unit_price", "discount_pct", "net_revenue",
        "log_net_revenue", "rolling_3_rev", "order_month", "day_of_week", "is_weekend",
    ]
    categorical_cols = ["category", "city"]
    text_col = "review_text"
    target_col = "is_returned"

    # Drop IDs, raw dates, and original 'order_status' to prevent data leakage (Session 21)
    feature_cols = numeric_cols + categorical_cols + [text_col]
    X = df_feat[feature_cols]
    y = df_feat[target_col]

    # 2. Reproducible Train/Test Split (Session 11: random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Imbalanced Target Counts (BEFORE SMOTE) ---")
    print(y_train.value_counts().to_dict())

    # 3. Export ML-Ready Transformed Dataset (Session 35 deliverable)
    preprocessor_export = build_preprocessor(numeric_cols, categorical_cols, text_col)
    X_ml_array = preprocessor_export.fit_transform(X)
    if hasattr(X_ml_array, "toarray"):
        X_ml_array = X_ml_array.toarray()
    feature_names = preprocessor_export.get_feature_names_out()
    df_ml_ready = pd.DataFrame(X_ml_array, columns=feature_names)
    df_ml_ready[target_col] = y.values

    ml_dataset_path = os.path.join(processed_dir, f"ecommerce_ml_ready_{today_str}.csv")
    df_ml_ready.to_csv(ml_dataset_path, index=False)
    print(f"\n[SUCCESS] ML-ready dataset exported to: {ml_dataset_path} | Shape: {df_ml_ready.shape}")

    # Verify SMOTE resampling counts explicitly (Session 32 checkpoint)
    X_train_enc = preprocessor_export.fit_transform(X_train)
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
    smote_check = SMOTE(random_state=42)
    _, y_train_res = smote_check.fit_resample(X_train_enc, y_train)
    print("--- Balanced Target Counts (AFTER SMOTE) ---")
    print(pd.Series(y_train_res).value_counts().to_dict())

    # 4. Experiment Tracking across 2 Models wrapped in Full Pipelines (Session 33)
    candidates = {
        "LogisticRegression_SMOTE": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest_SMOTE": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    experiment_results = []
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for model_name, classifier in candidates.items():
        full_pipeline = ImbPipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols, text_col)),
                ("smote", SMOTE(random_state=42)),
                ("model", classifier),
            ]
        )
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)

        metrics = {
            "experiment": model_name,
            "date": today_str,
            "random_state": 42,
            "train_rows_pre_smote": int(len(X_train)),
            "train_rows_post_smote": int(len(y_train_res)),
            "test_rows": int(len(X_test)),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }
        experiment_results.append(metrics)
        print(f"\n[EXPERIMENT] {model_name} -> Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']}")

        if metrics["f1_score"] >= best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_pipeline = full_pipeline

    # 5. Save Experiment Log & Export Best Versioned Pipeline (.pkl) (Session 11)
    log_path = os.path.join(reports_dir, f"experiment_log_{today_str}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": target_col,
                "best_model": best_model_name,
                "experiments": experiment_results,
            },
            f,
            indent=4,
        )
    print(f"[SUCCESS] Experiment tracking log saved to: {log_path}")

    model_path = os.path.join(models_dir, f"return_predictor_{best_model_name}_seed42_{today_str}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"[SUCCESS] Serialized pipeline exported to: {model_path}")


if __name__ == "__main__":
    run_ml_pipeline(
        processed_dir="data/processed",
        models_dir="models",
        reports_dir="reports",
    )