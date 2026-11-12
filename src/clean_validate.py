import glob
import os
from datetime import date
import numpy as np
import pandas as pd
import sweetviz as sv


def get_latest_raw_files(raw_dir: str) -> tuple[str, str]:
    """Finds the most recent timestamped raw customer and order CSV files."""
    cust_files = sorted(glob.glob(os.path.join(raw_dir, "customers_raw_*.csv")))
    order_files = sorted(glob.glob(os.path.join(raw_dir, "orders_raw_*.csv")))
    if not cust_files or not order_files:
        raise FileNotFoundError("Raw CSV snapshots not found. Run src/ingest.py first.")
    return cust_files[-1], order_files[-1]


def inspect_dataset(df: pd.DataFrame, label: str) -> None:
    """Runs Session 5 structural, missing value, and duplicate inspection."""
    print(f"\n==================== {label} ====================")
    print(f"Shape: {df.shape}")
    print("\nMissing Values per Column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"Duplicate Rows: {df.duplicated().sum()}")


def validate_orders_schema(df: pd.DataFrame) -> dict:
    """Session 7 schema blueprint checking completeness, types, ranges, and uniqueness."""
    allowed_categories = ["Electronics", "Apparel", "Home & Kitchen", "Books"]
    allowed_statuses = ["Completed", "Returned", "Cancelled"]
    required_cols = [
        "order_id", "customer_id", "order_date", "category",
        "quantity", "unit_price", "discount_pct", "order_status", "review_text"
    ]

    checks = {
        "1. No missing required fields": df[required_cols].isnull().sum().sum() == 0,
        "2. Unit price is numeric": pd.api.types.is_numeric_dtype(df["unit_price"]),
        "3. Quantity within range (1-10)": df["quantity"].between(1, 10).all(),
        "4. Unit price within valid range (1-1000)": df["unit_price"].between(1.0, 1000.0).all(),
        "5. Discount within range (0.0-0.5)": df["discount_pct"].between(0.0, 0.5).all(),
        "6. Categories match approved list": df["category"].isin(allowed_categories).all(),
        "7. Order statuses match approved list": df["order_status"].isin(allowed_statuses).all(),
        "8. No duplicate rows": df.duplicated().sum() == 0,
        "9. Order IDs are strictly unique": df["order_id"].is_unique,
    }
    return checks


def run_cleaning_and_validation(raw_dir: str, processed_dir: str, reports_dir: str) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    cust_path, order_path = get_latest_raw_files(raw_dir)
    df_customers = pd.read_csv(cust_path)
    df_orders = pd.read_csv(order_path)

    # 1. Inspect raw orders
    inspect_dataset(df_orders, "RAW ORDERS INSPECTION")

    # 2. Run schema validation BEFORE cleaning to document caught errors
    print("\n--- Schema Validation (BEFORE Cleaning) ---")
    for check, passed in validate_orders_schema(df_orders).items():
        print(f"[{'PASS' if passed else 'FAIL'}] {check}")

    # 3. Clean data (Session 6)
    df_clean = df_orders.drop_duplicates(keep="first").copy()
    df_clean["category"] = df_clean["category"].astype(str).str.strip().str.title()
    df_clean["order_status"] = df_clean["order_status"].astype(str).str.strip().str.title()
    df_clean["discount_pct"] = df_clean["discount_pct"].fillna(df_clean["discount_pct"].median())

    # 4. Outlier Detection & Capping on 'unit_price' (Session 8)
    col = "unit_price"
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0.0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR

    iqr_outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    z_outliers = df_clean[z_scores.abs() > 3.0]

    print(f"\n--- Outlier Check ({col}) ---")
    print(f"IQR Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Flagged by IQR: {len(iqr_outliers)} | Flagged by Z-Score (|Z|>3): {len(z_outliers)}")

    # Cap extreme values using .clip() to preserve sample size
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound).round(2)

    # 5. Merge flattened API customer attributes (Session 9 transformation)
    cust_subset = df_customers[["id", "name", "address.city", "company.name"]].rename(
        columns={
            "id": "customer_id",
            "name": "customer_name",
            "address.city": "city",
            "company.name": "company",
        }
    )
    df_merged = df_clean.merge(cust_subset, on="customer_id", how="left")
    df_merged["net_revenue"] = (
        df_merged["quantity"] * df_merged["unit_price"] * (1 - df_merged["discount_pct"])
    ).round(2)

    # 6. Re-validate AFTER cleaning & capping
    print("\n--- Schema Validation (AFTER Cleaning) ---")
    all_passed = True
    for check, passed in validate_orders_schema(df_merged).items():
        print(f"[{'PASS' if passed else 'FAIL'}] {check}")
        if not passed:
            all_passed = False

    if not all_passed:
        raise ValueError("Dataset failed post-cleaning schema validation!")

    inspect_dataset(df_merged, "CLEANED & MERGED DATASET")

    # 7. Automated EDA Report with Sweetviz (Session 10)
    today_str = str(date.today())
    report_path = os.path.join(reports_dir, f"sweetviz_eda_report_{today_str}.html")
    sv_report = sv.analyze(df_merged)
    sv_report.show_html(report_path, open_browser=False)
    print(f"[SUCCESS] Automated Sweetviz report saved to: {report_path}")

    # 8. Export versioned cleaned dataset (Session 11)
    out_csv = os.path.join(processed_dir, f"ecommerce_cleaned_{today_str}.csv")
    df_merged.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Cleaned dataset exported to: {out_csv}")


if __name__ == "__main__":
    run_cleaning_and_validation(
        raw_dir="data/raw",
        processed_dir="data/processed",
        reports_dir="reports",
    )