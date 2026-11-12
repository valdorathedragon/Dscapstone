import json
import os
from datetime import date
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine


def fetch_api_customers(api_url: str) -> pd.DataFrame:
    """Fetches user records from a REST API and flattens nested JSON."""
    response = requests.get(api_url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"API request failed with status code: {response.status_code}")

    print(f"[CHECKPOINT] API Status Code: {response.status_code}")
    raw_json = response.json()
    df_customers = pd.json_normalize(raw_json)
    print(f"[CHECKPOINT] Flattened API Customers Shape: {df_customers.shape}")
    return df_customers


def load_sql_orders(db_path: str, customer_ids: list) -> pd.DataFrame:
    """Creates and queries an e-commerce orders table in SQLite via SQLAlchemy."""
    engine = create_engine(f"sqlite:///{db_path}")
    rng = np.random.default_rng(42)

    n_rows = 500
    categories = ["Electronics", "electronics ", "Apparel", "Home & Kitchen", "Books"]
    statuses = ["Completed", "Completed", "Completed", "Returned", "Cancelled"]
    reviews = [
        "Great product and fast delivery",
        "Item arrived damaged, poor packaging",
        "Works as expected, good value",
        "Terrible quality, requesting refund",
        "Exceeded expectations, highly recommend",
    ]

    orders_data = pd.DataFrame({
        "order_id": range(1001, 1001 + n_rows),
        "customer_id": rng.choice(customer_ids, size=n_rows),
        "order_date": pd.date_range("2026-01-01", periods=n_rows, freq="6h").astype(str),
        "category": rng.choice(categories, size=n_rows),
        "quantity": rng.integers(1, 6, size=n_rows),
        "unit_price": np.round(rng.uniform(15.0, 450.0, size=n_rows), 2),
        "discount_pct": rng.choice([0.0, 0.05, 0.10, 0.20, np.nan], size=n_rows),
        "order_status": rng.choice(statuses, size=n_rows),
        "review_text": rng.choice(reviews, size=n_rows),
    })

    # Inject realistic anomalies (duplicates & extreme outlier) to test Block 2 validation
    orders_data.loc[10, "unit_price"] = 9999.99
    orders_data = pd.concat([orders_data, orders_data.iloc[:5]], ignore_index=True)

    # Write to SQLite and read back via SQL query
    orders_data.to_sql("orders", engine, if_exists="replace", index=False)
    df_orders = pd.read_sql_query("SELECT * FROM orders", engine)
    print(f"[CHECKPOINT] SQL Orders Shape: {df_orders.shape}")
    return df_orders


def save_raw_snapshot(df_customers: pd.DataFrame, df_orders: pd.DataFrame, raw_dir: str) -> None:
    """Saves immutable timestamped raw datasets and reproducibility metadata."""
    os.makedirs(raw_dir, exist_ok=True)
    today_str = str(date.today())

    customers_file = os.path.join(raw_dir, f"customers_raw_{today_str}.csv")
    orders_file = os.path.join(raw_dir, f"orders_raw_{today_str}.csv")
    metadata_file = os.path.join(raw_dir, f"metadata_{today_str}.json")

    df_customers.to_csv(customers_file, index=False)
    df_orders.to_csv(orders_file, index=False)

    metadata = {
        "project": "AI-Ready E-Commerce Analytics Pipeline",
        "date_collected": today_str,
        "version": "v1_raw",
        "sources": {
            "customers": "https://jsonplaceholder.typicode.com/users (REST API)",
            "orders": "sqlite:///data/raw/ecommerce.db (SQL Table: orders)",
        },
        "customers_shape": list(df_customers.shape),
        "orders_shape": list(df_orders.shape),
        "notes": "Raw immutable snapshot prior to deduplication, imputation, and validation.",
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Saved raw files & metadata to '{raw_dir}/'")


if __name__ == "__main__":
    API_URL = "https://jsonplaceholder.typicode.com/users"
    RAW_DIR = "data/raw"
    DB_PATH = os.path.join(RAW_DIR, "ecommerce.db")

    customers_df = fetch_api_customers(API_URL)
    orders_df = load_sql_orders(DB_PATH, customers_df["id"].tolist())
    save_raw_snapshot(customers_df, orders_df, RAW_DIR)