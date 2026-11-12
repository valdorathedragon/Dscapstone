import glob
import os
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


def get_latest_cleaned_file(processed_dir: str) -> str:
    """Finds the most recent timestamped cleaned dataset."""
    files = sorted(glob.glob(os.path.join(processed_dir, "ecommerce_cleaned_*.csv")))
    if not files:
        raise FileNotFoundError("Cleaned CSV not found. Run src/clean_validate.py first.")
    return files[-1]


def run_statistical_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Session 14: Univariate, Bivariate, and Multivariate tabular summaries."""
    print("\n--- 1. Univariate: Category & Order Status Counts ---")
    print(df["category"].value_counts())
    print("\n", df["order_status"].value_counts())

    print("\n--- 2. Bivariate: Pearson Correlation Matrix ---")
    num_cols = ["quantity", "unit_price", "discount_pct", "net_revenue"]
    corr_matrix = df[num_cols].corr().round(3)
    print(corr_matrix)

    print("\n--- 3. Multivariate: Revenue & Volume by Category ---")
    cat_summary = (
        df.groupby("category", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_revenue=("net_revenue", "sum"),
            avg_order_value=("net_revenue", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
        .round(2)
    )
    print(cat_summary)

    city_summary = (
        df.groupby("city", as_index=False)["net_revenue"]
        .sum()
        .sort_values("net_revenue", ascending=False)
        .head(5)
        .round(2)
    )
    return corr_matrix, cat_summary, city_summary


def save_static_eda_charts(df: pd.DataFrame, reports_dir: str, today_str: str) -> None:
    """Sessions 15-16: Static Matplotlib & Seaborn distribution and outlier visuals."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram + KDE of Net Revenue
    sns.histplot(df["net_revenue"], kde=True, bins=20, color="teal", ax=axes[0])
    axes[0].set_title("Net Revenue Skewed Right With Most Orders Under $500")
    axes[0].set_xlabel("Net Revenue ($)")
    axes[0].set_ylabel("Order Count")

    # Boxplot of Unit Price across Categories (post-IQR capping)
    sns.boxplot(x="category", y="unit_price", data=df, palette="Set2", ax=axes[1])
    axes[1].set_title("Capped Unit Prices Stay Consistent Across Categories")
    axes[1].set_xlabel("Product Category")
    axes[1].set_ylabel("Unit Price ($)")

    plt.tight_layout()
    static_path = os.path.join(reports_dir, f"static_eda_charts_{today_str}.png")
    plt.savefig(static_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Static EDA chart saved to: {static_path}")


def build_interactive_dashboard(
    df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    cat_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    reports_dir: str,
    today_str: str,
) -> None:
    """Sessions 17, 18, 20 & 23: Single-page interactive Plotly dashboard."""
    # Session 18: Sort chronologically before plotting time-series trend
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_day"] = df["order_date"].dt.date
    daily_sales = (
        df.groupby("order_day", as_index=False)["net_revenue"]
        .sum()
        .sort_values("order_day")
    )
    daily_sales["rolling_7d_avg"] = daily_sales["net_revenue"].rolling(window=7, min_periods=1).mean()

    top_cat = cat_summary.iloc[0]["category"]
    top_cat_rev = cat_summary.iloc[0]["total_revenue"]
    total_rev = df["net_revenue"].sum()
    completion_rate = (df["order_status"] == "Completed").mean() * 100

    # Top-left priority layout with finding-based chart titles (Session 20)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{top_cat} Generates Highest Revenue (${top_cat_rev:,.0f})",
            "7-Day Moving Average Smooths Daily Order Volatility",
            "Quantity & Unit Price Drive Net Revenue (r > 0.55)",
            "Top 5 Cities Account for Core Regional Revenue",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    # Chart 1 (Top-Left Primary Metric): Total Revenue by Category
    fig.add_trace(
        go.Bar(
            x=cat_summary["category"],
            y=cat_summary["total_revenue"],
            text=cat_summary["total_revenue"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside",
            marker_color="#1f77b4",
            name="Category Revenue",
        ),
        row=1,
        col=1,
    )

    # Chart 2 (Top-Right): Time-Series Daily Revenue + 7-Day Moving Average
    fig.add_trace(
        go.Scatter(
            x=daily_sales["order_day"],
            y=daily_sales["net_revenue"],
            mode="lines",
            line=dict(color="lightgray", width=1.5),
            name="Daily Revenue",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=daily_sales["order_day"],
            y=daily_sales["rolling_7d_avg"],
            mode="lines",
            line=dict(color="#d62728", width=2.5),
            name="7-Day Rolling Avg",
        ),
        row=1,
        col=2,
    )

    # Chart 3 (Bottom-Left): Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Blues",
            text=corr_matrix.values,
            texttemplate="%{text}",
            showscale=False,
            name="Correlation",
        ),
        row=2,
        col=1,
    )

    # Chart 4 (Bottom-Right): Top 5 Cities by Net Revenue
    fig.add_trace(
        go.Bar(
            x=city_summary["city"],
            y=city_summary["net_revenue"],
            marker_color="#2ca02c",
            name="City Revenue",
        ),
        row=2,
        col=2,
    )

    # Enforce zero baseline on bar charts (Ethical Visualization - Session 21)
    fig.update_yaxes(rangemode="tozero", title_text="Revenue ($)", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", title_text="Revenue ($)", row=1, col=2)
    fig.update_yaxes(rangemode="tozero", title_text="Revenue ($)", row=2, col=2)

    fig.update_layout(
        title_text=(
            f"<b>Executive E-Commerce Analytics Dashboard</b> | "
            f"Total Net Revenue: <b>${total_rev:,.2f}</b> | "
            f"Completed Orders: <b>{completion_rate:.1f}%</b>"
        ),
        height=850,
        template="plotly_white",
        showlegend=True,
    )

    dash_path = os.path.join(reports_dir, f"interactive_dashboard_{today_str}.html")
    fig.write_html(dash_path)
    print(f"[SUCCESS] Interactive Plotly dashboard saved to: {dash_path}")


if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    REPORTS_DIR = "reports"
    TODAY = str(date.today())

    latest_csv = get_latest_cleaned_file(PROCESSED_DIR)
    cleaned_df = pd.read_csv(latest_csv)

    corr_df, cat_df, city_df = run_statistical_eda(cleaned_df)
    save_static_eda_charts(cleaned_df, REPORTS_DIR, TODAY)
    build_interactive_dashboard(cleaned_df, corr_df, cat_df, city_df, REPORTS_DIR, TODAY)