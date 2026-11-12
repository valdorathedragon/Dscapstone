
# AI-Ready E-Commerce Data Engineering & Visualization Pipeline

End-to-end data acquisition, validation, exploratory visualization, and reusable feature engineering pipeline built for **AML001: Data Handling, Visualization & Feature Engineering**.

## Problem Statement

E-commerce order logs and customer API records contain missing discounts, duplicate transactions, unstandardized category labels, extreme pricing outliers, and imbalanced return outcomes (~20% returned orders). This platform transforms multi-source raw records into validated analytics dashboards and leak-free, ML-ready datasets.

## Repository Structure

```text
Dscapstone/
├── data/
│   ├── raw/                  # Immutable API + SQLite snapshots & metadata JSON
│   └── processed/            # Cleaned & ML-ready versioned CSV exports
├── models/                   # Serialized Scikit-learn + SMOTE pipelines (.pkl)
├── reports/                  # Sweetviz HTML profile, static PNGs, Plotly dashboard & experiment JSON
├── src/
│   ├── ingest.py             # REST API (requests) + SQLite (SQLAlchemy) ingestion
│   ├── clean_validate.py     # Deduplication, imputation, schema checks, IQR/Z-score capping & Sweetviz
│   ├── eda_dashboard.py      # Statistical EDA, Seaborn visuals & interactive Plotly dashboard
│   └── train_pipeline.py     # Feature engineering, ColumnTransformer, SMOTE & experiment tracking
├── requirements.txt          # Frozen Python dependencies
└── README.md
```


## Setup & Execution Steps

1. **Clone & Activate Virtual Environment:**

git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/Dscapstone.git
cd Dscapstone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. **Run the Pipeline Sequentially:**

python src/ingest.py  # Step 1: Fetch REST API + SQLite orders & save raw snapshots
python src/clean_validate.py  # Step 2: Clean, validate schema, cap outliers & build Sweetviz report
python src/eda_dashboard.py  # Step 3: Run statistical EDA & export interactive Plotly dashboard
python src/train_pipeline.py  # Step 4: Engineer features, run SMOTE experiments & export .pkl model


**Key Outputs Generated**
Automated EDA Report: reports/sweetviz_eda_report_<date></date>.html

Interactive Dashboard: reports/interactive_dashboard_<date></date>.html

ML-Ready Dataset: data/processed/ecommerce_ml_ready_<date></date>.csv

Experiment Log & Model: reports/experiment_log_<date></date>.json & models/return_predictor_<model></model>_seed42_<date></date>.pkl
