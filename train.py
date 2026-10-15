
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    os.getenv("OLIST_DATA_DIR", str(PROJECT_DIR / "data"))
).expanduser().resolve()

MODEL_PATH = PROJECT_DIR / "late_delivery_model.joblib"

REQUIRED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_customers_dataset.csv",
]


def main():
    # --------------------------------------------------
    # 2. VALIDATE AND LOAD DATA
    # --------------------------------------------------

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (DATA_DIR / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing CSV files in {DATA_DIR}: {missing_files}"
        )

    print(f"Loading data from: {DATA_DIR}")

    orders_raw = pd.read_csv(DATA_DIR / REQUIRED_FILES[0])
    items_raw = pd.read_csv(DATA_DIR / REQUIRED_FILES[1])
    products_raw = pd.read_csv(DATA_DIR / REQUIRED_FILES[2])
    customers_raw = pd.read_csv(DATA_DIR / REQUIRED_FILES[3])

    # --------------------------------------------------
    # 3. CLEAN DATES AND DEFINE TARGET
    # --------------------------------------------------

    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_cols:
        orders_raw[col] = pd.to_datetime(
            orders_raw[col], errors="coerce"
        )

    eligible = orders_raw[
        (orders_raw["order_status"] == "delivered")
        & orders_raw["order_delivered_customer_date"].notna()
        & orders_raw["order_purchase_timestamp"].notna()
        & orders_raw["order_estimated_delivery_date"].notna()
    ].copy()

    eligible["late_delivery"] = (
        eligible["order_delivered_customer_date"]
        > eligible["order_estimated_delivery_date"]
    ).astype(int)

    # --------------------------------------------------
    # 4. ORDER-LEVEL ITEM FEATURES
    # --------------------------------------------------

    item_features = (
        items_raw.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            seller_count=("seller_id", "nunique"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # 5. MAIN PRODUCT CATEGORY PER ORDER
    # --------------------------------------------------

    item_categories = items_raw.merge(
        products_raw[
            ["product_id", "product_category_name"]
        ],
        on="product_id",
        how="left",
    )

    item_categories["product_category_name"] = (
        item_categories["product_category_name"]
        .fillna("unknown")
    )

    category_features = (
        item_categories.groupby(
            ["order_id", "product_category_name"]
        )
        .size()
        .reset_index(name="category_item_count")
        .sort_values(
            [
                "order_id",
                "category_item_count",
                "product_category_name",
            ],
            ascending=[True, False, True],
        )
        .drop_duplicates("order_id")
        [["order_id", "product_category_name"]]
        .rename(
            columns={"product_category_name": "main_category"}
        )
    )

    # --------------------------------------------------
    # 6. CUSTOMER STATE
    # --------------------------------------------------

    customer_features = (
        customers_raw[["customer_id", "customer_state"]]
        .drop_duplicates("customer_id")
    )

    # --------------------------------------------------
    # 7. BUILD MODELLING DATASET
    # --------------------------------------------------

    model_df = eligible[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "late_delivery",
        ]
    ].copy()

    model_df = model_df.merge(
        item_features, on="order_id", how="left"
    )
    model_df = model_df.merge(
        category_features, on="order_id", how="left"
    )
    model_df = model_df.merge(
        customer_features, on="customer_id", how="left"
    )

    purchase_time = model_df["order_purchase_timestamp"]

    model_df["purchase_hour"] = purchase_time.dt.hour
    model_df["purchase_day_of_week"] = purchase_time.dt.dayofweek
    model_df["purchase_month"] = purchase_time.dt.month

    model_df["promised_delivery_days"] = (
        model_df["order_estimated_delivery_date"] - purchase_time
    ).dt.total_seconds() / (24 * 60 * 60)

    model_df["main_category"] = (
        model_df["main_category"].fillna("unknown")
    )
    model_df["customer_state"] = (
        model_df["customer_state"].fillna("unknown")
    )

    # --------------------------------------------------
    # 8. CHRONOLOGICAL 70/15/15 SPLIT
    # --------------------------------------------------

    model_df = (
        model_df.sort_values("order_purchase_timestamp")
        .reset_index(drop=True)
    )

    n = len(model_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = model_df.iloc[:train_end].copy()
    val_df = model_df.iloc[train_end:val_end].copy()
    test_df = model_df.iloc[val_end:].copy()

    print("\nDataset and split sizes")
    print(f"All eligible orders: {n:,}")

    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(part):,} rows | "
            f"late rate: {part['late_delivery'].mean() * 100:.2f}% | "
            f"{part['order_purchase_timestamp'].min()} to "
            f"{part['order_purchase_timestamp'].max()}"
        )

    # --------------------------------------------------
    # 9. FEATURES AND PREPROCESSING
    # --------------------------------------------------

    numeric_features = [
        "item_count",
        "total_price",
        "total_freight",
        "seller_count",
        "purchase_hour",
        "purchase_day_of_week",
        "purchase_month",
        "promised_delivery_days",
    ]

    categorical_features = [
        "main_category",
        "customer_state",
    ]

    feature_cols = numeric_features + categorical_features
    target_col = "late_delivery"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # --------------------------------------------------
    # 10. TRAIN ON TRAINING DATA ONLY
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")
    lr_model.fit(X_train, y_train)

    # --------------------------------------------------
    # 11. SELECT THRESHOLD ON VALIDATION DATA
    # --------------------------------------------------

    val_probabilities = lr_model.predict_proba(X_val)[:, 1]

    threshold_rows = []

    for threshold_value in np.arange(0.20, 0.801, 0.01):
        threshold_value = round(float(threshold_value), 2)

        val_predictions = (
            val_probabilities >= threshold_value
        ).astype(int)

        threshold_rows.append({
            "threshold": threshold_value,
            "precision": precision_score(
                y_val, val_predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, val_predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, val_predictions, zero_division=0
            ),
            "flagged_orders": int(val_predictions.sum()),
        })

    threshold_results = pd.DataFrame(threshold_rows)
    lr_best = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    final_threshold = float(lr_best["threshold"])

    print("\nBest validation threshold")
    print(lr_best.round(4).to_string())

    # --------------------------------------------------
    # 12. FINAL TEST EVALUATION
    # Threshold is already selected; do not tune on test.
    # --------------------------------------------------

    test_scores = lr_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_scores >= final_threshold
    ).astype(int)

    test_ap = average_precision_score(y_test, test_scores)
    test_roc_auc = roc_auc_score(y_test, test_scores)
    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )

    print("\nFinal test results")
    print(f"Threshold: {final_threshold:.2f}")
    print(f"Test late rate: {y_test.mean() * 100:.2f}%")
    print(f"Average Precision: {test_ap:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # --------------------------------------------------
    # 13. SAVE ARTIFACT IN APP-COMPATIBLE FORMAT
    # --------------------------------------------------

    artifact = {
        "model": lr_model,
        "threshold": final_threshold,
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target_col,
        "model_name": "LogisticRegression",
        "validation_best_f1": float(lr_best["f1"]),
        "validation_average_precision": float(
            average_precision_score(y_val, val_probabilities)
        ),
        "test_average_precision": float(test_ap),
        "test_roc_auc": float(test_roc_auc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel artifact saved to: {MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
