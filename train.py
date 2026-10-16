
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier

from sklearn.base import clone

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
        "customer_state"
    ]

    selected_features = numeric_features + categorical_features
    target_col = "late_delivery"

    X_train = train_df[selected_features]
    y_train = train_df[target_col]

    X_val = val_df[selected_features]
    y_val = val_df[target_col]

    X_test = test_df[selected_features]
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
    # 10A. BASELINE COMPARISON
    # --------------------------------------------------

    from sklearn.dummy import DummyClassifier

    # Baseline 1: Predict the training-set class prevalence
    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(X_train, y_train)

    # Baseline 2: Use promised delivery days only
    promised_days_pipeline = Pipeline([
        (
            "preprocessor",
            ColumnTransformer([
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    ["promised_delivery_days"],
                ),
            ]),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    promised_days_pipeline.fit(
        train_df[["promised_delivery_days"]],
        y_train,
    )

    baseline_models = {
        "Dummy prior": dummy_model,
        "Promised days only": promised_days_pipeline,
        "Full Logistic Regression": lr_model,
    }

    print("\nBaseline comparison on validation and test")
    baseline_results = []

    for name, baseline_model in baseline_models.items():
        if name == "Dummy prior":
            val_X = X_val
            test_X = X_test
        elif name == "Promised days only":
            val_X = val_df[["promised_delivery_days"]]
            test_X = test_df[["promised_delivery_days"]]
        else:
            val_X = X_val
            test_X = X_test

        val_scores = baseline_model.predict_proba(val_X)[:, 1]
        test_scores = baseline_model.predict_proba(test_X)[:, 1]

        result = {
            "model": name,
            "validation_AP": average_precision_score(y_val, val_scores),
            "validation_ROC_AUC": roc_auc_score(y_val, val_scores),
            "test_AP": average_precision_score(y_test, test_scores),
            "test_ROC_AUC": roc_auc_score(y_test, test_scores),
        }

        baseline_results.append(result)

    baseline_results = pd.DataFrame(baseline_results)
    print(baseline_results.round(4).to_string(index=False))


    # --------------------------------------------------
    # 10c BOOTSTRAP CONFIDENCE INTERVALS
    # --------------------------------------------------

    print("\nBootstrapping test metrics (95% confidence intervals)...")

    rng = np.random.default_rng(42)
    n_bootstrap = 1000

    y_test_array = y_test.to_numpy()

    test_score_map = {
        "Dummy prior": dummy_model.predict_proba(X_test)[:, 1],
        "Promised days only": promised_days_pipeline.predict_proba(
            test_df[["promised_delivery_days"]]
        )[:, 1],
        "Full Logistic Regression": lr_model.predict_proba(X_test)[:, 1],
    }

    bootstrap_metrics = {
        name: {"ap": [], "auc": []}
        for name in test_score_map
    }

    # Paired comparisons use the same sampled test rows
    delta_ap = []
    delta_auc = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0, len(y_test_array), size=len(y_test_array)
        )
        y_sample = y_test_array[indices]

        # ROC-AUC is undefined if a bootstrap sample has one class
        if np.unique(y_sample).size < 2:
            continue

        sample_scores = {}

        for name, scores in test_score_map.items():
            sampled_scores = scores[indices]
            sample_scores[name] = sampled_scores

            bootstrap_metrics[name]["ap"].append(
                average_precision_score(y_sample, sampled_scores)
            )
            bootstrap_metrics[name]["auc"].append(
                roc_auc_score(y_sample, sampled_scores)
            )

        delta_ap.append(
            average_precision_score(
                y_sample, sample_scores["Full Logistic Regression"]
            )
            - average_precision_score(
                y_sample, sample_scores["Promised days only"]
            )
        )

        delta_auc.append(
            roc_auc_score(
                y_sample, sample_scores["Full Logistic Regression"]
            )
            - roc_auc_score(
                y_sample, sample_scores["Promised days only"]
            )
        )

    def bootstrap_ci(values):
        return np.percentile(values, [2.5, 97.5])

    print("\nTest metric estimates and bootstrap 95% CIs")

    for name, scores in test_score_map.items():
        ap_ci = bootstrap_ci(bootstrap_metrics[name]["ap"])
        auc_ci = bootstrap_ci(bootstrap_metrics[name]["auc"])

        print(
            f"{name}: "
            f"AP={average_precision_score(y_test_array, scores):.4f} "
            f"CI=[{ap_ci[0]:.4f}, {ap_ci[1]:.4f}] | "
            f"ROC-AUC={roc_auc_score(y_test_array, scores):.4f} "
            f"CI=[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]"
        )

    ap_delta_ci = bootstrap_ci(delta_ap)
    auc_delta_ci = bootstrap_ci(delta_auc)

    print("\nPaired difference: Full model minus promised-days baseline")
    print(
        f"AP difference CI: "
        f"[{ap_delta_ci[0]:.4f}, {ap_delta_ci[1]:.4f}]"
    )
    print(
        f"ROC-AUC difference CI: "
        f"[{auc_delta_ci[0]:.4f}, {auc_delta_ci[1]:.4f}]"
    )


    # --------------------------------------------------
    # 10d EXPANDING-WINDOW TEMPORAL VALIDATION
    # --------------------------------------------------



    print("\nExpanding-window temporal validation...")

    # Keep eligible orders in chronological order.
    # train_df, val_df and test_df were created chronologically.
    historical_df = (
    train_df
    .sort_values("order_purchase_timestamp")
    .reset_index(drop=True)
    )

    # Three sequential validation windows.
    # Each fold trains on earlier data and validates on the next period.
    n_rows = len(historical_df)
    initial_train_size = int(n_rows * 0.55)
    validation_window_size = int(n_rows * 0.15)

    temporal_results = []

    for fold in range(3):
        train_end = initial_train_size + fold * validation_window_size
        val_end = train_end + validation_window_size

        fold_train_df = historical_df.iloc[:train_end]
        fold_val_df = historical_df.iloc[train_end:val_end]

        X_fold_train = fold_train_df[selected_features]
        y_fold_train = fold_train_df[target_col]

        X_fold_val = fold_val_df[selected_features]
        y_fold_val = fold_val_df[target_col]

        # Skip a fold if its validation labels contain only one class.
        if y_fold_val.nunique() < 2:
            print(f"Fold {fold + 1}: skipped (only one validation class)")
            continue

        # Clone the pipeline so each fold learns preprocessing
        # and model parameters only from its own training window.
        fold_model = clone(lr_model)
        fold_model.fit(X_fold_train, y_fold_train)

        full_scores = fold_model.predict_proba(X_fold_val)[:, 1]

        # Simple baseline: promised delivery days only.
        promised_days_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        promised_days_model.fit(
            fold_train_df[["promised_delivery_days"]],
            y_fold_train
        )

        simple_scores = promised_days_model.predict_proba(
            fold_val_df[["promised_delivery_days"]]
        )[:, 1]

        temporal_results.append({
            "fold": fold + 1,
            "train_rows": len(fold_train_df),
            "validation_rows": len(fold_val_df),
            "validation_start": fold_val_df["order_purchase_timestamp"].min(),
            "validation_end": fold_val_df["order_purchase_timestamp"].max(),
            "late_rate": y_fold_val.mean(),
            "full_model_AP": average_precision_score(
                y_fold_val, full_scores
            ),
            "full_model_ROC_AUC": roc_auc_score(
                y_fold_val, full_scores
            ),
            "promised_days_AP": average_precision_score(
                y_fold_val, simple_scores
            ),
            "promised_days_ROC_AUC": roc_auc_score(
                y_fold_val, simple_scores
            ),
        })

    temporal_results_df = pd.DataFrame(temporal_results)

    if not temporal_results_df.empty:
        print("\nTemporal validation results")
        display_results = temporal_results_df.copy()

        metric_cols = [
            "late_rate",
            "full_model_AP",
            "full_model_ROC_AUC",
            "promised_days_AP",
            "promised_days_ROC_AUC",
        ]

        display_results[metric_cols] = display_results[metric_cols].round(4)

        print(display_results.to_string(index=False))

        metric_cols = [
            "full_model_AP",
            "full_model_ROC_AUC",
            "promised_days_AP",
            "promised_days_ROC_AUC",
        ]

        print("\nMean metrics across temporal folds")
        print(temporal_results_df[metric_cols].mean().round(4))
    else:
        print("No valid temporal folds were available.")




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
        "feature_columns": selected_features,
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
