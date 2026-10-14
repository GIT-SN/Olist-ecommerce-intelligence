
from pathlib import Path
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="Olist Delivery Intelligence",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Olist E-commerce Intelligence")
st.caption("Delivery Performance & Late-Delivery Risk Analysis")

# -----------------------------
# 2. Dataset paths
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = Path(r"C:\Users\KISHAN PANWAR\Downloads\archive")
PROJECT_DATA_DIR = PROJECT_DIR / "data"

if (PROJECT_DATA_DIR / "olist_orders_dataset.csv").exists():
    DEFAULT_DATA_DIR = PROJECT_DATA_DIR
else:
    DEFAULT_DATA_DIR = LOCAL_DATA_DIR

DATA_DIR = Path(__file__).resolve().parent / "data"

ORDERS_FILE = DATA_DIR / "olist_orders_dataset.csv"
ITEMS_FILE = DATA_DIR / "olist_order_items_dataset.csv"
PRODUCTS_FILE = DATA_DIR / "olist_products_dataset.csv"

# -----------------------------
# 3. Load data
# -----------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv(ORDERS_FILE)
    items = pd.read_csv(ITEMS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    eligible = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_purchase_timestamp"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    eligible["late_delivery"] = (
        eligible["order_delivered_customer_date"]
        > eligible["order_estimated_delivery_date"]
    ).astype(int)

    monthly = eligible.copy()
    monthly["month"] = (
        monthly["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )

    monthly_summary = (
        monthly.groupby("month")
        .agg(
            total_orders=("order_id", "nunique"),
            late_orders=("late_delivery", "sum"),
            late_rate=("late_delivery", "mean")
        )
        .reset_index()
    )

    order_category = (
        items[["order_id", "product_id"]]
        .merge(
            products[["product_id", "product_category_name"]],
            on="product_id",
            how="left"
        )
        .drop_duplicates(["order_id", "product_category_name"])
    )

    category_summary = (
        eligible[["order_id", "late_delivery"]]
        .merge(order_category, on="order_id", how="inner")
    )

    category_summary["product_category_name"] = (
        category_summary["product_category_name"].fillna("unknown")
    )

    category_summary = (
        category_summary.groupby("product_category_name")
        .agg(
            eligible_orders=("order_id", "nunique"),
            late_orders=("late_delivery", "sum"),
            late_rate=("late_delivery", "mean")
        )
        .reset_index()
    )

    category_summary = category_summary[
        category_summary["eligible_orders"] >= 100
    ].sort_values("late_rate", ascending=False)

    return eligible, monthly_summary, category_summary


# -----------------------------
# 4. Load with error handling
# -----------------------------
try:
    eligible, monthly_summary, category_summary = load_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.info(
        "Check that the CSV files exist in your Downloads/archive folder."
    )
    st.stop()

# -----------------------------
# 5. Sidebar filters
# -----------------------------

# -----------------------------
# 5. Sidebar filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

months = sorted(monthly_summary["month"].unique())

if months:
    date_range = st.sidebar.select_slider(
        "Purchase month range",
        options=months,
        value=(months[0], months[-1])
    )

    selected_months = [
        month for month in months
        if date_range[0] <= month <= date_range[1]
    ]
else:
    selected_months = []

filtered = eligible.copy()

filtered["month"] = (
    filtered["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

filtered = filtered[
    filtered["month"].isin(selected_months)
]


# -----------------------------
# 6. KPI cards
# -----------------------------
total_orders = filtered["order_id"].nunique()
late_orders = int(filtered["late_delivery"].sum())

late_rate = (
    filtered["late_delivery"].mean() * 100
    if total_orders > 0 else 0
)

on_time_orders = total_orders - late_orders

c1, c2, c3, c4 = st.columns(4)

c1.metric("Eligible Delivered Orders", f"{total_orders:,}")
c2.metric("Late Orders", f"{late_orders:,}")
c3.metric("On-time / Early Orders", f"{on_time_orders:,}")
c4.metric("Late Delivery Rate", f"{late_rate:.2f}%")

st.divider()




# -----------------------------
# 7. Monthly delivery trends
# -----------------------------
st.subheader("Monthly Delivery Performance")

if not filtered.empty:
    monthly_filtered = (
        filtered.groupby("month")
        .agg(
            total_orders=("order_id", "nunique"),
            late_orders=("late_delivery", "sum"),
            late_rate=("late_delivery", "mean")
        )
        .reset_index()
    )

    monthly_filtered["late_rate_pct"] = (
        monthly_filtered["late_rate"] * 100
    )

    # Calculate a rolling trend weighted by order counts
    monthly_filtered["rolling_3m_late_rate_pct"] = (
        monthly_filtered["late_orders"].rolling(3).sum()
        / monthly_filtered["total_orders"].rolling(3).sum()
        * 100
    )

    fig = px.line(
        monthly_filtered,
        x="month",
        y="late_rate_pct",
        markers=True,
        custom_data=["total_orders", "late_orders"],
        title="Monthly Late-Delivery Rate",
        labels={
            "month": "Purchase Month",
            "late_rate_pct": "Late-Delivery Rate (%)"
        }
    )

    fig.update_traces(
        hovertemplate=(
            "Month: %{x}<br>"
            "Late rate: %{y:.2f}%<br>"
            "Eligible orders: %{customdata[0]:,.0f}<br>"
            "Late orders: %{customdata[1]:,.0f}"
            "<extra></extra>"
        )
    )

    fig.add_scatter(
        x=monthly_filtered["month"],
        y=monthly_filtered["rolling_3m_late_rate_pct"],
        mode="lines",
        name="Rolling 3-month rate",
        line={"dash": "dash"},
        connectgaps=False
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    fig_volume = px.bar(
        monthly_filtered,
        x="month",
        y="total_orders",
        title="Eligible Orders by Purchase Month",
        labels={
            "month": "Purchase Month",
            "total_orders": "Number of Orders"
        }
    )

    st.plotly_chart(fig_volume, use_container_width=True)
else:
    st.warning("Select at least one month to display the charts.")





# -----------------------------
# 8. Category-level insights
# -----------------------------
st.subheader("Product Categories with Higher Late-Delivery Rates")

if not category_summary.empty:
    top_n = st.slider(
        "Number of categories to display",
        min_value=5,
        max_value=min(20, len(category_summary)),
        value=min(10, len(category_summary))
    )

    top_categories = category_summary.head(top_n).copy()
    top_categories["late_rate_pct"] = top_categories["late_rate"] * 100

    fig_category = px.bar(
        top_categories.sort_values("late_rate_pct"),
        x="late_rate_pct",
        y="product_category_name",
        orientation="h",
        hover_data=["eligible_orders", "late_orders"],
        title="Highest Late-Delivery Rates (Minimum 100 Orders)",
        labels={
            "late_rate_pct": "Late-Delivery Rate (%)",
            "product_category_name": "Product Category",
            "eligible_orders": "Eligible Orders",
            "late_orders": "Late Orders"
        }
    )

    st.plotly_chart(fig_category, use_container_width=True)

    st.dataframe(
        top_categories[
            [
                "product_category_name",
                "eligible_orders",
                "late_orders",
                "late_rate_pct"
            ]
        ].rename(columns={
            "product_category_name": "Category",
            "eligible_orders": "Eligible Orders",
            "late_orders": "Late Orders",
            "late_rate_pct": "Late Rate (%)"
        }).round(2),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Dataset: public Olist Brazilian e-commerce dataset. "
    "Late-delivery analysis includes delivered orders with known actual "
    "and estimated delivery dates. Category counts can overlap when an "
    "order contains products from multiple categories."
)



# -----------------------------
# 9. Late-Delivery Risk Predictor
# -----------------------------
import joblib

MODEL_FILE = (
    Path(__file__).resolve().parent / "late_delivery_model.joblib"
)

st.divider()
st.header("Late-Delivery Risk Predictor")

st.warning(
    "Experimental model: test Average Precision was 0.1122 "
    "and recall was 25.6% at the selected threshold. "
    "Use this score for demonstration, not operational decisions."
)

if not MODEL_FILE.exists():
    st.error(f"Model file not found: {MODEL_FILE}")
else:
    @st.cache_resource
    def load_model_artifact(path):
        return joblib.load(path)

    artifact = load_model_artifact(str(MODEL_FILE))
    predictor = artifact["model"]
    encoder = (
        artifact["model"]
        .named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    category_options = sorted(encoder.categories_[0].tolist())
    state_options = sorted(encoder.categories_[1].tolist())



    threshold = artifact["threshold"]

    st.write(
        "Enter order details available at purchase time "
        "to estimate late-delivery risk."
    )

    with st.form("risk_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            item_count = st.number_input(
                "Number of items",
                min_value=1,
                max_value=50,
                value=1
            )

            total_price = st.number_input(
                "Total item price",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            total_freight = st.number_input(
                "Total freight cost",
                min_value=0.0,
                value=20.0,
                step=5.0
            )

            seller_count = st.number_input(
                "Number of sellers",
                min_value=1,
                max_value=50,
                value=1
            )

            promised_days = st.number_input(
                "Promised delivery time (days)",
                min_value=0.0,
                max_value=365.0,
                value=15.0,
                step=1.0
            )

        with col2:
            purchase_hour = st.slider(
                "Purchase hour",
                min_value=0,
                max_value=23,
                value=14
            )

            purchase_day = st.selectbox(
                "Day of week (Monday = 0)",
                options=list(range(7)),
                index=2
            )

            purchase_month = st.selectbox(
                "Purchase month",
                options=list(range(1, 13)),
                index=5
            )
            
            main_category = st.selectbox(
                "Main product category",
                options=category_options,
                index=(
                    category_options.index("unknown")
                    if "unknown" in category_options else 0
                )
            )

            
            customer_state = st.selectbox(
                "Customer state",
                options=state_options,
                index=(
                    state_options.index("SP")
                    if "SP" in state_options else 0
                )
            )



            

            

        submitted = st.form_submit_button("Estimate Risk")

    if submitted:
        input_data = pd.DataFrame([{
            "item_count": item_count,
            "total_price": total_price,
            "total_freight": total_freight,
            "seller_count": seller_count,
            "purchase_hour": purchase_hour,
            "purchase_day_of_week": purchase_day,
            "purchase_month": purchase_month,
            "promised_delivery_days": promised_days,
            "main_category": main_category,
            "customer_state": customer_state
        }])

        risk_score = predictor.predict_proba(input_data)[0, 1]
        predicted_late = risk_score >= threshold

        st.subheader("Prediction Result")
        st.metric("Model risk score", f"{risk_score:.1%}")
        st.caption(
            "This is an uncalibrated model score, not a verified "
            "real-world probability."
        )

        if predicted_late:
            st.error(
                f"Flagged as higher risk at threshold {threshold:.2f}."
            )
        else:
            st.success(
                f"Not flagged at threshold {threshold:.2f}. "
                "This does not guarantee on-time delivery."
            )
