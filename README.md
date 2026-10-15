# Olist E-commerce Intelligence & Late Delivery Risk Prediction

An end-to-end Data Science project that analyzes e-commerce order and delivery data, identifies delivery trends, and predicts potential late-delivery risks using Machine Learning.

## Live Dashboard

**[Open the Streamlit Dashboard](https://olist-ecommerce-intelligence-nh7dwm34qmmnsjue5h2cek.streamlit.app/)**

## Project Objectives

- Analyze e-commerce order and delivery performance.
- Identify trends in late deliveries over time.
- Explore order, product, seller, and customer-related patterns.
- Build a machine learning model to identify orders at risk of late delivery.
- Present business insights through an interactive dashboard.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Logistic Regression
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib

## Machine Learning

The project uses a Logistic Regression classification pipeline with numerical preprocessing, categorical encoding, and a decision threshold selected using validation data.

### Test Results

| Metric | Score |
|---|---:|
| Average Precision | 0.1122 |
| ROC-AUC | 0.6583 |
| Precision | 0.0917 |
| Recall | 0.2560 |
| F1-score | 0.1351 |
| Decision Threshold | 0.67 |

The test set contains approximately **6.61% late deliveries**. The model is experimental: its precision is limited, so its predictions should be treated as risk signals for further investigation rather than definitive delivery forecasts.

## Project Structure

```text
Olist Project/
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   └── olist_customers_dataset.csv
├── app.py
├── train.py
├── Olist_Analysis.ipynb
├── insight.ipynb
├── late_delivery_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/GIT-SN/Olist-ecommerce-intelligence.git
cd Olist-ecommerce-intelligence
```

### 2. Create and activate a virtual environment

**Windows — Git Bash:**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the four required CSV files inside the `data/` directory. Obtain the dataset from its original source and ensure the filenames match those expected by `train.py`.

### 5. Run the dashboard

```bash
streamlit run app.py
```

### 6. Retrain the model

```bash
python train.py
```

The training script rebuilds the dataset features, uses chronological train/validation/test splits, evaluates the Logistic Regression model, and saves the trained artifact as `late_delivery_model.joblib`.

## Reproducibility

- Dependency versions are pinned in `requirements.txt`.
- `train.py` provides a repeatable model-training workflow.
- The data directory defaults to `data/`.
- To use a different dataset directory, set the `OLIST_DATA_DIR` environment variable to that directory before running the training script.

## Limitations

- Model performance is modest, particularly precision.
- Historical relationships may not generalize to future orders.
- Predictions depend on the available features and the quality of the input data.
- The dashboard and model should be considered an analytical project, not a production delivery guarantee.

## Future Improvements

- Engineer additional seller, product, and shipping features.
- Investigate performance changes across different time periods.
- Improve precision and risk ranking through further validation.
- Add SQL-based business analysis.
- Expand monitoring and model evaluation.

---

**Disclaimer:** This is an independent portfolio project for educational and analytical purposes.