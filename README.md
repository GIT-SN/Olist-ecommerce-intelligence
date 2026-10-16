# Olist E-commerce Intelligence & Late Delivery Risk Prediction

An end-to-end Data Science project that analyzes e-commerce order and delivery data, identifies delivery trends, and predicts potential late-delivery risks using Machine Learning.

## Live Dashboard

**[Open the Streamlit Dashboard](https://olist-ecommerce-intelligence-bfpfawg8pgqylnswgbw69w.streamlit.app/)**

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
| Average Precision (AP) | 0.1152 |
| ROC-AUC | 0.6697 |
| Precision | 0.0886 |
| Recall | 0.2320 |
| F1-score | 0.1282 |
| Decision Threshold | 0.67 |

The chronological test set contains **14,471 orders**, with approximately **6.61% late deliveries**.

The model is experimental. At the selected threshold, it identifies some late deliveries but has limited precision, meaning many flagged orders are not actually late.

**Generalization note:** A promised-delivery-days-only baseline outperformed the full model on the final test set, achieving an Average Precision of 0.2680 and ROC-AUC of 0.7845, compared with 0.1152 and 0.6697 for the full model. This suggests that the model's performance may vary across time periods. The cause has not yet been established, so the results should be interpreted cautiously.

Predictions should be treated as risk signals for further investigation, not definitive delivery forecasts.



## Project Structure

```text
Olist-ecommerce-intelligence/
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   └── olist_products_dataset.csv
├── app.py
├── train.py
├── Olist_Analysis.ipynb
├── insight.ipynb
├── late_delivery_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

The required dataset files must be placed in the `data/` directory locally before running the project. Files excluded from version control will not appear in the cloned repository.



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
