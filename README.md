# Olist E-commerce Intelligence & Late-Delivery Risk Prediction

An end-to-end Data Science project that analyzes e-commerce delivery performance, identifies patterns associated with late deliveries, and demonstrates an ML-based late-delivery risk scoring application using Streamlit.

## Project Overview

Late deliveries can negatively affect customer experience and satisfaction. This project uses the public Olist Brazilian e-commerce dataset to investigate historical delivery performance and explore whether order-time information can help identify potentially late deliveries.

The project combines SQL analytics, exploratory data analysis, statistical testing, machine learning, and an interactive dashboard.

## Key Features

- Delivery KPIs and monthly performance trends
- SQL-based analysis using SQLite, including CTEs and window functions
- Product-category delivery analysis
- Statistical comparison of customer review scores for late versus on-time deliveries
- Logistic Regression and Random Forest model comparison
- Chronological train, validation, and test split
- Threshold analysis using precision, recall, and F1-score
- Interactive Streamlit dashboard with a late-delivery risk predictor

## Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Database:** SQLite
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Statistics:** SciPy
- **Machine learning:** Scikit-learn
- **Application:** Streamlit
- **Model persistence:** Joblib
- **Environment:** Jupyter Notebook, VS Code

## Dataset

This project uses the public Olist Brazilian e-commerce dataset.

The model is trained on delivered orders with known actual and estimated delivery dates. Therefore, its target is whether an eligible, eventually delivered order arrived later than its estimated delivery date.

This is a retrospective prediction exercise; the dataset does not represent every order that could be at risk at the moment of purchase.

## Machine Learning Results

The models were evaluated using a chronological split to better reflect prediction on future orders.

### Logistic Regression

- Validation Average Precision: **0.1320**
- Validation ROC-AUC: **0.7456**
- Validation F1-score at the selected threshold: **0.2278**
- Test Average Precision: **0.1122**
- Test ROC-AUC: **0.6583**
- Test precision at the selected threshold: **0.0917**
- Test recall at the selected threshold: **0.2560**
- Test F1-score at the selected threshold: **0.1351**

### Random Forest

- Validation Average Precision: **0.1363**
- Validation ROC-AUC: **0.7056**
- Validation F1-score at the best tested threshold: **0.2073**

Logistic Regression was selected provisionally because it achieved better validation ROC-AUC and F1-score. Its test results show that performance declined on later data, so the predictor should be considered experimental rather than production-ready.

## Statistical Analysis

Review scores were compared between late and on-time/early delivered orders at the order level.

- Mean review score for on-time/early orders: **4.294**
- Mean review score for late orders: **2.567**
- Mean difference (late minus on-time): **-1.728**
- 95% bootstrap confidence interval: **[-1.767, -1.690]**
- Cliff's delta: **-0.554**

These results indicate a strong association between late delivery and lower review scores. They do not establish causation. Review scores were excluded from model features because they are not available at order placement.

## Project Structure

```text
Olist Project/
├── app.py
├── late_delivery_model.joblib
├── requirements.txt
├── README.md
└── Olist_Analysis.ipynb
```

The notebook filename above is illustrative; replace it with the actual filename in your project folder.

## Setup and Run

1. Install Python 3.11 or a compatible Python version.
2. Clone or download the project.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Place the required Olist CSV files in the configured dataset directory, or update the dataset path in `app.py`.
5. Ensure `late_delivery_model.joblib` is in the same folder as `app.py`.
6. Launch the application:

   ```bash
   python -m streamlit run app.py
   ```

## Limitations

- The model is trained only on eligible orders that were eventually delivered.
- Test performance is modest and shows a decline from validation performance.
- The risk score is not calibrated as a real-world probability.
- Historical relationships do not establish causal explanations for late delivery.
- The current application is a learning and analytical prototype, not a production delivery-alert system.

## Future Improvements

- Improve temporal generalization and investigate distribution shift.
- Evaluate probability calibration and business-specific alert thresholds.
- Add seller and geographic insights while preventing data leakage.
- Improve automated data-path configuration and application testing.
- Deploy the dashboard and document reproducible experiments.

## Data Source

The dataset is publicly available through the Olist Brazilian e-commerce dataset on Kaggle. This project is an independent analysis and is not affiliated with Olist.