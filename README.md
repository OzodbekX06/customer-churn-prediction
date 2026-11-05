# Customer Churn Prediction

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Overview
Customer Churn Prediction is an applied machine learning project developed to identify telecom subscribers at high risk of canceling their subscription contracts. By quantifying churn probability, telecommunications providers can enact proactive retention campaigns and optimize customer lifetime value (CLV).

## Objectives
- Perform comprehensive data hygiene and feature transformations on the Telco Churn dataset.
- Implement and benchmark both parametric (Logistic Regression) and non-parametric (K-Nearest Neighbors) classification models.
- Evaluate classification performance using threshold-independent metrics (ROC-AUC) alongside Precision, Recall, and F1-score to account for class imbalance.
- Provide a clean, modular, and testable codebase adhering to strict software engineering standards.

## Features
- **Automated Data Cleaning:** Robust type casting of total charges, median imputation for missing values, and customer identifier stripping.
- **Categorical Feature Encoding:** Dynamic one-hot dummy encoding for multi-level categorical attributes.
- **Feature Scaling:** Z-score standardization (`StandardScaler`) applied to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`).
- **Stratified Train-Test Splitting:** Preserves the underlying 73:27 class distribution between retained and churned subscribers.
- **Model Evaluation Suite:** Automated extraction of Confusion Matrices, Precision, Recall, F1-Score, and ROC-AUC metrics.

## Technologies Used
- **Python 3.10+**
- **Pandas:** Data loading, string sanitation, and DataFrame manipulation.
- **NumPy:** Vectorized calculations and array slicing.
- **Scikit-Learn:** Model architectures, preprocessing transformers, and evaluation metrics.

## Project Structure
```text
customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── telco_churn.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── results/
│   └── metrics_comparison.json
└── main.py
```

## Dataset
The project evaluates the widely referenced Telco Customer Churn dataset containing 7,043 records and 21 attributes detailing customer demographics, enrolled services (Internet, streaming, security), contract arrangements, billing methods, and churn indicators.

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/OzodbekX06/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage
Run the unit test suite:

```bash
python -m unittest discover tests
```

Execute the full training, evaluation, and reporting pipeline:

```bash
python main.py
```

The evaluation report is automatically formatted in the console and persisted to `results/metrics_comparison.json`.

## Methodology
1. **Data Preprocessing:** Missing records in `TotalCharges` are resolved via median imputation. Categorical attributes are converted using dummy variables with `drop_first=True` to mitigate multicollinearity.
2. **Standardization:** Continuous attributes are standardized to zero mean and unit variance based exclusively on training set parameters to prevent data leakage.
3. **Model Selection:**
   - **Logistic Regression:** Serves as a linear baseline utilizing log-odds formulation.
   - **K-Nearest Neighbors ($k=5$):** Evaluates non-linear distance relationships in feature space.
4. **Evaluation:** Models are tested on an unseen 20% stratified holdout sample.

## Results
The models achieved the following verified performance metrics on the holdout evaluation set:

| Metric | Logistic Regression | K-Nearest Neighbors ($k=5$) |
| :--- | :--- | :--- |
| **Accuracy** | **80.55%** | 76.30% |
| **Precision** | **65.72%** | 55.38% |
| **Recall** | **55.88%** | 55.08% |
| **F1-Score** | **60.40%** | 55.23% |
| **ROC-AUC** | **0.8421** | 0.7928 |

### Confusion Matrix (Logistic Regression)
- True Negatives (Retained correctly predicted): **926**
- False Positives (Retained falsely flagged as churn): **109**
- False Negatives (Churned missed): **165**
- True Positives (Churned correctly identified): **209**

## Screenshots
The terminal outputs cleanly formatted ASCII tabular reports, and metric artifacts are exported to structured JSON.

## Future Improvements
- Implement SMOTE (Synthetic Minority Over-sampling Technique) to elevate recall on the minority churn class.
- Incorporate gradient-boosted decision trees (XGBoost / LightGBM) for comparison.
- Implement cost-benefit threshold optimization based on business retention costs.

## Author
**Ozodbek Dalaboyev**  
Undergraduate in Artificial Intelligence, Tashkent University of Information Technologies (TUIT)  
GitHub: [@OzodbekX06](https://github.com/OzodbekX06) &bull; Email: ozodbekd06@gmail.com

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
