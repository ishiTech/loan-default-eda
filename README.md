# Loan Default EDA & Prediction

## Overview

Exploratory Data Analysis and Logistic Regression model on a real-world loan default
dataset from a housing finance company. This project covers data cleaning, missing value
imputation, outlier detection, feature transformation, and binary classification with
iterative model optimization.

## Dataset

- Source: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
- Rows: 148,670
- Columns: 34
- Target: `Status` (1 = defaulted, 0 = did not default)

## Key Steps

1. **Data overview** — shape, dtypes, null counts, unique values
2. **Missing value detection and imputation**
   - Numeric columns → median imputation (robust to outliers)
   - Categorical columns → mode imputation
3. **Outlier detection** using boxplots
4. **Log transformation** for skewed features
5. **Feature selection** — dropped ID, year, low-signal columns
6. **One hot encoding** for categorical columns
7. **Feature scaling** using StandardScaler (prevents data leakage)
8. **Logistic Regression** — baseline + two optimization techniques

## Key Findings

### EDA

- `Upfront_charges`, `rate_of_interest`, `dtir1` had 16–27% missing values
- `income`, `property_value`, `Upfront_charges` are heavily right-skewed
- Outliers up to ₹580,000 (income) and ₹1.6cr (property_value) are real observations — high-value applicants exist in loan data
- Log transformation compresses skewed distributions without losing any data points

### Model Performance

| Model                        | Accuracy | Recall (Defaulters) | Precision (Defaulters) |
| ---------------------------- | -------- | ------------------- | ---------------------- |
| Baseline Logistic Regression | 0.87     | 0.51                | 0.86                   |
| + class_weight='balanced'    | 0.83     | 0.66                | 0.64                   |
| + threshold=0.3              | 0.66     | 0.87                | 0.41                   |

### Model Insights

- **Class imbalance** (3:1 ratio) caused baseline model to ignore minority class (defaulters)
- `class_weight='balanced'` improved recall by 15% with a single parameter change
- Lowering classification threshold to 0.3 pushed recall to 87% but hurt precision
- **Key tradeoff**: In banking, missing a defaulter (false negative) is more costly than rejecting a good customer (false positive) — recall for class 1 matters more than overall accuracy
- Optimal threshold depends on bank's **risk appetite**: conservative banks prefer lower thresholds

## Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/ishiTech/loan-default-eda.git
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Download the dataset from Kaggle and place `Loan_Default.csv` in the project folder
4. Run the script
   ```bash
   python loan_default.py
   ```
