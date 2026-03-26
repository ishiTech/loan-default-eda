# Loan Default EDA

## Overview

Exploratory Data Analysis on a real-world loan default dataset
from a housing finance company. This project covers data cleaning,
missing value imputation, outlier detection, and feature transformation.

## Dataset

- Source: Kaggle - Loan Default Dataset
- Rows: 148,670
- Columns: 34

## Key Steps

1. Data overview (shape, dtypes, null counts)
2. Missing value detection and imputation
   - Numeric columns → median imputation
   - Categorical columns → mode imputation
3. Outlier detection using boxplots
4. Log transformation for skewed features

## Key Findings

- `Upfront_charges`, `rate_of_interest`, `dtir1` had 16-27% missing values
- `income`, `property_value`, `Upfront_charges` are heavily right-skewed
- Extreme values in income/property reflect real-world loan applicant diversity
- Log transformation applied to compress skewed distributions

## Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn

## How to Run

1. Clone the repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn`
3. Open `loan_default_eda.ipynb` in Jupyter
