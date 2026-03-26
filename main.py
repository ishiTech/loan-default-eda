# =========================================
# IMPORTS
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("Loan_Default.csv")

# =========================================
# UNDERSTANDING THE DATA
# =========================================

# 1. Check first five rows of your data
head = df.head()
print(f"First five rows of your df: \n{head}")

# 2. Size of you data
shape_df = df.shape
print(f"\nShape of dataset: {(shape_df)}")

# 3. Columns 
cols = df.columns
print(f"\nColumns: {(cols)}")

# Data Type of each column
print(f"\nDatatype of each column :\n{(df.dtypes)}")

# Number of Null values in each column
print(f"\nNumber of nulls per column : \n{df.isnull().sum()}")

# Percentage of column that has missing values
null_pct = (df.isnull().sum() / len(df)) * 100
print(f"\nPercentage of column that has missing values: \n{null_pct}")

# Number of unique values in each column
print(f"\nNumber of unique per column : \n{df.nunique()}")

# Stats of numeric columns
stats = df.describe()
print(f"\nStats : \n {stats}")

# =========================================
# IDENTIFY MISSING VALUE COLUMNS
# =========================================
cols_to_impute = df.isnull().sum()
cols_to_impute = cols_to_impute[cols_to_impute > 0].index.to_list()
print(f"\n\nColumns with missing values: \n{cols_to_impute}")

num_cols = df.select_dtypes(include='number').columns.to_list()
cat_cols = df.select_dtypes(exclude='number').columns.to_list()

nums_to_impute = list(set(cols_to_impute) & set(num_cols))
print(f"\nNumeric columns with missing values : \n{nums_to_impute}")

cat_to_impute = list(set(cols_to_impute) & set(cat_cols))
print(f"\nCategorical columns with missing values : \n{cat_to_impute}")

# =========================================
# HANDLE MISSING VALUES
# =========================================
for col in nums_to_impute:
    df.fillna({col: df[col].median()}, inplace=True)

for col in cat_to_impute:
    df.fillna({col: df[col].mode()[0]}, inplace=True)

# =========================================
# OUTLIER DETECTION (BOXPLOTS)
# =========================================

# Income
sns.boxplot(x=df['income'])
plt.title('Income Distribution')
plt.show()

# Upfront Charges
sns.boxplot(x=df['Upfront_charges'])
plt.title('Upfront Charge Distribution')
plt.show()

# Property Value
sns.boxplot(x=df['property_value'])
plt.title('Property Value Distribution')
plt.show()

# =========================================
# FEATURE TRANSFORMATION
# =========================================

# Log Transformation 

df['income_log'] = np.log1p(df['income'])
df['property_value_log'] = np.log1p(df['property_value'])
df['Upfront_charges_log'] = np.log1p(df['Upfront_charges'])

# =========================================
# BOXPLOTS AFTER TRANSFORMATION
# =========================================

print("\nBoxplots After Log Transformation")

# Income Log
sns.boxplot(x=df['income_log'])
plt.title('Income (Log Transformed)')
plt.show()

# Property Value Log
sns.boxplot(x=df['property_value_log'])
plt.title('Property Value (Log Transformed)')
plt.show()

# Upfront Charges Log
sns.boxplot(x=df['Upfront_charges_log'])
plt.title('Upfront Charges (Log Transformed)')
plt.show()