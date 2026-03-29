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

# =========================================
# MODEL TRAINING (LOGISTIC REGRESSION)
# =========================================

print("="*60)
print("MODEL TRAINING")
print("="*60)

# FEATURE SELECTION
df = df.drop(
    columns =[
        "ID", "year", "income_log", "property_value_log", "Upfront_charges_log",
        "Gender", "Interest_rate_spread"
    ]   
)

# Checking number of unique values of categorical columns to decide encoding method
obj_col = df.select_dtypes(include='object').columns
print(f"\nNumber of Unique values per Categorical Column: \n{df[obj_col].nunique()}")

# Separate target variable
X = df.drop(columns=['Status'])
y = df['Status']

# One hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data in training and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state = 41
)

# =====================SCALING THE DATA=====================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================Training the model=====================
from sklearn.linear_model import LogisticRegression

# Instance of model
model = LogisticRegression(max_iter=500)

# Train the model
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ===================== CLASSIFICATION REPORT =====================
print("="*30 + "Classification Report Initial" + "="*30)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# =========================================
# OPTIMISING THE MODEL FOR BETTER RESULT
# =========================================
print("="*30 + "OPTIMIZING VIA 'CLASS WEIGHTING'" + "="*30)

# CLASS WEIGHTING

model_2 = LogisticRegression(max_iter=500, class_weight='balanced')
model_2.fit(X_train_scaled, y_train)

y_pred = model_2.predict(X_test_scaled)
print(classification_report(y_test, y_pred))


# CLASSIFICATION THRESHOLD
print("="*20 + "OPTIMIZING VIA 'CLASSIFICATION THRESHOLD'" + "="*20)
y_proba = model_2.predict_proba(X_test_scaled)[:,1]
y_pred_custom = (y_proba >= 0.3).astype(int)
print(classification_report(y_test, y_pred_custom))