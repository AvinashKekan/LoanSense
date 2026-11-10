# create_model.py (NEW VERSION FOR LENDINGCLUB DATA)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import re

print("--- LendingClub Model Training Script Started ---")

# 1. Load Data
print("Step 1: Loading raw LendingClub data (this may take a moment)...")
# Load only a subset of columns to save memory
use_cols = ['loan_status', 'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length', 
            'home_ownership', 'annual_inc', 'purpose', 'dti', 'fico_range_low', 
            'inq_last_6mths', 'pub_rec', 'revol_util']
df = pd.read_csv('data/accepted_2007_to_2018Q4.csv', usecols=use_cols, low_memory=False)

print(f"Loaded {len(df)} records.")

# 2. Data Cleaning and Filtering
print("Step 2: Cleaning and filtering data...")

# A) Filter for only completed loans
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# B) Create binary target variable: 0 for good loan, 1 for bad loan
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# C) Clean emp_length feature
def clean_emp_length(emp_length):
    if pd.isna(emp_length) or emp_length == 'n/a':
        return 0
    elif emp_length == '< 1 year':
        return 0.5
    elif emp_length == '10+ years':
        return 10
    else:
        return int(re.findall(r'\d+', emp_length)[0])

df['emp_length'] = df['emp_length'].apply(clean_emp_length)

# D) Clean term feature (convert ' 36 months' to 36)
df['term'] = df['term'].apply(lambda s: int(s.strip().split()[0]))

# E) Drop rows with missing values in key columns
df.dropna(subset=['dti', 'fico_range_low', 'inq_last_6mths', 'revol_util', 'pub_rec'], inplace=True)

print(f"Data cleaned. Working with {len(df)} completed loan records.")

# 3. Define Features and Target
print("Step 3: Defining features and target...")
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Identify categorical and numerical features
categorical_features = ['grade', 'home_ownership', 'purpose']
numerical_features = ['loan_amnt', 'term', 'int_rate', 'emp_length', 'annual_inc', 
                      'dti', 'fico_range_low', 'inq_last_6mths', 'pub_rec', 'revol_util']

# 4. Create Preprocessing Pipeline
print("Step 4: Creating preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though there shouldn't be
)

# 5. Define the Model and Full Pipeline
print("Step 5: Defining Logistic Regression model...")
# Using Logistic Regression for speed with this large dataset. RandomForest/XGBoost would be even better but much slower to train.
model = LogisticRegression(random_state=42, max_iter=1000)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# 6. Split Data and Train the Model
print("Step 6: Splitting data and training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

# 7. Evaluate the Model
print("Step 7: Evaluating the model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on LendingClub data: {accuracy:.4f}")

# 8. Save the final pipeline
print("Step 8: Saving the final model pipeline...")
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
with open('saved_model/lendingclub_model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\n--- New model trained and saved successfully! ---")