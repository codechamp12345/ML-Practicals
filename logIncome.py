# # 1) Import required libraries
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report

# # 2) Load the built-in digits dataset
# digits = load_digits()
# X = digits.data      # features
# y = digits.target    # labels (0-9 digits)

# # 3) Split the dataset into training and testing (80-20)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 4) Data Scaling (Standardization)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # 5) Create Logistic Regression model
# model = LogisticRegression(max_iter=5000)

# # 6) Train (Fit) the model
# model.fit(X_train, y_train)

# # 7) Predict for test data
# y_pred = model.predict(X_test)

# # 8) Confusion Matrix & Classification Report
# print("\n--- Confusion Matrix ---")
# print(confusion_matrix(y_test, y_pred))

# print("\n--- Classification Report ---")
# print(classification_report(y_test, y_pred))

# # 9) K-Fold Cross Validation (cv=5)
# cv_score = cross_val_score(model, X, y, cv=5)
# print("\n--- 5-Fold Cross Validation Score ---")
# print("Scores:", cv_score)
# print("Mean Score:", np.mean(cv_score))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load CSV
data = pd.read_csv("income.csv")
print(data.head())
print("\nColumns:", data.columns)

# Create classification target: High vs Low income
median_income = data['income'].median()
data['income_class'] = (data['income'] > median_income).astype(int)

# Features & target
X = data[['age', 'experience']]
y = data['income_class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix & Classification Report
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# K-Fold Cross validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("\n--- 5-Fold Cross Validation ---")
print("Scores:", cv_scores)
print("Mean:", np.mean(cv_scores))
