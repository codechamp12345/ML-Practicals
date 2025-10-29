# Step 1: Import required libraries
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Load dataset and show first rows
data = pd.read_csv("salary.csv")
print(data.head())

# Step 3: Divide into feature (X) and target (y)
X = data[['months']]
y = data[['salary']]

# Step 4: Split data into training and testing (80% train, 20% test)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale X and y using StandardScaler
sx = StandardScaler()   # scaler for X
sy = StandardScaler()   # scaler for y
X_tr = sx.fit_transform(X_tr)
X_te = sx.transform(X_te)
y_tr = sy.fit_transform(y_tr)
y_te = sy.transform(y_te)

# Step 6: Create and train Linear Regression model
model = LinearRegression()
model.fit(X_tr, y_tr)

# Step 7: Predict and convert predictions back to original scale
y_pred = model.predict(X_te)
y_pred = sy.inverse_transform(y_pred)
y_te = sy.inverse_transform(y_te)

# Step 8: Evaluate model using MAE, MSE, RMSE
mae = mean_absolute_error(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Results:")
print("MAE =", mae)
print("MSE =", mse)
print("RMSE =", rmse)