import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("salary.csv")
print(data.head())

X = data[['months']]
y = data[['salary']]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

sx = StandardScaler()   
sy = StandardScaler()
X_tr = sx.fit_transform(X_tr)
X_te = sx.transform(X_te)
y_tr = sy.fit_transform(y_tr)
y_te = sy.transform(y_te)

model = LinearRegression()
model.fit(X_tr, y_tr)

y_pred = model.predict(X_te)
y_pred = sy.inverse_transform(y_pred)
y_te = sy.inverse_transform(y_te)


mae = mean_absolute_error(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Results:")
print("MAE =", mae)
print("MSE =", mse)
print("RMSE =", rmse)