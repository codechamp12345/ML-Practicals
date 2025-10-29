import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


iris = load_iris()
X = iris.data      
y = iris.target    

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')

print("\nK-Fold Cross-Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())
