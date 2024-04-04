from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import numpy as np

# Load the dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
x = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
print(y)
y = y.replace({'M': 1, 'B': 0})

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Option to scale the data
scale_data = False  # Change to True if you wish to scale the data
if scale_data:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
