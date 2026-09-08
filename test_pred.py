import joblib
import pandas as pd
import numpy as np

model_data = joblib.load('ev_model.pkl')
model = model_data['model']
target_col = model_data.get('target_col', 'Unknown')
print(f"Target column in model metadata: {target_col}")

# Load a sample and predict
df = pd.read_csv('seoul_ev_ml_dataset.csv')
features = model_data['feature_cols']
pred = model.predict(df[features].iloc[[0]])[0]
print(f"Sample prediction value: {pred}")
