import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv('seoul_ev_predictions.csv')
y_true_log = df['로그_총_충전량']
y_pred_log = np.log1p(df['예측_충전량'])

print(f"Log scale combined R2: {r2_score(y_true_log, y_pred_log)}")
