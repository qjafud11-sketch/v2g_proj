import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('seoul_ev_ml_dataset.csv')
FEATURE_COLS = [
    '인구수', '추정_전기차수', '쇼핑시설물_수', '추정_주택수', '추정_아파트수', '아파트_비율',
    '구별_평균_총충전량', '구별_총_지하철승객수', '구별_총_점포수',
    '로그_공영주차장_면수', '로그_상권_총_점포수', '로그_지하철_총_승하차수', '로그_기존_충전기수',
    '충전기당_인구수', '전기차_충전기_비율', '주차면_충전기_비율',
    '주차면_대비_지하철승객수', '점포당_평균_생활인구', '전기차_대비_충전기_로그비율'
]
TARGET_COL = '로그_총_충전량'

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X, y, name="Current"):
    model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
    r2 = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
    print(f"[{name}] R2: {r2:.4f}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]
evaluate(X, y, "Baseline (Current)")

# Experiment 1: Add 자치구 One-Hot Encoding
df_exp1 = pd.get_dummies(df, columns=['자치구'])
exp1_cols = FEATURE_COLS + [c for c in df_exp1.columns if c.startswith('자치구_')]
evaluate(df_exp1[exp1_cols], y, "Exp 1 (Gu One-Hot)")

# Experiment 2: Log-transform more features
df_exp2 = df.copy()
for col in ['인구수', '추정_전기차수', '추정_주택수', '추정_아파트수']:
    df_exp2[f'로그_{col}'] = np.log1p(df_exp2[col])
exp2_cols = FEATURE_COLS + [f'로그_{col}' for col in ['인구수', '추정_전기차수', '추정_주택수', '추정_아파트수']]
evaluate(df_exp2[exp2_cols], y, "Exp 2 (More Log features)")

# Experiment 3: Combine Exp 1 & 2
df_exp3 = pd.get_dummies(df_exp2, columns=['자치구'])
exp3_cols = exp2_cols + [c for c in df_exp3.columns if c.startswith('자치구_')]
evaluate(df_exp3[exp3_cols], y, "Exp 3 (Gu One-Hot + Log features)")

# Experiment 4: Drop raw features replaced by logs
exp4_cols = [c for c in exp3_cols if c not in ['인구수', '추정_전기차수', '추정_주택수', '추정_아파트수']]
evaluate(df_exp3[exp4_cols], y, "Exp 4 (Exp 3 minus raw features)")

