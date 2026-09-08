import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import os

df = pd.read_csv('seoul_ev_ml_dataset.csv')
FEATURE_COLS = [
    '인구수', '추정_전기차수', '쇼핑시설물_수', '추정_주택수', '추정_아파트수', '아파트_비율',
    '구별_평균_총충전량', '구별_총_지하철승객수', '구별_총_점포수',
    '로그_공영주차장_면수', '로그_상권_총_점포수', '로그_지하철_총_승하차수', 
    '로그_기존_충전기수', '로그_급속_충전기수', '로그_완속_충전기수', '로그_도심_교통량',
    '충전기당_인구수', '전기차_충전기_비율', '주차면_충전기_비율',
    '주차면_대비_지하철승객수', '점포당_평균_생활인구', '전기차_대비_충전기_로그비율',
    '급속_비율', '완속_비율', '전기차_급속_비율', '전기차_완속_비율', '주차면_급속_비율', '주차면_완속_비율',
    '아파트_급속_비율', '아파트_완속_비율', '교통량_대비_충전기비율', '아파트_대비_충전기비율'
]
TARGET_COL = '로그_총_충전량'

X = df[FEATURE_COLS].copy()
X = pd.concat([X, pd.get_dummies(df['자치구'], prefix='자치구')], axis=1)
y = df[TARGET_COL].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
r2 = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
print(f"With 자치구 OHE: R2 = {r2:.4f}")

# Target leakage test
X_leak = X.copy()
X_leak['충전_건수'] = df['충전_건수']
r2_leak = cross_val_score(model, X_leak, y, cv=kf, scoring='r2').mean()
print(f"With 충전_건수: R2 = {r2_leak:.4f}")

