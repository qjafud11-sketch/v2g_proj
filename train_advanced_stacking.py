import pandas as pd
import numpy as np
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

# New Advanced Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, 'seoul_ev_ml_dataset.csv')

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')

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

X = df[FEATURE_COLS].copy()
y_log = df['로그_총_충전량'].copy() 
y_true_total = df['총_충전량'].copy()

print("🚀 부스팅 3대장(Base Learners) 세팅 중...")
estimators = [
    ('XGB', XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)),
    ('LGBM', LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)),
    ('Cat', CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, random_state=42, verbose=0)),
    ('RF', RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1))
]

print("🧠 초고성능 스태킹 모델 구성 중 (Meta-Learner: Ridge)...")
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0), # 오차 폭발 방지 규제
    cv=5,
    n_jobs=1 
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("⏳ 5-Fold 교차 검증 예측 시작 (약 1~2분 소요)...")
pred_log = cross_val_predict(stacking_model, X, y_log, cv=kf)

# 평가
pred_total = np.expm1(pred_log)
r2 = r2_score(y_true_total, pred_total)
mae = mean_absolute_error(y_true_total, pred_total)
rmse = np.sqrt(mean_squared_error(y_true_total, pred_total))

print("\n" + "="*50)
print(f"🌟 Advanced Stacking 앙상블 평가 결과 (원래 스케일)")
print("="*50)
print(f"R²   : {r2:.4f}  (기존 0.5429 대비 향상 여부 확인)")
print(f"MAE  : {mae:,.0f} kWh")
print(f"RMSE : {rmse:,.0f} kWh")
print("="*50)

print("\n전체 데이터로 최종 학습 진행 중...")
stacking_model.fit(X, y_log)

# 시뮬레이션용 핫스팟 기준점 계산
df['예측_충전량'] = np.expm1(stacking_model.predict(X))
df['핫스팟_점수'] = df['예측_충전량'] / (df['기존_충전기수'] + 1)
df['핫스팟_점수'] = df['핫스팟_점수'].round(2)

q75 = df['핫스팟_점수'].quantile(0.75)
q25 = df['핫스팟_점수'].quantile(0.25)

model_data = {
    'model': stacking_model,
    'feature_cols': FEATURE_COLS,
    'hotspot_q75': q75,
    'hotspot_q25': q25,
    'dual_mode': False
}

model_path = os.path.join(DATA_DIR, 'ev_advanced_stacking_model.pkl')
joblib.dump(model_data, model_path)
df.to_csv(os.path.join(DATA_DIR, 'seoul_ev_advanced_predictions.csv'), index=False, encoding='utf-8-sig')

print(f"\n✅ 최고 성능 모델 저장 완료: {model_path}")
