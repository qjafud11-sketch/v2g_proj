import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, 'seoul_ev_ml_dataset.csv')

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')

# Orange3에서 설정했던 Feature들과 동일하게 유지 (누수 방지를 위해 충전_건수 제외)
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
y_log = df['로그_총_충전량'].copy()  # 학습 안정성을 위해 로그 타겟 유지
y_true_total = df['총_충전량'].copy()

print("Base Learner(기본 모델) 세팅 중...")
estimators = [
    ('RF', RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)),
    ('GB', GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)),
    ('kNN', KNeighborsRegressor(n_neighbors=5))
]

print("Stacking 모델 구성 중 (Meta-Learner: Linear Regression)...")
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

print("Stacking 앙상블 학습 시작 (약 10~20초 소요)...")
stacking_model.fit(X, y_log)

# 예측 및 성능 평가
pred_log = stacking_model.predict(X)
pred_total = np.expm1(pred_log) # 원래 스케일로 복원

r2 = r2_score(y_true_total, pred_total)
print(f"--- Stacking 앙상블 결과 ---")
print(f"R²: {r2:.4f}")

# 핫스팟 기준점 계산용 예측
df['예측_충전량'] = pred_total
df['핫스팟_점수'] = df['예측_충전량'] / (df['기존_충전기수'] + 1)
df['핫스팟_점수'] = df['핫스팟_점수'].round(2)

q75 = df['핫스팟_점수'].quantile(0.75)
q25 = df['핫스팟_점수'].quantile(0.25)

# 웹사이트 연동을 위한 사전(Dictionary) 형태로 포장하여 저장
model_data = {
    'model': stacking_model,
    'feature_cols': FEATURE_COLS,
    'hotspot_q75': q75,
    'hotspot_q25': q25,
    'dual_mode': False # 단일 모델(스태킹)임을 표시
}
joblib.dump(model_data, os.path.join(DATA_DIR, 'ev_stacking_model.pkl'))
df.to_csv(os.path.join(DATA_DIR, 'seoul_ev_stacking_predictions.csv'), index=False, encoding='utf-8-sig')

print("✅ 스태킹 모델(ev_stacking_model.pkl) 및 결과 저장 완료!")
