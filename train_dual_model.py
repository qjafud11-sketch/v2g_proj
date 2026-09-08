import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold

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
y_fast = df['로그_급속_충전량'].copy()
y_slow = df['로그_완속_충전량'].copy()
y_true_total = df['총_충전량'].copy()

# 파라미터 세팅 (이전 XGBoost 세팅과 유사하게)
model_fast = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)
model_slow = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)

# CV 예측
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred_fast_log = cross_val_predict(model_fast, X, y_fast, cv=kf)
pred_slow_log = cross_val_predict(model_slow, X, y_slow, cv=kf)

# 앙상블 총 예측량 (원래 스케일 복원 후 합산)
pred_total = np.expm1(pred_fast_log) + np.expm1(pred_slow_log)

r2 = r2_score(y_true_total, pred_total)
mae = mean_absolute_error(y_true_total, pred_total)
rmse = np.sqrt(mean_squared_error(y_true_total, pred_total))

print(f"--- 급속/완속 분리 스태킹 방식 결과 ---")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 전체 데이터 학습
model_fast.fit(X, y_fast)
model_slow.fit(X, y_slow)

# 핫스팟 기준점 계산용 예측
df['예측_급속'] = np.expm1(model_fast.predict(X))
df['예측_완속'] = np.expm1(model_slow.predict(X))
df['예측_충전량'] = df['예측_급속'] + df['예측_완속']
df['핫스팟_점수'] = df['예측_충전량'] / (df['기존_충전기수'] + 1)
df['핫스팟_점수'] = df['핫스팟_점수'].round(2)

q75 = df['핫스팟_점수'].quantile(0.75)
q25 = df['핫스팟_점수'].quantile(0.25)

# 모델 저장
model_data = {
    'model_fast': model_fast,
    'model_slow': model_slow,
    'feature_cols': FEATURE_COLS,
    'hotspot_q75': q75,
    'hotspot_q25': q25,
    'dual_mode': True # 플래그
}
joblib.dump(model_data, os.path.join(DATA_DIR, 'ev_model.pkl'))
df.to_csv(os.path.join(DATA_DIR, 'seoul_ev_predictions.csv'), index=False, encoding='utf-8-sig')
print("모델 및 예측 결과 저장 완료!")
