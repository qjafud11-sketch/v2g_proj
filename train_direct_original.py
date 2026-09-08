import pandas as pd
import numpy as np
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, 'seoul_ev_ml_dataset.csv')

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')

# 파생 변수 추가 (강력한 힌트 제공)
df['인구수_x_전기차수'] = df['인구수'] * df['추정_전기차수']
df['쇼핑시설_x_주차장'] = df['쇼핑시설물_수'] * df['공영주차장_면수']
df['전기차밀집도'] = df['추정_전기차수'] / (df['인구수'] + 1)

FEATURE_COLS = [
    '인구수', '추정_전기차수', '쇼핑시설물_수', '추정_주택수', '추정_아파트수', '아파트_비율',
    '구별_평균_총충전량', '구별_총_지하철승객수', '구별_총_점포수',
    '로그_공영주차장_면수', '로그_상권_총_점포수', '로그_지하철_총_승하차수', 
    '로그_기존_충전기수', '로그_급속_충전기수', '로그_완속_충전기수', '로그_도심_교통량',
    '충전기당_인구수', '전기차_충전기_비율', '주차면_충전기_비율',
    '주차면_대비_지하철승객수', '점포당_평균_생활인구', '전기차_대비_충전기_로그비율',
    '급속_비율', '완속_비율', '전기차_급속_비율', '전기차_완속_비율', '주차면_급속_비율', '주차면_완속_비율',
    '아파트_급속_비율', '아파트_완속_비율', '교통량_대비_충전기비율', '아파트_대비_충전기비율',
    '인구수_x_전기차수', '쇼핑시설_x_주차장', '전기차밀집도'
]

X = df[FEATURE_COLS].copy()
y_true_total = df['총_충전량'].copy() # 🚨 로그 변환 완전히 제거! 원래 단위(kWh) 그대로 타겟 설정

print("🚀 로그 변환 없이 원래 단위(총 충전량)로 정면 승부 학습 시작...")

# 3대장 부스팅 + RF 구성 (트리 깊이와 학습 횟수 조금 넉넉히)
estimators = [
    ('XGB', XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42, n_jobs=-1)),
    ('LGBM', LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)),
    ('Cat', CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, random_state=42, verbose=0)),
    ('RF', RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1))
]

# Meta Learner (Ridge를 사용하되 페널티(alpha)를 강하게 줘서 오버피팅 억제)
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=10.0), 
    cv=5,
    n_jobs=1 
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("⏳ 5-Fold 교차 검증 중 (약 1~2분 소요)...")

# 예측 자체가 로그 변환 없이 수만 단위로 곧바로 나옵니다.
pred_total = cross_val_predict(stacking_model, X, y_true_total, cv=kf)

# 선형 결합(Ridge) 특성상 드물게 음수 예측이 나올 수 있으므로 0으로 방어(Clipping)
pred_total = np.clip(pred_total, 0, None)

r2 = r2_score(y_true_total, pred_total)
mae = mean_absolute_error(y_true_total, pred_total)
rmse = np.sqrt(mean_squared_error(y_true_total, pred_total))

print("\n" + "="*50)
print(f"🔥 [로그 제거 & 파생변수 & 3대장] 최종 성능")
print("="*50)
print(f"R²   : {r2:.4f}  (기존 한계치 0.5429 돌파 여부 확인)")
print(f"MAE  : {mae:,.0f} kWh")
print(f"RMSE : {rmse:,.0f} kWh")
print("="*50)

# 앱 연동을 위한 전체 데이터 핏 및 저장
stacking_model.fit(X, y_true_total)

df['예측_충전량'] = np.clip(stacking_model.predict(X), 0, None)
df['핫스팟_점수'] = df['예측_충전량'] / (df['기존_충전기수'] + 1)
df['핫스팟_점수'] = df['핫스팟_점수'].round(2)

q75 = df['핫스팟_점수'].quantile(0.75)
q25 = df['핫스팟_점수'].quantile(0.25)

model_data = {
    'model': stacking_model,
    'feature_cols': FEATURE_COLS,
    'hotspot_q75': q75,
    'hotspot_q25': q25,
    'dual_mode': False,
    'no_log_transform': True # 앱 구동 시 구분을 위한 중요한 플래그!
}

model_path = os.path.join(DATA_DIR, 'ev_direct_model.pkl')
joblib.dump(model_data, model_path)
df.to_csv(os.path.join(DATA_DIR, 'seoul_ev_direct_predictions.csv'), index=False, encoding='utf-8-sig')

print(f"\n✅ 직공법 모델 저장 완료: {model_path}")
print("※ 주의: 이 모델은 예측값이 이미 원래 스케일입니다. app.py 연동 시 업데이트가 필요합니다.")
