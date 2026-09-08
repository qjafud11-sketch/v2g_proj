import pandas as pd
import numpy as np
import os
import joblib
import optuna
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, 'seoul_ev_ml_dataset.csv')

df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')

# --- 1. 강력한 파생 변수(Interaction Features) 추가 ---
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
    # 새롭게 추가된 파생 변수들
    '인구수_x_전기차수', '쇼핑시설_x_주차장', '전기차밀집도'
]

X = df[FEATURE_COLS].copy()
y_log = df['로그_총_충전량'].copy() 
y_true_total = df['총_충전량'].copy()

# --- 2. Optuna를 활용한 하이퍼파라미터 자동 튜닝 ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("🔍 Optuna: Random Forest 최적화 진행 중 (20 Trials)...")
def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 400, step=50)
    max_depth = trial.suggest_int('max_depth', 4, 12)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    score = cross_val_score(model, X, y_log, cv=kf, scoring='r2').mean()
    return score

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=20)
best_rf = RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)
print(f"✅ RF 최적 파라미터 찾기 완료!")

print("🔍 Optuna: Gradient Boosting 최적화 진행 중 (20 Trials)...")
def gb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 400, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 8)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        random_state=42
    )
    score = cross_val_score(model, X, y_log, cv=kf, scoring='r2').mean()
    return score

gb_study = optuna.create_study(direction='maximize')
gb_study.optimize(gb_objective, n_trials=20)
best_gb = GradientBoostingRegressor(**gb_study.best_params, random_state=42)
print(f"✅ GB 최적 파라미터 찾기 완료!")

# --- 3. 튜닝된 베이스 모델로 Stacking 구성 ---
print("\n🚀 튜닝 완료된 맞춤형 모델들로 Stacking 앙상블 학습 시작...")
estimators = [
    ('RF', best_rf),
    ('GB', best_gb),
    ('kNN', KNeighborsRegressor(n_neighbors=5))
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0), # 릿지 규제를 추가하여 오차 튀는 것 방지
    cv=5,
    n_jobs=-1
)

# --- 4. 최종 모델 평가 (가장 엄격한 원래 단위 변환 후 채점) ---
pred_log = cross_val_predict(stacking_model, X, y_log, cv=kf)
pred_total = np.expm1(pred_log)

r2 = r2_score(y_true_total, pred_total)
mae = mean_absolute_error(y_true_total, pred_total)
rmse = np.sqrt(mean_squared_error(y_true_total, pred_total))

print("\n" + "="*50)
print(f"🌟 [최종 최적화 버전] 단일 스태킹 평가 결과")
print("="*50)
print(f"R²   : {r2:.4f}  (기존 0.5429 대비 향상 여부 확인)")
print(f"MAE  : {mae:,.0f} kWh")
print(f"RMSE : {rmse:,.0f} kWh")
print("="*50)

# --- 5. 전체 데이터 학습 및 앱 연동을 위한 저장 ---
stacking_model.fit(X, y_log)

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
model_path = os.path.join(DATA_DIR, 'ev_stacking_tuned_model.pkl')
joblib.dump(model_data, model_path)
df.to_csv(os.path.join(DATA_DIR, 'seoul_ev_stacking_tuned_predictions.csv'), index=False, encoding='utf-8-sig')

print(f"\n✅ 최적화된 새로운 모델 저장 완료: {model_path}")
