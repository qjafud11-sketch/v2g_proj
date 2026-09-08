#!/usr/bin/env python3
"""
[Task 2] 서울시 EV 충전 인프라 - 머신러닝 모델링
=================================================
Random Forest & Gradient Boosting 회귀 모델 학습
핫스팟 점수 계산 및 모델 저장

입력: seoul_ev_ml_dataset.csv
출력: ev_model.pkl, seoul_ev_predictions.csv
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("=" * 60)
print("[1] 데이터 로드...")
print("=" * 60)

df = pd.read_csv(
    os.path.join(DATA_DIR, 'seoul_ev_ml_dataset.csv'),
    encoding='utf-8-sig',
    dtype={'행정동코드': str}
)
print(f"  → {len(df)} 행정동, {len(df.columns)} 컬럼")

# ─────────────────────────────────────────────
# 2. Feature / Target 분리
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[2] Feature / Target 분리...")
print("=" * 60)

META_COLS = ['자치구', '행정동', '행정동코드']
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
y = df[TARGET_COL].copy()

print(f"  → Features: {FEATURE_COLS}")
print(f"  → Target: {TARGET_COL}")
print(f"  → X shape: {X.shape}, y shape: {y.shape}")
print(f"  → Target 분포: mean={y.mean():.1f}, std={y.std():.1f}, min={y.min():.1f}, max={y.max():.1f}")

# ─────────────────────────────────────────────
# 3. 모델 학습 및 교차검증
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[3] 모델 학습 및 5-Fold 교차검증...")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    print(f"\n  🔧 {name}:")
    
    # Cross-validation scores
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    results[name] = {
        'MAE': mae_scores.mean(),
        'RMSE': rmse_scores.mean(),
        'R2': r2_scores.mean(),
        'model': model
    }
    
    print(f"     MAE:  {mae_scores.mean():.2f} (±{mae_scores.std():.2f})")
    print(f"     RMSE: {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f})")
    print(f"     R²:   {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")

# ─────────────────────────────────────────────
# 4. 최적 모델 선택 및 전체 데이터 학습
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4] 최적 모델 선택 및 저장...")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"  → 최적 모델: {best_name} (R²={results[best_name]['R2']:.4f})")

# 전체 데이터로 학습
best_model.fit(X, y)
y_pred = best_model.predict(X)

# Feature Importance
print(f"\n  📊 Feature Importance:")
importances = best_model.feature_importances_
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"     {feat:20s} {imp:.4f} {bar}")

# ─────────────────────────────────────────────
# 5. 핫스팟 점수 계산
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[5] 인프라 정체 핫스팟 점수 계산...")
print("=" * 60)

# 로그 변환된 예측값을 원래 스케일로 복원
df['예측_충전량'] = np.expm1(y_pred).round(2)
df['핫스팟_점수'] = (df['예측_충전량'] / (df['기존_충전기수'] + 1)).round(2)

# 핫스팟 등급 분류
q75 = df['핫스팟_점수'].quantile(0.75)
q25 = df['핫스팟_점수'].quantile(0.25)

def classify_hotspot(score):
    if score >= q75:
        return '🔴 최우선 시급'
    elif score >= q25:
        return '🟡 잠재적 정체'
    else:
        return '🟢 여유'

df['핫스팟_등급'] = df['핫스팟_점수'].apply(classify_hotspot)

print(f"  → 핫스팟 기준: 상위 25% (점수 ≥ {q75:.2f}) = 최우선 시급")
print(f"  → 등급 분포:")
print(df['핫스팟_등급'].value_counts().to_string())

print(f"\n  🔥 Top 15 핫스팟 행정동:")
top15 = df.nlargest(15, '핫스팟_점수')
print(top15[['자치구', '행정동', '인구수', '추정_전기차수', '기존_충전기수', 
             '예측_충전량', '핫스팟_점수', '핫스팟_등급']].to_string(index=False))

# ─────────────────────────────────────────────
# 6. 모델 및 결과 저장
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[6] 모델 및 결과 저장...")
print("=" * 60)

# 모델 + 메타데이터 저장
model_data = {
    'model': best_model,
    'feature_cols': FEATURE_COLS,
    'target_col': TARGET_COL,
    'model_name': best_name,
    'r2_score': results[best_name]['R2'],
    'hotspot_q75': q75,
    'hotspot_q25': q25
}
model_path = os.path.join(DATA_DIR, 'ev_model.pkl')
joblib.dump(model_data, model_path)
print(f"  ✅ 모델 저장: {model_path}")

# 예측 결과 CSV 저장
pred_path = os.path.join(DATA_DIR, 'seoul_ev_predictions.csv')
df.to_csv(pred_path, index=False, encoding='utf-8-sig')
print(f"  ✅ 예측 결과 저장: {pred_path}")

# ─────────────────────────────────────────────
# 7. Orange3 워크플로우 안내
# ─────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("[참고] Orange3 워크플로우 구성 안내:")
print("=" * 60)
print("""
  Orange3에서 동일한 분석을 수행하려면:
  
  1. [File] 위젯: seoul_ev_ml_dataset.csv 로드
  2. [Select Columns] 위젯:
     - Meta: 자치구, 행정동, 행정동코드
     - Features: 인구수, 추정_전기차수, 공영주차장_면수, 기존_충전기수,
                 충전기당_인구수, 전기차_충전기_비율, 주차면_충전기_비율
     - Target: 총_충전량
  3. [Random Forest] / [Gradient Boosting] 위젯 연결
  4. [Test & Score] 위젯: 5-Fold Cross Validation
  5. [Predictions] 위젯 → [Data Table] 위젯으로 결과 확인
""")

print("🎯 전체 학습 파이프라인 완료!")
