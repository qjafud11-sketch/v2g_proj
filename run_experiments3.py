import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
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
X = df[FEATURE_COLS]
y = df[TARGET_COL]

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X[['구별_평균_총충전량', '로그_기존_충전기수', '주차면_대비_지하철승객수', '로그_공영주차장_면수']])
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
X_combined = pd.concat([X, X_poly_df], axis=1)

model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
r2 = cross_val_score(model, X_combined, y, cv=kf, scoring='r2').mean()
print(f"Polynomial Interactions R2: {r2:.4f}")

