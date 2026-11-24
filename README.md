# v2g_proj · 서울시 전기차 충전 인프라 핫스팟 분석

서울시 행정동별 전기차 충전량을 예측하고, 기존 충전기 수 대비 예상 수요가 높은 지역을 Streamlit 지도에서 비교하는 프로젝트입니다. 행정동을 선택한 뒤 인구수, 추정 전기차 수, 충전기 수, 공영주차장 면수, 쇼핑시설 수를 바꾸면 예상 충전량과 핫스팟 등급을 확인할 수 있습니다.

## 프로젝트 구성

구성 요소는 모두 저장소 루트에 있습니다.

| 구성 요소 | 역할 | 진입점 |
| --- | --- | --- |
| 충전 인프라 대시보드 | 행정동 지도, 수요 시뮬레이션, 핫스팟 상위 10곳, 자치구별 요약 | [기본 대시보드](app.py) · [스태킹 대시보드](app_stacking.py) |
| 데이터 전처리 | 지역 통계와 충전소 데이터를 행정동 단위로 통합 | [`preprocess.py`](preprocess.py) |
| 모델 학습 | 충전량을 예측하고 모델과 행정동별 결과를 저장 | [기본 모델](train_model.py) · [스태킹 모델](train_stacking.py) |
| 모델 실험 | 급속·완속 분리 예측, 스태킹 튜닝, 모델 비교 | [급속·완속 분리](train_dual_model.py) · [스태킹 튜닝](train_stacking_tuned.py) · [모델 비교](run_experiments.py) |

핫스팟 점수는 `예측 충전량(kWh) ÷ (기존 충전기 수 + 1)`로 계산합니다. 모델 학습 시 구한 점수의 75·25백분위수를 기준으로 최우선 시급, 잠재적 정체, 여유 등급을 나눕니다. [점수 계산 코드](train_model.py)

## 기본 대시보드 실행

저장소 루트에서 Python 가상환경을 만들고 라이브러리를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy scikit-learn joblib openpyxl streamlit folium streamlit-folium shapely
```

[원본 데이터](#원본-데이터)를 같은 폴더에 준비한 뒤 전처리, 학습, 대시보드 실행 순으로 진행합니다.

```bash
python preprocess.py
python train_model.py
python -m streamlit run app.py
```

전처리는 `seoul_ev_ml_dataset.csv`를 생성하고, 기본 학습은 `ev_model.pkl`과 `seoul_ev_predictions.csv`를 생성합니다. 이미 이 모델과 예측 결과를 준비했다면, 두 파일과 `서울시_행정동_중심점_2017.csv`를 루트에 놓고 Streamlit 실행 명령부터 사용할 수 있습니다. 행정동 경계 GeoJSON은 저장소에 포함되어 있습니다. 지도에서 행정동코드를 찾지 못하면 중심점 CSV의 자치구·행정동 이름이 경계 파일과 일치하는지 확인하세요. [행정동 경계 파일](서울_행정동_경계_2017.geojson)

## 스태킹 대시보드 실행

기본 학습까지 마친 상태에서 스태킹 모델을 추가로 학습합니다.

```bash
python train_stacking.py
python -m streamlit run app_stacking.py
```

스태킹 학습은 `ev_stacking_model.pkl`과 `seoul_ev_stacking_predictions.csv`를 생성합니다. 스태킹 대시보드는 시뮬레이션에 `ev_stacking_model.pkl`을 사용하며, 지도와 순위는 기본 학습이 생성한 `seoul_ev_predictions.csv`에서 읽습니다. [스태킹 학습 코드](train_stacking.py) · [대시보드 데이터 로더](app_stacking.py)

## 원본 데이터

전처리 스크립트가 읽는 CSV·XLSX는 별도로 확보해 `preprocess.py`와 같은 폴더에 아래 파일명으로 배치합니다. 전처리에는 코드에 정의된 열 이름과 인코딩에 맞는 파일이 필요합니다. [입력 처리 코드](preprocess.py)

| 데이터 | 파일명 |
| --- | --- |
| 행정동 중심점 | `서울시_행정동_중심점_2017.csv` |
| 생활인구 | `LOCAL_PEOPLE_DONG_202605.csv` |
| 자치구별 차량 등록 | `서울시 자치구별 연료별 차종별 용도별 등록현황(26년5월).xlsx` |
| 법정동·행정동 연계 | `국가데이터처_법정동 연계정보_20250602.csv` |
| 충전소 목록 | `충전소 리스트 (2).xlsx` |
| 충전량 실적 | `한국전력공사_서울시 전기차 충전소 충전량_20220331.xlsx` |
| 공영주차장 | `서울시 공영주차장 안내 정보.csv` |
| 쇼핑시설 | `서울시 쇼핑시설물 정보 (한국어_2015년).csv` |
| 주택 유형별 집계 | `101_DT_1JU1501_20260708121347.csv` |
| 상권 점포 수 | `서울시 상권분석서비스(점포-행정동).csv` |
| 지하철 승차 인원 | `서울시 행정동별 지하철 총 승차 승객수 정보.csv` |
| 도심 교통량(선택) | `도심지점별+교통량_20260708152259.csv` |

도심 교통량 파일을 읽거나 처리하지 못하면 전처리에서 교통량을 0으로 채웁니다. 원본 데이터와 생성된 CSV·PKL 파일은 Git 추적 대상에서 제외되어 있습니다. [`.gitignore`](.gitignore)
