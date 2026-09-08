#!/usr/bin/env python3
"""
[Task 3] 서울시 EV 충전 인프라 - Streamlit 실시간 시뮬레이션 대시보드
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="서울시 EV 충전 인프라 핫스팟 분석", layout="wide", page_icon="⚡")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
* { font-family: 'Noto Sans KR', sans-serif; }
.main-title {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-title h1 { color: #00d4ff; font-size: 1.8rem; margin: 0; font-weight: 900; }
.main-title p { color: #a0aec0; margin: 0.3rem 0 0; font-size: 0.9rem; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(0,212,255,0.2); border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .label { color: #a0aec0; font-size: 0.75rem; font-weight: 500; }
.metric-card .value { color: #00d4ff; font-size: 1.5rem; font-weight: 900; margin: 0.3rem 0; }
.metric-card .unit { color: #718096; font-size: 0.7rem; }
.hotspot-badge {
    display: inline-block; padding: 0.5rem 1.5rem; border-radius: 25px;
    font-weight: 700; font-size: 1rem; text-align: center; width: 100%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.badge-critical { background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white; }
.badge-warning { background: linear-gradient(135deg, #f7971e, #ffd200); color: #1a1a2e; }
.badge-safe { background: linear-gradient(135deg, #11998e, #38ef7d); color: #1a1a2e; }
.info-panel {
    background: linear-gradient(135deg, #0f0c29, #302b63);
    border: 1px solid rgba(0,212,255,0.15); border-radius: 12px;
    padding: 1.2rem; margin: 0.5rem 0;
}
.info-panel h4 { color: #00d4ff; margin: 0 0 0.5rem; }
.rank-table { width: 100%; border-collapse: collapse; }
.rank-table th { color: #00d4ff; font-size: 0.75rem; padding: 0.4rem; border-bottom: 1px solid rgba(0,212,255,0.2); text-align: left; }
.rank-table td { color: #e2e8f0; font-size: 0.8rem; padding: 0.35rem 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
.stApp { background-color: #0a0a1a; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #1a1a2e); }
</style>
""", unsafe_allow_html=True)

# ─── 데이터 로드 ───
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'seoul_ev_predictions.csv'),
                     encoding='utf-8-sig', dtype={'행정동코드': str})
    with open(os.path.join(DATA_DIR, '서울_행정동_경계_2017.geojson'), 'r', encoding='utf-8') as f:
        geojson = json.load(f)
    # 중심점 로드
    df_center = pd.read_csv(os.path.join(DATA_DIR, '서울시_행정동_중심점_2017.csv'),
                            encoding='cp949', dtype={'코드': str})
    return df, geojson, df_center

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(DATA_DIR, 'ev_model.pkl'))

df, geojson, df_center = load_data()
model_data = load_model()
if model_data.get('dual_mode'):
    model_fast = model_data['model_fast']
    model_slow = model_data['model_slow']
    model = None
else:
    model = model_data['model']
FEATURE_COLS = model_data['feature_cols']
q75 = model_data['hotspot_q75']
q25 = model_data['hotspot_q25']

# GeoJSON adm_cd → 행정동코드 매핑
def normalize_dong(name):
    return name.strip().replace('·', '.').replace('ㆍ', '.').replace('제', '')

center_lookup = {}
for _, row in df_center.iterrows():
    key = (row['시군구명'], normalize_dong(row['읍면동명']))
    center_lookup[key] = row['코드']

geo_to_pop = {}
for feat in geojson['features']:
    parts = feat['properties']['adm_nm'].split()
    gu, dong = parts[1], normalize_dong(' '.join(parts[2:]))
    if (gu, dong) in center_lookup:
        code = center_lookup[(gu, dong)]
        geo_to_pop[feat['properties']['adm_cd']] = code
        feat['properties']['행정동코드'] = code

# ─── 타이틀 ───
st.markdown("""
<div class="main-title">
    <h1>⚡ 서울시 EV 충전 인프라 핫스팟 분석 대시보드</h1>
    <p>공간 데이터 융합 기반 · 행정동별 충전 수요 예측 및 실시간 시뮬레이션</p>
</div>
""", unsafe_allow_html=True)

# ─── 레이아웃 ───
col_map, col_panel = st.columns([3, 2])

with col_map:
    st.markdown("#### 🗺️ 서울시 행정동별 핫스팟 지도")
    st.caption("행정동을 클릭하면 우측에서 상세 분석 및 시뮬레이션이 가능합니다")

    # Choropleth 맵 생성
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11,
                   tiles='CartoDB dark_matter')

    # 핫스팟 점수 → GeoJSON에 매핑
    score_map = dict(zip(df['행정동코드'].astype(str), df['핫스팟_점수']))

    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=df,
        columns=['행정동코드', '핫스팟_점수'],
        key_on='feature.properties.행정동코드',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.3,
        line_weight=0.5,
        legend_name='핫스팟 점수 (예측 충전량 / 충전기수)',
        nan_fill_color='#1a1a2e'
    ).add_to(m)

    # 툴팁 추가
    folium.GeoJsonTooltip(
        fields=['adm_nm'],
        aliases=['행정동:'],
        style="background-color:#1a1a2e;color:#00d4ff;font-size:12px;padding:6px;"
    ).add_to(choropleth.geojson)

    map_data = st_folium(m, width=None, height=550, returned_objects=["last_object_clicked"])

# 클릭된 행정동 감지
clicked_dong_code = None
clicked_info = None

if map_data and map_data.get("last_object_clicked"):
    lat = map_data["last_object_clicked"].get("lat")
    lng = map_data["last_object_clicked"].get("lng")
    if lat and lng:
        from shapely.geometry import Point, shape
        click_point = Point(lng, lat)
        for feat in geojson['features']:
            poly = shape(feat['geometry'])
            if poly.contains(click_point):
                adm_cd = feat['properties']['adm_cd']
                if adm_cd in geo_to_pop:
                    clicked_dong_code = geo_to_pop[adm_cd]
                    clicked_info = feat['properties']
                break

# 세션 상태 관리
if clicked_dong_code:
    st.session_state['selected_dong'] = clicked_dong_code

selected_dong = st.session_state.get('selected_dong', df.iloc[0]['행정동코드'])

# ─── 우측 패널 ───
with col_panel:
    dong_data = df[df['행정동코드'] == str(selected_dong)]
    if len(dong_data) == 0:
        dong_data = df.iloc[[0]]
    dong = dong_data.iloc[0]

    st.markdown(f"""
    <div class="info-panel">
        <h4>📍 {dong['자치구']} {dong['행정동']}</h4>
    </div>
    """, unsafe_allow_html=True)

    # 현재 데이터 메트릭
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">인구수</div>
            <div class="value">{dong['인구수']:,.0f}</div>
            <div class="unit">명</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">추정 전기차</div>
            <div class="value">{dong['추정_전기차수']:,.0f}</div>
            <div class="unit">대</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">기존 충전기</div>
            <div class="value">{dong['기존_충전기수']:,.0f}</div>
            <div class="unit">기</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">실제 충전량</div>
            <div class="value">{dong['총_충전량']:,.0f}</div>
            <div class="unit">kWh</div>
        </div>""", unsafe_allow_html=True)

    # 시뮬레이션 슬라이더
    st.markdown("---")
    st.markdown("#### 🎛️ 수요 시뮬레이션")
    st.caption("슬라이더를 조정하여 미래 시나리오의 예상 충전 수요를 예측합니다")

    sim_pop = st.slider("👥 인구수", min_value=1000, max_value=150000,
                        value=int(dong['인구수']), step=1000, key="sim_pop")
    sim_ev = st.slider("🚗 추정 전기차수", min_value=10, max_value=5000,
                       value=int(dong['추정_전기차수']), step=50, key="sim_ev")
    sim_charger = st.slider("🔌 기존 충전기수", min_value=0, max_value=1000,
                            value=int(dong['기존_충전기수']), step=10, key="sim_charger")
    sim_parking = st.slider("🅿️ 공영주차장 면수", min_value=0, max_value=3000,
                            value=int(dong['공영주차장_면수']), step=50, key="sim_parking")
    sim_shop = st.slider("🏬 쇼핑시설물 수", min_value=0, max_value=20,
                            value=int(dong['쇼핑시설물_수']), step=1, key="sim_shop")

    # 선택된 동의 원본 데이터를 복사
    sim_dict = dong.to_dict()
    
    # 슬라이더 값 덮어쓰기
    sim_dict['인구수'] = sim_pop
    sim_dict['추정_전기차수'] = sim_ev
    sim_dict['기존_충전기수'] = sim_charger
    sim_dict['공영주차장_면수'] = sim_parking
    sim_dict['쇼핑시설물_수'] = sim_shop
    
    # 주요 파생 변수 재계산
    sim_dict['로그_공영주차장_면수'] = np.log1p(sim_parking)
    sim_dict['로그_기존_충전기수'] = np.log1p(sim_charger)
    sim_dict['충전기당_인구수'] = round(sim_pop / (sim_charger + 1), 1)
    sim_dict['전기차_충전기_비율'] = round(sim_ev / (sim_charger + 1), 2)
    sim_dict['주차면_충전기_비율'] = round(sim_parking / (sim_charger + 1), 2)
    sim_dict['점포당_평균_생활인구'] = round(sim_pop / (sim_dict.get('상권_총_점포수', 0) + 1), 2)
    sim_dict['전기차_대비_충전기_로그비율'] = round(np.log1p(sim_ev) / (np.log1p(sim_charger) + 1), 4)
    sim_dict['급속_비율'] = round(sim_dict.get('급속_충전기수', 0) / (sim_charger + 1), 2)
    sim_dict['완속_비율'] = round(sim_dict.get('완속_충전기수', 0) / (sim_charger + 1), 2)
    sim_dict['전기차_급속_비율'] = round(sim_ev / (sim_dict.get('급속_충전기수', 0) + 1), 2)
    sim_dict['전기차_완속_비율'] = round(sim_ev / (sim_dict.get('완속_충전기수', 0) + 1), 2)
    sim_dict['주차면_급속_비율'] = round(sim_parking / (sim_dict.get('급속_충전기수', 0) + 1), 2)
    sim_dict['주차면_완속_비율'] = round(sim_parking / (sim_dict.get('완속_충전기수', 0) + 1), 2)
    sim_dict['교통량_대비_충전기비율'] = round(sim_dict.get('도심_교통량', 0) / (sim_charger + 1), 2)
    sim_dict['아파트_대비_충전기비율'] = round(sim_dict.get('추정_아파트수', 0) / (sim_charger + 1), 2)

    sim_features = pd.DataFrame([sim_dict])
    if model is None: # Dual mode
        log_fast = model_fast.predict(sim_features[FEATURE_COLS])[0]
        log_slow = model_slow.predict(sim_features[FEATURE_COLS])[0]
        sim_pred = np.expm1(log_fast) + np.expm1(log_slow)
    else:
        log_sim_pred = model.predict(sim_features[FEATURE_COLS])[0]
        sim_pred = np.expm1(log_sim_pred)
    sim_hotspot = sim_pred / (sim_charger + 1)

    # 등급 판정
    if sim_hotspot >= q75:
        grade, badge_cls, emoji = "최우선 시급 지역 (핫스팟)", "badge-critical", "🔴"
        advice = "충전 인프라가 심각하게 부족합니다. 즉각적인 충전소 신규 설치가 필요한 최우선 지역입니다."
    elif sim_hotspot >= q25:
        grade, badge_cls, emoji = "잠재적 정체 구역", "badge-warning", "🟡"
        advice = "현재는 수용 가능하나, 전기차 증가 추세를 고려하면 중기적 충전 인프라 확충이 권장됩니다."
    else:
        grade, badge_cls, emoji = "여유 구역", "badge-safe", "🟢"
        advice = "현재 충전 인프라가 충분한 상태입니다. 다른 시급 지역에 투자를 우선하는 것이 효율적입니다."

    st.markdown("---")
    st.markdown("#### 📊 시뮬레이션 예측 결과")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">예상 요구 충전량</div>
            <div class="value" style="color:#ff6b6b">{sim_pred:,.0f}</div>
            <div class="unit">kWh</div>
        </div>""", unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">핫스팟 점수</div>
            <div class="value" style="color:#ffd93d">{sim_hotspot:,.1f}</div>
            <div class="unit">점</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="margin-top:0.8rem">
        <div class="hotspot-badge {badge_cls}">{emoji} {grade}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="info-panel" style="margin-top:0.8rem">
        <h4>💡 비즈니스 제언</h4>
        <p style="color:#e2e8f0;font-size:0.85rem;line-height:1.6">{advice}</p>
    </div>""", unsafe_allow_html=True)

# ─── 하단: Top 10 핫스팟 ───
st.markdown("---")
st.markdown("#### 🔥 서울시 Top 10 인프라 정체 핫스팟 행정동")

top10 = df.nlargest(10, '핫스팟_점수')
rows_html = ""
for i, (_, r) in enumerate(top10.iterrows(), 1):
    color = "#ff416c" if r['핫스팟_점수'] >= q75 else "#ffd200"
    rows_html += f"""<tr>
        <td style="color:{color};font-weight:700">{i}</td>
        <td>{r['자치구']}</td><td>{r['행정동']}</td>
        <td>{r['인구수']:,.0f}</td><td>{r['추정_전기차수']:,.0f}</td>
        <td>{r['기존_충전기수']:,.0f}</td><td>{r['예측_충전량']:,.0f}</td>
        <td style="color:{color};font-weight:700">{r['핫스팟_점수']:,.1f}</td>
    </tr>"""

st.markdown(f"""<div class="info-panel">
<table class="rank-table">
<thead><tr><th>#</th><th>자치구</th><th>행정동</th><th>인구수</th><th>전기차수</th><th>충전기수</th><th>예측충전량</th><th>핫스팟점수</th></tr></thead>
<tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)

# ─── 구별 요약 차트 ───
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 📈 자치구별 평균 핫스팟 점수")
    gu_avg = df.groupby('자치구')['핫스팟_점수'].mean().sort_values(ascending=False).reset_index()
    st.bar_chart(gu_avg.set_index('자치구'), height=350, color="#ff416c")
with c2:
    st.markdown("#### 📈 자치구별 전기차 대비 충전기 비율")
    gu_ratio = df.groupby('자치구')['전기차_충전기_비율'].mean().sort_values(ascending=False).reset_index()
    st.bar_chart(gu_ratio.set_index('자치구'), height=350, color="#00d4ff")
