#!/usr/bin/env python3
"""
[Task 1] 서울시 EV 충전 인프라 분석 - 데이터 전처리 파이프라인
==========================================================
5개 원천 데이터를 행정동 단위로 집계·융합하여 ML 학습용 CSV 생성

출력: seoul_ev_ml_dataset.csv
"""
import pandas as pd
import numpy as np
import json
import re
import warnings
import os

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def normalize_dong_name(name: str) -> str:
    """행정동명 정규화 (특수문자 통일)"""
    name = name.strip()
    name = name.replace('·', '.').replace('ㆍ', '.')
    name = name.replace('제', '')
    return name


def extract_gu_dong_from_address(addr: str):
    """
    주소 문자열에서 (자치구, 법정동) 추출
    예: '서울특별시 영등포구 경인로 878' → ('영등포구', None)  (도로명)
    예: '서울특별시 송파구 마천동 ...'   → ('송파구', '마천동')
    """
    if not isinstance(addr, str):
        return None, None
    
    # 자치구 추출
    gu_match = re.search(r'([\w]+구)\s', addr)
    gu = gu_match.group(1) if gu_match else None
    
    # 법정동 추출 (구 이후, ~동/~가 패턴)
    dong = None
    if gu:
        after_gu = addr.split(gu)[-1].strip()
        dong_match = re.search(r'([\w]+[동가])\b', after_gu)
        if dong_match:
            candidate = dong_match.group(1)
            # 도로명(~로, ~길)이 아닌 실제 동 이름인지 확인
            if not re.search(r'[로길]$', candidate):
                dong = candidate
    
    return gu, dong


# ─────────────────────────────────────────────
# Step 0: 행정동 마스터 테이블 구축
# ─────────────────────────────────────────────
print("=" * 60)
print("[Step 0] 행정동 마스터 테이블 구축...")
print("=" * 60)

# 중심점 데이터: 8자리 코드 ↔ 행정동명 ↔ 자치구명
df_center = pd.read_csv(
    os.path.join(DATA_DIR, '서울시_행정동_중심점_2017.csv'),
    encoding='cp949',
    dtype={'코드': str}
)
df_center = df_center.rename(columns={
    '코드': '행정동코드', '시군구명': '자치구', '읍면동명': '행정동'
})
df_center['행정동_정규화'] = df_center['행정동'].apply(normalize_dong_name)

# GeoJSON: 7자리 adm_cd ↔ 동명  → 8자리 코드 매핑
with open(os.path.join(DATA_DIR, '서울_행정동_경계_2017.geojson'), 'r', encoding='utf-8') as f:
    geojson = json.load(f)

center_lookup = {}
for _, row in df_center.iterrows():
    key = (row['자치구'], row['행정동_정규화'])
    center_lookup[key] = row['행정동코드']

geo_to_pop_code = {}
for feat in geojson['features']:
    adm_cd = feat['properties']['adm_cd']
    parts = feat['properties']['adm_nm'].split()
    gu = parts[1]
    dong = normalize_dong_name(' '.join(parts[2:]))
    key = (gu, dong)
    if key in center_lookup:
        geo_to_pop_code[adm_cd] = center_lookup[key]

# 마스터 테이블
master = df_center[['행정동코드', '자치구', '행정동']].copy()
print(f"  → 행정동 마스터: {len(master)}개 동")


# ─────────────────────────────────────────────
# Step 1: 동별 인구수 집계
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 1] 동별 인구수 집계...")
print("=" * 60)

df_pop = pd.read_csv(
    os.path.join(DATA_DIR, 'LOCAL_PEOPLE_DONG_202605.csv'),
    encoding='utf-8-sig',
    index_col=False,
    usecols=['행정동코드', '총생활인구수'],
    dtype={'행정동코드': str}
)
df_pop = df_pop[df_pop['행정동코드'].str.startswith('11')]
pop_by_dong = df_pop.groupby('행정동코드')['총생활인구수'].mean().reset_index()
pop_by_dong = pop_by_dong.rename(columns={'총생활인구수': '인구수'})
pop_by_dong['인구수'] = pop_by_dong['인구수'].round(1)
print(f"  → 인구 데이터 집계 완료: {len(pop_by_dong)}개 동")

# 마스터에 병합
master = master.merge(pop_by_dong, on='행정동코드', how='left')


# ─────────────────────────────────────────────
# Step 2: 구별 전기차 등록대수 추출 → 동별 추정
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 2] 구별 전기차 등록대수 → 동별 추정...")
print("=" * 60)

df_ev_raw = pd.read_excel(
    os.path.join(DATA_DIR, '서울시 자치구별 연료별 차종별 용도별 등록현황(26년5월).xlsx'),
    header=None
)

# 전기차 행 추출 (연료='전기')
ev_records = []
current_gu = None
for i in range(len(df_ev_raw)):
    gu_val = df_ev_raw.iloc[i, 0]
    fuel = df_ev_raw.iloc[i, 3]
    
    if pd.notna(gu_val) and isinstance(gu_val, str) and gu_val.endswith('구'):
        current_gu = gu_val
    
    if fuel == '전기' and current_gu is not None:
        total = df_ev_raw.iloc[i, 15]  # '계' 열
        if pd.notna(total):
            ev_records.append({'자치구': current_gu, '전기차수': int(total)})

df_ev_gu = pd.DataFrame(ev_records)
df_ev_gu = df_ev_gu.groupby('자치구')['전기차수'].sum().reset_index()
print(f"  → 전기차 등록: {len(df_ev_gu)}개 구, 총 {df_ev_gu['전기차수'].sum():,}대")

# 구별 인구 합계 계산
gu_pop = master.groupby('자치구')['인구수'].sum().reset_index()
gu_pop = gu_pop.rename(columns={'인구수': '구_총인구'})

# 동별 추정 전기차수 = 구_전기차수 × (동_인구 / 구_총인구)
master = master.merge(gu_pop, on='자치구', how='left')
master = master.merge(df_ev_gu, on='자치구', how='left')
master['전기차수'] = master['전기차수'].fillna(0)
master['구_총인구'] = master['구_총인구'].fillna(1)
master['추정_전기차수'] = (
    master['전기차수'] * (master['인구수'] / master['구_총인구'])
).round(1)
master = master.drop(columns=['구_총인구', '전기차수'])
print(f"  → 동별 추정 전기차수 계산 완료")


# ─────────────────────────────────────────────
# Step 3: 법정동 ↔ 행정동 매핑 테이블
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 3] 법정동 ↔ 행정동 매핑 테이블 구축...")
print("=" * 60)

df_mapping = pd.read_csv(
    os.path.join(DATA_DIR, '국가데이터처_법정동 연계정보_20250602.csv'),
    encoding='cp949'
)
df_mapping_seoul = df_mapping[df_mapping['시도명'] == '서울특별시'].copy()

# 법정동명 → 행정동코드 (8자리) 매핑
# 행정동코드 10자리 → 8자리 변환
df_mapping_seoul['행정동코드_8'] = df_mapping_seoul['행정동코드'].astype(str).str[:8]

# 법정동명 + 자치구 → 행정동코드 매핑 딕셔너리
beopjeong_to_haengjeong = {}
for _, row in df_mapping_seoul.iterrows():
    key = (row['시군구명'], row['법정동명'])
    beopjeong_to_haengjeong[key] = row['행정동코드_8']

print(f"  → 법정동→행정동 매핑: {len(beopjeong_to_haengjeong)}개 항목")


def address_to_dong_code(addr, fallback_gu_map=None):
    """
    주소 → 행정동코드 매핑
    1차: 법정동명으로 직접 매핑
    2차: 자치구 fallback
    """
    gu, dong = extract_gu_dong_from_address(addr)
    if gu is None:
        return None, None
    
    # 1차: 법정동 직접 매핑
    if dong:
        key = (gu, dong)
        if key in beopjeong_to_haengjeong:
            return beopjeong_to_haengjeong[key], gu
        
        # 동 이름 변형 시도 (숫자 포함된 동명)
        # 예: '당산동3가' → '당산동3가', '신길동' 등
        for suffix in ['', '동']:
            trial_key = (gu, dong.rstrip('동가') + suffix)
            if trial_key in beopjeong_to_haengjeong:
                return beopjeong_to_haengjeong[trial_key], gu
    
    # 2차: 자치구 fallback
    return None, gu


# ─────────────────────────────────────────────
# Step 4: 충전소 시설 정보 (동별 기존 충전기수)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 4] 충전소 시설 정보 집계...")
print("=" * 60)

df_station = pd.read_excel(
    os.path.join(DATA_DIR, '충전소 리스트 (2).xlsx'),
    header=2
)

# 서울 필터링
df_station_seoul = df_station[
    df_station['지역'].str.contains('서울', na=False)
].copy()
print(f"  → 서울 충전기 레코드: {len(df_station_seoul):,}건")

# 주소에서 행정동코드 매핑
df_station_seoul['행정동코드'] = None
df_station_seoul['매핑구'] = None

for idx in df_station_seoul.index:
    addr = df_station_seoul.at[idx, '주소']
    code, gu = address_to_dong_code(addr)
    df_station_seoul.at[idx, '행정동코드'] = code
    df_station_seoul.at[idx, '매핑구'] = gu

direct_match = df_station_seoul['행정동코드'].notna().sum()
print(f"  → 직접 매핑 성공: {direct_match:,}건 ({direct_match/len(df_station_seoul)*100:.1f}%)")

df_station_seoul['급속여부'] = df_station_seoul['충전용량'].astype(str).str.contains('급속', na=False)

# 자치구 fallback: 매핑 실패 건은 해당 구의 동들에 균등 배분
gu_fallback = df_station_seoul[df_station_seoul['행정동코드'].isna()].copy()
if len(gu_fallback) > 0:
    gu_fallback_counts = gu_fallback.groupby('매핑구').size().reset_index(name='fallback_충전기수')
    
    # 구별 행정동 수
    dong_per_gu = master.groupby('자치구')['행정동코드'].count().reset_index(name='동수')
    gu_fallback_counts = gu_fallback_counts.merge(
        dong_per_gu, left_on='매핑구', right_on='자치구', how='left'
    )
    gu_fallback_counts['동당_추가'] = (
        gu_fallback_counts['fallback_충전기수'] / gu_fallback_counts['동수']
    ).round(1)
    
    print(f"  → 자치구 fallback 배분: {len(gu_fallback):,}건")


# 동별 충전기 수 집계 (직접 매핑분)
charger_by_dong = (
    df_station_seoul[df_station_seoul['행정동코드'].notna()]
    .groupby('행정동코드')
    .agg(
        기존_충전기수=('충전기ID', 'size'),
        급속_충전기수=('급속여부', 'sum')
    )
    .reset_index()
)
charger_by_dong['완속_충전기수'] = charger_by_dong['기존_충전기수'] - charger_by_dong['급속_충전기수']

# fallback 추가
if len(gu_fallback) > 0:
    gu_fallback_counts = gu_fallback.groupby('매핑구').agg(
        fallback_총=('충전기ID', 'size'),
        fallback_급속=('급속여부', 'sum')
    ).reset_index()
    
    # 구별 행정동 수
    dong_per_gu = master.groupby('자치구')['행정동코드'].count().reset_index(name='동수')
    gu_fallback_counts = gu_fallback_counts.merge(dong_per_gu, left_on='매핑구', right_on='자치구', how='left')
    gu_fallback_counts['동당_추가'] = gu_fallback_counts['fallback_총'] / gu_fallback_counts['동수']
    gu_fallback_counts['동당_추가_급속'] = gu_fallback_counts['fallback_급속'] / gu_fallback_counts['동수']
    
    fallback_records = []
    for _, row in gu_fallback_counts.iterrows():
        gu = row['매핑구']
        if pd.isna(gu):
            continue
        dongs_in_gu = master[master['자치구'] == gu]['행정동코드'].tolist()
        per_dong = row['동당_추가']
        per_dong_fast = row['동당_추가_급속']
        for d in dongs_in_gu:
            fallback_records.append({
                '행정동코드': d,
                '기존_충전기수': per_dong,
                '급속_충전기수': per_dong_fast,
                '완속_충전기수': per_dong - per_dong_fast
            })
    
    if fallback_records:
        df_fallback = pd.DataFrame(fallback_records)
        charger_by_dong = pd.concat([charger_by_dong, df_fallback])
        charger_by_dong = charger_by_dong.groupby('행정동코드')[['기존_충전기수', '급속_충전기수', '완속_충전기수']].sum().reset_index()

charger_by_dong[['기존_충전기수', '급속_충전기수', '완속_충전기수']] = charger_by_dong[['기존_충전기수', '급속_충전기수', '완속_충전기수']].round(0).astype(int)
master = master.merge(charger_by_dong, on='행정동코드', how='left')
print(f"  → 총 충전기 집계: {charger_by_dong['기존_충전기수'].sum():,}기 (급속: {charger_by_dong['급속_충전기수'].sum():,}, 완속: {charger_by_dong['완속_충전기수'].sum():,})")


# ─────────────────────────────────────────────
# Step 5: 충전량 데이터 (Target 변수)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 5] 충전량 데이터 집계 (Target: 총_충전량)...")
print("=" * 60)

df_usage = pd.read_excel(
    os.path.join(DATA_DIR, '한국전력공사_서울시 전기차 충전소 충전량_20220331.xlsx')
)
print(f"  → 충전 이력: {len(df_usage):,}건")

# 주소 → 행정동코드 매핑
addr_cache = {}
usage_codes = []
usage_gus = []
for addr in df_usage['주소']:
    if addr in addr_cache:
        code, gu = addr_cache[addr]
    else:
        code, gu = address_to_dong_code(addr)
        addr_cache[addr] = (code, gu)
    usage_codes.append(code)
    usage_gus.append(gu)

df_usage['행정동코드'] = usage_codes
df_usage['매핑구'] = usage_gus

direct_usage = df_usage['행정동코드'].notna().sum()
print(f"  → 직접 매핑 성공: {direct_usage:,}건 ({direct_usage/len(df_usage)*100:.1f}%)")

# 직접 매핑된 데이터로 동별 집계
usage_direct = df_usage[df_usage['행정동코드'].notna()].copy()
usage_by_dong = usage_direct.groupby('행정동코드').agg(
    총_충전량=('충전량', 'sum'),
    평균_충전분=('충전분', 'mean'),
    충전_건수=('충전량', 'count')
).reset_index()

# 급속/완속 분리
fast_usage = usage_direct[usage_direct['충전구분'] == '급속'].groupby('행정동코드')['충전량'].sum().reset_index(name='급속_충전량')
slow_usage = usage_direct[usage_direct['충전구분'] == '완속'].groupby('행정동코드')['충전량'].sum().reset_index(name='완속_충전량')
usage_by_dong = usage_by_dong.merge(fast_usage, on='행정동코드', how='left').merge(slow_usage, on='행정동코드', how='left').fillna(0)

usage_by_dong['총_충전량'] = usage_by_dong['총_충전량'].round(2)
usage_by_dong['급속_충전량'] = usage_by_dong['급속_충전량'].round(2)
usage_by_dong['완속_충전량'] = usage_by_dong['완속_충전량'].round(2)
usage_by_dong['평균_충전분'] = usage_by_dong['평균_충전분'].round(1)

# 자치구 fallback: 미매핑분을 구 내 동에 균등 배분
usage_fallback = df_usage[df_usage['행정동코드'].isna() & df_usage['매핑구'].notna()]
if len(usage_fallback) > 0:
    fb_by_gu = usage_fallback.groupby('매핑구').agg(
        fb_총_충전량=('충전량', 'sum'),
        fb_충전분=('충전분', 'mean'),
        fb_건수=('충전량', 'count')
    ).reset_index()
    
    fb_fast = usage_fallback[usage_fallback['충전구분'] == '급속'].groupby('매핑구')['충전량'].sum().reset_index(name='fb_급속_충전량')
    fb_slow = usage_fallback[usage_fallback['충전구분'] == '완속'].groupby('매핑구')['충전량'].sum().reset_index(name='fb_완속_충전량')
    fb_by_gu = fb_by_gu.merge(fb_fast, on='매핑구', how='left').merge(fb_slow, on='매핑구', how='left').fillna(0)
    
    dong_per_gu = master.groupby('자치구')['행정동코드'].count().reset_index(name='동수')
    fb_by_gu = fb_by_gu.merge(dong_per_gu, left_on='매핑구', right_on='자치구', how='left')
    
    fb_records = []
    for _, row in fb_by_gu.iterrows():
        gu = row['매핑구']
        dongs = master[master['자치구'] == gu]['행정동코드'].tolist()
        n = len(dongs)
        if n == 0:
            continue
        for d in dongs:
            fb_records.append({
                '행정동코드': d,
                '총_충전량': round(row['fb_총_충전량'] / n, 2),
                '급속_충전량': round(row['fb_급속_충전량'] / n, 2),
                '완속_충전량': round(row['fb_완속_충전량'] / n, 2),
                '평균_충전분': round(row['fb_충전분'], 1),
                '충전_건수': int(row['fb_건수'] / n)
            })
    
    if fb_records:
        df_fb = pd.DataFrame(fb_records)
        usage_by_dong = pd.concat([usage_by_dong, df_fb])
        usage_by_dong = usage_by_dong.groupby('행정동코드').agg({
            '총_충전량': 'sum',
            '급속_충전량': 'sum',
            '완속_충전량': 'sum',
            '평균_충전분': 'mean',
            '충전_건수': 'sum'
        }).reset_index()
    
    print(f"  → 자치구 fallback 배분: {len(usage_fallback):,}건")

master = master.merge(usage_by_dong, on='행정동코드', how='left')
print(f"  → 충전량 집계 완료")


# ─────────────────────────────────────────────
# Step 6: 공영주차장 면수 집계
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 6] 공영주차장 면수 집계...")
print("=" * 60)

df_parking = pd.read_csv(
    os.path.join(DATA_DIR, '서울시 공영주차장 안내 정보.csv'),
    encoding='cp949'
)
print(f"  → 공영주차장: {len(df_parking):,}건")

# 주소 → 행정동코드 매핑
park_codes = []
park_gus = []
for addr in df_parking['주소']:
    code, gu = address_to_dong_code(addr)
    park_codes.append(code)
    park_gus.append(gu)

df_parking['행정동코드'] = park_codes
df_parking['매핑구'] = park_gus

# 동별 총 주차면수 집계
park_direct = df_parking[df_parking['행정동코드'].notna()]
parking_by_dong = park_direct.groupby('행정동코드')['총 주차면'].sum().reset_index()
parking_by_dong = parking_by_dong.rename(columns={'총 주차면': '공영주차장_면수'})

# fallback
park_fb = df_parking[df_parking['행정동코드'].isna() & df_parking['매핑구'].notna()]
if len(park_fb) > 0:
    fb_park_gu = park_fb.groupby('매핑구')['총 주차면'].sum().reset_index()
    dong_per_gu = master.groupby('자치구')['행정동코드'].count().reset_index(name='동수')
    fb_park_gu = fb_park_gu.merge(dong_per_gu, left_on='매핑구', right_on='자치구', how='left')
    
    fb_records = []
    for _, row in fb_park_gu.iterrows():
        dongs = master[master['자치구'] == row['매핑구']]['행정동코드'].tolist()
        n = len(dongs)
        if n == 0:
            continue
        for d in dongs:
            fb_records.append({
                '행정동코드': d,
                '공영주차장_면수': round(row['총 주차면'] / n, 0)
            })
    
    if fb_records:
        df_fb = pd.DataFrame(fb_records)
        parking_by_dong = pd.concat([parking_by_dong, df_fb])
        parking_by_dong = parking_by_dong.groupby('행정동코드')['공영주차장_면수'].sum().reset_index()

parking_by_dong['공영주차장_면수'] = parking_by_dong['공영주차장_면수'].astype(int)
master = master.merge(parking_by_dong, on='행정동코드', how='left')
print(f"  → 주차장 집계 완료")

# ─────────────────────────────────────────────
# Step 6.5: 대형 쇼핑시설물 수 집계
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 6.5] 쇼핑시설물 수 집계...")
print("=" * 60)

df_shopping = pd.read_csv(
    os.path.join(DATA_DIR, '서울시 쇼핑시설물 정보 (한국어_2015년).csv'),
    encoding='cp949'
)
print(f"  → 대형 쇼핑시설물: {len(df_shopping):,}건")

shopping_codes = []
for _, row in df_shopping.iterrows():
    gu = row['행정 구']
    dong = row['행정 동']
    if pd.isna(gu) or pd.isna(dong):
        continue
    norm_dong = normalize_dong_name(dong)
    key = (gu, norm_dong)
    
    if key in center_lookup:
        shopping_codes.append(center_lookup[key])
    else:
        # Fallback to similar name if slight mismatch
        for k in center_lookup.keys():
            if k[0] == gu and (norm_dong in k[1] or k[1] in norm_dong):
                shopping_codes.append(center_lookup[k])
                break

shopping_by_dong = pd.Series(shopping_codes).value_counts().reset_index()
shopping_by_dong.columns = ['행정동코드', '쇼핑시설물_수']
master = master.merge(shopping_by_dong, on='행정동코드', how='left')
print(f"  → 쇼핑시설물 집계 완료 (매핑성공: {len(shopping_codes)}건)")


# ─────────────────────────────────────────────
# Step 6.6: 주택종류별 주택수 (아파트 비율) 집계
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 6.6] 주택종류별 주택수 및 아파트 비율 집계...")
print("=" * 60)

try:
    df_housing = pd.read_csv(os.path.join(DATA_DIR, '101_DT_1JU1501_20260708121347.csv'), encoding='cp949')
except:
    df_housing = pd.read_csv(os.path.join(DATA_DIR, '101_DT_1JU1501_20260708121347.csv'), encoding='utf-8-sig')

df_housing = df_housing.iloc[:234]

df_house = df_housing[df_housing['항목'] == '주택'][['행정구역별(읍면동)', '2024 년']].rename(columns={'행정구역별(읍면동)': '자치구', '2024 년': '구_총주택수'})
df_apt = df_housing[df_housing['항목'] == '아파트'][['행정구역별(읍면동)', '2024 년']].rename(columns={'행정구역별(읍면동)': '자치구', '2024 년': '구_아파트수'})

df_house_apt = pd.merge(df_house, df_apt, on='자치구', how='outer')
df_house_apt['구_총주택수'] = pd.to_numeric(df_house_apt['구_총주택수'], errors='coerce').fillna(0)
df_house_apt['구_아파트수'] = pd.to_numeric(df_house_apt['구_아파트수'], errors='coerce').fillna(0)

master = master.merge(df_house_apt, on='자치구', how='left')

gu_pop_temp = master.groupby('자치구')['인구수'].sum().reset_index().rename(columns={'인구수': '구_총인구_temp'})
master = master.merge(gu_pop_temp, on='자치구', how='left')
master['구_총인구_temp'] = master['구_총인구_temp'].fillna(1).replace(0, 1)

master['추정_주택수'] = (master['구_총주택수'] * (master['인구수'] / master['구_총인구_temp'])).round(1)
master['추정_아파트수'] = (master['구_아파트수'] * (master['인구수'] / master['구_총인구_temp'])).round(1)
master['아파트_비율'] = (master['추정_아파트수'] / (master['추정_주택수'] + 1)).round(2)

master = master.drop(columns=['구_총주택수', '구_아파트수', '구_총인구_temp'])
print("  → 주택수 및 아파트 비율 산출 완료")

# ─────────────────────────────────────────────
# Step 6.7: 상권분석서비스 점포 수
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 6.7] 상권분석 점포 수 집계...")
print("=" * 60)

try:
    df_store = pd.read_csv(os.path.join(DATA_DIR, '서울시 상권분석서비스(점포-행정동).csv'), encoding='cp949')
except:
    df_store = pd.read_csv(os.path.join(DATA_DIR, '서울시 상권분석서비스(점포-행정동).csv'), encoding='utf-8-sig')

df_store = df_store[df_store['기준_년분기_코드'] == 20261]
df_store['행정동_코드'] = df_store['행정동_코드'].astype(str).str[:8]
store_by_dong = df_store.groupby('행정동_코드')['전체_점포_수'].sum().reset_index()
store_by_dong.columns = ['행정동코드', '상권_총_점포수']

master = master.merge(store_by_dong, on='행정동코드', how='left')
print("  → 점포 수 집계 완료")

# ─────────────────────────────────────────────
# Step 6.8: 지하철 승하차 데이터
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 6.8] 지하철 승하차 수 집계...")
print("=" * 60)

try:
    df_subway = pd.read_csv(os.path.join(DATA_DIR, '서울시 행정동별 지하철 총 승차 승객수 정보.csv'), encoding='cp949')
except:
    df_subway = pd.read_csv(os.path.join(DATA_DIR, '서울시 행정동별 지하철 총 승차 승객수 정보.csv'), encoding='utf-8-sig')

code_map = {
    '1101': '1111', '1102': '1114', '1103': '1117', '1104': '1120', '1105': '1121',
    '1106': '1123', '1107': '1126', '1108': '1129', '1109': '1130', '1110': '1132',
    '1111': '1135', '1112': '1138', '1113': '1141', '1114': '1144', '1115': '1147',
    '1116': '1150', '1117': '1153', '1118': '1154', '1119': '1156', '1120': '1159',
    '1121': '1162', '1122': '1165', '1123': '1168', '1124': '1171', '1125': '1174'
}

df_subway['행정동_ID_str'] = df_subway['행정동_ID'].astype(str).str[:8]
df_subway['앞4자리'] = df_subway['행정동_ID_str'].str[:4]
df_subway['뒤4자리'] = df_subway['행정동_ID_str'].str[4:]
df_subway['변환_앞4자리'] = df_subway['앞4자리'].map(code_map)
df_subway['행정동코드_수정'] = df_subway['변환_앞4자리'].fillna(df_subway['앞4자리']) + df_subway['뒤4자리']

# 자치구 이름 매핑 (원천 데이터 기반 구 평균을 구하기 위함)
gu_name_map = {
    '1101': '종로구', '1102': '중구', '1103': '용산구', '1104': '성동구', '1105': '광진구',
    '1106': '동대문구', '1107': '중랑구', '1108': '성북구', '1109': '강북구', '1110': '도봉구',
    '1111': '노원구', '1112': '은평구', '1113': '서대문구', '1114': '마포구', '1115': '양천구',
    '1116': '강서구', '1117': '구로구', '1118': '금천구', '1119': '영등포구', '1120': '동작구',
    '1121': '관악구', '1122': '서초구', '1123': '강남구', '1124': '송파구', '1125': '강동구'
}
df_subway['자치구'] = df_subway['앞4자리'].map(gu_name_map)

# 1. 동 단위 직접 매핑 (218개 동 매핑)
subway_by_dong = df_subway.groupby('행정동코드_수정')['지하철_승객_수'].mean().reset_index()
subway_by_dong.columns = ['행정동코드', '지하철_총_승하차수']

master = master.merge(subway_by_dong, on='행정동코드', how='left')

# 2. 지하철역 없는 동(NaN)에 대해 원천 데이터 기준 자치구(Gu) 평균으로 Fallback (인구 가중치 반영)
gu_subway_mean = df_subway.groupby('자치구')['지하철_승객_수'].mean().reset_index(name='구별_지하철_평균')
master = master.merge(gu_subway_mean, on='자치구', how='left')

# 단일화(균등) 배분 문제를 해결하기 위해 동별 인구 가중치 적용
master['구_평균_인구수'] = master.groupby('자치구')['인구수'].transform('mean')
master['인구_가중치'] = master['인구수'] / master['구_평균_인구수']

master['지하철_총_승하차수'] = master['지하철_총_승하차수'].fillna(master['구별_지하철_평균'] * master['인구_가중치']).round(0)
master = master.drop(columns=['구별_지하철_평균', '구_평균_인구수', '인구_가중치'])
print("  → 지하철 승하차 수 집계 완료 (통계청->행안부 코드 변환 및 인구 비례 자치구 Fallback 적용)")


# ─────────────────────────────────────────────
# Step 6.9: 도심지점별 교통량 데이터 매핑
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 6.9] 도심지점별 교통량 매핑...")
print("=" * 60)

try:
    df_traffic = pd.read_csv(os.path.join(DATA_DIR, '도심지점별+교통량_20260708152259.csv'))
    gu_map_traffic = {
        '성산로(금화터널)': '서대문구', '사직로(사직터널)': '종로구', '자하문로(자하문터널)': '종로구',
        '대사관로(삼청터널)': '성북구', '율곡로(안국역)': '종로구', '창경궁로(서울여자대학교)': '종로구',
        '대학로(한국방송통신대학교)': '종로구', '종로(동묘앞역)': '종로구', '퇴계로(신당역)': '중구',
        '동호로(장충체육관)': '중구', '장충단로(장충단공원)': '중구', '퇴계로(회현역)': '중구',
        '세종대로(서울역)': '중구', '새문안로(서울역사박물관)': '종로구', '종로(종로3가역)': '종로구',
        '서소문로(시청역)': '중구', '세종대로(시청역2)': '중구', '을지로(을지로3가역)': '중구',
        '칠패로(숭례문)': '중구', '남산1호터널': '용산구', '남산2호터널': '용산구',
        '남산3호터널': '용산구', '소월로(회현역)': '중구', '소파로(숭의여자대학교)': '중구'
    }
    df_traffic['자치구'] = df_traffic['도심지점별(2)'].map(gu_map_traffic)
    df_traffic['도심_교통량_수치'] = pd.to_numeric(df_traffic['2025'], errors='coerce')
    gu_traffic = df_traffic.groupby('자치구')['도심_교통량_수치'].sum().reset_index(name='도심_교통량')
    master = master.merge(gu_traffic, on='자치구', how='left')
    master['도심_교통량'] = master['도심_교통량'].fillna(0)
    print("  → 도심 교통량 매핑 완료")
except Exception as e:
    print(f"  → 교통량 매핑 실패: {e}")
    master['도심_교통량'] = 0# ─────────────────────────────────────────────
# Step 7: Feature Engineering + 최종 저장
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 7] Feature Engineering 및 최종 데이터셋 저장...")
print("=" * 60)

# 최종 데이터 정리 및 결측치 처리
master = master.fillna(0)
numeric_cols = ['인구수', '추정_전기차수', '기존_충전기수', '급속_충전기수', '완속_충전기수', 
             '도심_교통량', '총_충전량', '급속_충전량', '완속_충전량', '평균_충전분', '충전_건수']
for c in numeric_cols:
    if c in master.columns:
        master[c] = pd.to_numeric(master[c], errors='coerce').fillna(0)

# 로그 스케일 타겟 생성
master['로그_총_충전량'] = np.log1p(master['총_충전량']).round(4)
master['로그_급속_충전량'] = np.log1p(master['급속_충전량']).round(4)
master['로그_완속_충전량'] = np.log1p(master['완속_충전량']).round(4)

# 1-2. 독립변수 사전 로그 변환 (학습 안정성)
master['로그_공영주차장_면수'] = np.log1p(master['공영주차장_면수']).round(4)
master['로그_상권_총_점포수'] = np.log1p(master['상권_총_점포수']).round(4)
master['로그_지하철_총_승하차수'] = np.log1p(master['지하철_총_승하차수']).round(4)
master['로그_기존_충전기수'] = np.log1p(master['기존_충전기수']).round(4)
master['로그_급속_충전기수'] = np.log1p(master['급속_충전기수']).round(4)
master['로그_완속_충전기수'] = np.log1p(master['완속_충전기수']).round(4)
master['로그_도심_교통량'] = np.log1p(master['도심_교통량']).round(4)

# 2. 자치구 레벨 거시적 공간 피처 (Macro Baseline)
master['구별_평균_총충전량'] = master.groupby('자치구')['총_충전량'].transform('mean').round(2)
master['구별_총_지하철승객수'] = master.groupby('자치구')['지하철_총_승하차수'].transform('sum').round(0)
master['구별_총_점포수'] = master.groupby('자치구')['상권_총_점포수'].transform('sum').round(0)

# 3. 모델 친화적 상호작용 피처 (Interaction Features)
master['주차면_대비_지하철승객수'] = (master['지하철_총_승하차수'] / (master['공영주차장_면수'] + 1)).round(2)
master['점포당_평균_생활인구'] = (master['인구수'] / (master['상권_총_점포수'] + 1)).round(2)
master['전기차_대비_충전기_로그비율'] = (np.log1p(master['추정_전기차수']) / (np.log1p(master['기존_충전기수']) + 1)).round(4)

# 기존 파생 변수 유지
master['충전기당_인구수'] = (master['인구수'] / (master['기존_충전기수'] + 1)).round(1)
master['전기차_충전기_비율'] = (master['추정_전기차수'] / (master['기존_충전기수'] + 1)).round(2)
master['주차면_충전기_비율'] = (master['공영주차장_면수'] / (master['기존_충전기수'] + 1)).round(2)

# 신규 추가 (급속/완속 관련 및 기타 상호작용)
master['급속_비율'] = (master['급속_충전기수'] / (master['기존_충전기수'] + 1)).round(2)
master['완속_비율'] = (master['완속_충전기수'] / (master['기존_충전기수'] + 1)).round(2)
master['전기차_급속_비율'] = (master['추정_전기차수'] / (master['급속_충전기수'] + 1)).round(2)
master['전기차_완속_비율'] = (master['추정_전기차수'] / (master['완속_충전기수'] + 1)).round(2)
master['주차면_급속_비율'] = (master['공영주차장_면수'] / (master['급속_충전기수'] + 1)).round(2)
master['주차면_완속_비율'] = (master['공영주차장_면수'] / (master['완속_충전기수'] + 1)).round(2)
master['아파트_급속_비율'] = (master['추정_아파트수'] / (master['급속_충전기수'] + 1)).round(2)
master['아파트_완속_비율'] = (master['추정_아파트수'] / (master['완속_충전기수'] + 1)).round(2)
master['교통량_대비_충전기비율'] = (master['도심_교통량'] / (master['기존_충전기수'] + 1)).round(2)
master['아파트_대비_충전기비율'] = (master['추정_아파트수'] / (master['기존_충전기수'] + 1)).round(2)


# 4. 신규 변수 결측치 방어
new_derived_cols = [
    '로그_총_충전량', '로그_공영주차장_면수', '로그_상권_총_점포수', '로그_지하철_총_승하차수', '로그_기존_충전기수',
    '로그_급속_충전기수', '로그_완속_충전기수', '로그_도심_교통량',
    '구별_평균_총충전량', '구별_총_지하철승객수', '구별_총_점포수', 
    '주차면_대비_지하철승객수', '점포당_평균_생활인구', '전기차_대비_충전기_로그비율',
    '급속_비율', '완속_비율', '전기차_급속_비율', '전기차_완속_비율', '주차면_급속_비율', '주차면_완속_비율',
    '아파트_급속_비율', '아파트_완속_비율', '교통량_대비_충전기비율', '아파트_대비_충전기비율'
]
for col in new_derived_cols:
    master[col] = master[col].fillna(0)

# 5. 최종 컬럼 순서 정리
output_cols = [
    '자치구', '행정동', '행정동코드',
    '인구수', '추정_전기차수', '공영주차장_면수', '쇼핑시설물_수', '기존_충전기수',
    '급속_충전기수', '완속_충전기수', '도심_교통량',
    '추정_주택수', '추정_아파트수', '아파트_비율', '상권_총_점포수', '지하철_총_승하차수',
    '구별_평균_총충전량', '구별_총_지하철승객수', '구별_총_점포수',
    '로그_공영주차장_면수', '로그_상권_총_점포수', '로그_지하철_총_승하차수', '로그_기존_충전기수',
    '로그_급속_충전기수', '로그_완속_충전기수', '로그_도심_교통량',
    '충전기당_인구수', '전기차_충전기_비율', '주차면_충전기_비율',
    '주차면_대비_지하철승객수', '점포당_평균_생활인구', '전기차_대비_충전기_로그비율',
    '급속_비율', '완속_비율', '전기차_급속_비율', '전기차_완속_비율', '주차면_급속_비율', '주차면_완속_비율',
    '아파트_급속_비율', '아파트_완속_비율', '교통량_대비_충전기비율', '아파트_대비_충전기비율',
    '총_충전량', '급속_충전량', '완속_충전량', 
    '로그_총_충전량', '로그_급속_충전량', '로그_완속_충전량', 
    '평균_충전분', '충전_건수'
]
master = master[output_cols]

# 저장
output_path = os.path.join(DATA_DIR, 'seoul_ev_ml_dataset.csv')
master.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n  ✅ 저장 완료: {output_path}")
print(f"  → 행정동 수: {len(master)}")
print(f"  → 컬럼: {list(master.columns)}")
print(f"\n{'=' * 60}")
print("데이터셋 요약 통계:")
print("=" * 60)
print(master.describe().round(2).to_string())

print(f"\n\n상위 10개 행정동 (총_충전량 기준):")
print(master.nlargest(10, '총_충전량')[['자치구', '행정동', '인구수', '추정_전기차수', '기존_충전기수', '총_충전량']].to_string(index=False))
