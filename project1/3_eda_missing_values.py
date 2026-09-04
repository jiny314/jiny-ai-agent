from pathlib import Path
import pandas as pd
import numpy as np

# 1. 동적 경로 설정
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
csv_path = BASE_DIR / "data" / "소비자.csv"

# 데이터 불러오기
try:
    df = pd.read_csv(csv_path)
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='cp949')

# 2. 명시적 결측치(NaN/Null) 집계
missing_df = pd.DataFrame({
    '명시적_결측수(NaN)': df.isnull().sum(),
    '명시적_결측비율(%)': (df.isnull().sum() / len(df) * 100).round(2)
})

# 3. 암묵적 결측치 집계 (공백 제거 후, 셀 값 전체가 키워드와 "완전히 일치"하는 경우만 카운트)
# 주의: str.contains()를 쓰면 "기타 상품"처럼 키워드를 부분 문자열로 포함하는
# 정상 카테고리 값까지 결측으로 오분류되므로, 반드시 전체 일치(exact match)로 검사해야 함
implicit_missing_counts = []
keywords = ['미답변', '미응답', '없음', '모름', '기타']

for col in df.columns:
    col_str = df[col].astype(str).str.replace(r'\s+', '', regex=True)
    is_implicit = col_str.isin(keywords) & df[col].notnull()
    cnt = is_implicit.sum()
    implicit_missing_counts.append(cnt)

missing_df['암묵적_결측수'] = implicit_missing_counts
missing_df['암묵적_결측비율(%)'] = (missing_df['암묵적_결측수'] / len(df) * 100).round(2)

# 4. 총 실질 결측 비율 계산
missing_df['총_실질_결측수'] = missing_df['명시적_결측수(NaN)'] + missing_df['암묵적_결측수']
missing_df['총_실질_결측비율(%)'] = (missing_df['총_실질_결측수'] / len(df) * 100).round(2)

# 5. 결과 정렬 및 출력
missing_df = missing_df.sort_values(by='총_실질_결측비율(%)', ascending=False)

print("=" * 75)
print("컬럼별 최종 결측치 및 암묵적 결측 집계 결과")
print("=" * 75)
print(missing_df)