import pandas as pd
import numpy as np

# 1. 데이터 불러오기
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'

try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

# 2. 명시적 결측치(NaN/Null) 집계
missing_df = pd.DataFrame({
    '명시적_결측수(NaN)': df.isnull().sum(),
    '명시적_결측비율(%)': (df.isnull().sum() / len(df) * 100).round(2)
})

# 3. 암묵적 결측치 집계 (정규표현식을 활용한 '포함 여부' 검사)
implicit_missing_counts = []
keywords = '미답변|미응답|없음|모름|기타'

for col in df.columns:
    # 모든 컬럼을 문자열로 변환하여 공백 제거 후 키워드 포함 검사
    col_str = df[col].astype(str).str.strip()
    
    # NaN이었던 것은 명시적 결측에 이미 들어갔으므로 암묵적 결측 집계에서는 제외 (중복 방지)
    is_implicit = col_str.str.contains(keywords, regex=True, na=False) & df[col].notnull()
    
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