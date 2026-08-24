import pandas as pd
import numpy as np

# 1. 데이터 불러오기
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'

try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

# 2. 수치형 변수 선정
num_cols = ['product_amt', 'transaction_amt', 'product_period']

outlier_summary = []

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # IQR 상/하한선 기준 이상치 개수 계산
    outliers_cnt = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    outliers_pct = round((outliers_cnt / len(df)) * 100, 2)
    
    outlier_summary.append({
        '컬럼명': col,
        '최솟값 (Min)': f"{df[col].min():,}",
        '중앙값 (Median)': f"{df[col].median():,}",
        '최댓값 (Max)': f"{df[col].max():,}",
        'IQR 상한선': f"{upper_bound:,.1f}",
        '이상치 수 (IQR 기준)': f"{outliers_cnt:,}건",
        '이상치 비율 (%)': f"{outliers_pct}%"
    })

outlier_df = pd.DataFrame(outlier_summary)

print("=" * 80)
print("수치형 변수 값의 범위 및 IQR 기준 이상치 집계 결과")
print("=" * 80)
print(outlier_df.to_string(index=False))