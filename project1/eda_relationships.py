import pandas as pd
import numpy as np

# 1. 데이터 불러오기
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'

try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

print("=" * 70)
print("1. 수치형 변수 간 상관계수 (Correlation Matrix)")
print("=" * 70)
num_cols = ['product_amt', 'transaction_amt', 'product_period']
corr_matrix = df[num_cols].corr().round(3)
print(corr_matrix)

print("\n" + "=" * 70)
print("2. 연령대(age) x 투자성향(investment_propensity) 교차 분석 (비율 %)")
print("=" * 70)
crosstab_age_inv = pd.crosstab(df['age'], df['investment_propensity'], normalize='index') * 100
print(crosstab_age_inv.round(2))