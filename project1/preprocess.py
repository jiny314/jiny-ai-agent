import pandas as pd
import numpy as np

# 1. 데이터 불러오기
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'
try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

print("원본 데이터 크기:", df.shape)

# 2. 범주형 데이터 정제
# (1) 연령대 통합 ('70대' -> '70대 이상')
df['age'] = df['age'].replace({'70대': '70대 이상'})

# (2) 투자성향 결측치 처리 ('-' -> '미응답')
df['investment_propensity'] = df['investment_propensity'].replace({'-': '미응답'})

# 3. 고객(customer_id) 단위 피처 집계 (Customer Aggregation)
# 범주형 대표값(최빈값) 및 수치형 변수(총 금액, 평균 금액, 상품 개수 등) 집계
customer_df = df.groupby('customer_id').agg(
    age=('age', lambda x: x.mode()[0]),
    gender=('gender', lambda x: x.mode()[0]),
    income_bracket=('income_bracket', lambda x: x.mode()[0]),
    occupation_group=('occupation_group', lambda x: x.mode()[0]),
    investment_propensity=('investment_propensity', lambda x: x.mode()[0]),
    risk_grade=('risk_grade', lambda x: x.mode()[0]),
    total_product_amt=('product_amt', 'sum'),
    avg_product_amt=('product_amt', 'mean'),
    total_transaction_amt=('transaction_amt', 'sum'),
    product_count=('product_name', 'nunique'), # 가입한 서로 다른 상품 종류 수
    product_list=('product_name', lambda x: list(set(x))) # 가입 상품 리스트 (연관분석용)
).reset_index()

print("고객 단위 집계 후 데이터 크기:", customer_df.shape)
print("\n전처리된 고객 데이터 상위 5개:")
print(customer_df.head())

# 4. 전처리 완료된 데이터 저장
customer_df.to_pickle('customer_processed.pkl')
print("\n[저장 완료] 'customer_processed.pkl' 파일로 전처리 데이터가 저장되었습니다.")