import pandas as pd
import numpy as np

# 1. 데이터 불러오기
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'

try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

# 2. 금융 상품(product_name) 분포 및 쏠림 현상 집계
product_counts = df['product_name'].value_counts()
total_records = len(df)
total_products = df['product_name'].nunique()

# 상위 n개 상품 비중
top_10_cnt = product_counts.head(10).sum()
top_20_cnt = product_counts.head(20).sum()
top_50_cnt = product_counts.head(50).sum()

print("=" * 70)
print("금융 상품(product_name) 분포 및 불균형 분석 결과")
print("=" * 70)
print(f"전체 거래 건수: {total_records:,}건")
print(f"고유 금융 상품 수: {total_products:,}개")
print(f"상위 10개 상품 거래 비중: {top_10_cnt:,}건 ({top_10_cnt / total_records * 100:.2f}%)")
print(f"상위 20개 상품 거래 비중: {top_20_cnt:,}건 ({top_20_cnt / total_records * 100:.2f}%)")
print(f"상위 50개 상품 거래 비중: {top_50_cnt:,}건 ({top_50_cnt / total_records * 100:.2f}%)")

print("\n[상위 10개 인기 금융 상품]")
print(product_counts.head(10))