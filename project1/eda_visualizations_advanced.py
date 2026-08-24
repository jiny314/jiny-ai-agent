import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 1. 데이터 로드
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'
try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) 수치형 변수 상관관계 히트맵
num_cols = ['product_amt', 'transaction_amt', 'product_period']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('수치형 변수 간 상관관계 히트맵')

# (2) 연령대 x 투자성향 교차 누적 바 차트
ct = pd.crosstab(df['age'], df['investment_propensity'], normalize='index') * 100
ct.plot(kind='bar', stacked=True, ax=axes[1], colormap='Set3')
axes[1].set_title('연령대별 투자성향 분포 (%)')
axes[1].set_xlabel('연령대')
axes[1].set_ylabel('비율 (%)')
axes[1].legend(title='투자성향', bbox_to_anchor=(1.05, 1), loc='upper left')

# (3) 금융 상품 Top 15 가입 분포 (롱테일)
top_15 = df['product_name'].value_counts().head(15)
sns.barplot(x=top_15.values, y=top_15.index, ax=axes[2], palette='viridis')
axes[2].set_title('상위 15개 금융 상품 가입 분포 (쏠림 현상)')
axes[2].set_xlabel('거래 건수')

plt.tight_layout()
plt.savefig('eda_advanced_summary.png', dpi=300)
print("[시각화 완료] 'eda_advanced_summary.png' 저장 완료")