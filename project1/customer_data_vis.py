import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 기준)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 1. 데이터 불러오기
file_path = r'C:\Users\ddong\OneDrive\바탕 화면\jiny-ai-agent\ABA_1st_proj\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자\소비자.csv'
try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949')

print("="*50)
print("1. 주요 범주형 변수 유니크 값 확인")
print("="*50)
categorical_cols = ['age', 'gender', 'income_bracket', 'occupation_group', 'investment_propensity', 'risk_grade']
for col in categorical_cols:
    print(f"\n[ {col} ] 분포:")
    print(df[col].value_counts())

print("\n" + "="*50)
print("2. 수치형 변수 요약 통계량")
print("="*50)
print(df[['product_amt', 'transaction_amt', 'product_period']].describe())

print("\n" + "="*50)
print("3. 고유 고객 수 vs 전체 거래 건수")
print("="*50)
print(f"전체 거래 건수: {len(df):,}건")
print(f"고유 고객(customer_id) 수: {df['customer_id'].nunique():,}명")
print(f"고객 1인당 평균 거래 건수: {len(df) / df['customer_id'].nunique():.2f}건")

# 4. EDA 시각화 (그래프 저장)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 연령대별 고객 분포
sns.countplot(data=df, x='age', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('연령대별 거래 분포')

# (2) 소득 구간별 분포
sns.countplot(data=df, x='income_bracket', ax=axes[0, 1], palette='Set3')
axes[0, 1].set_title('소득 구간별 거래 분포')
axes[0, 1].tick_params(axis='x', rotation=45)

# (3) 투자 성향 분포
sns.countplot(data=df, x='investment_propensity', ax=axes[1, 0], palette='Pastel1')
axes[1, 0].set_title('투자 성향 분포')
axes[1, 0].tick_params(axis='x', rotation=45)

# (4) 상위 10개 인기 금융 상품
top_products = df['product_name'].value_counts().head(10)
sns.barplot(x=top_products.values, y=top_products.index, ax=axes[1, 1], palette='Blues_r')
axes[1, 1].set_title('상위 10개 가입 금융 상품')

plt.tight_layout()
plt.savefig('eda_summary.png', dpi=300)
print("\n[시각화 완료] 'eda_summary.png' 파일로 시각화 결과가 저장되었습니다.")