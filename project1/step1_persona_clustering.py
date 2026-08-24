import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# 1. 전처리 데이터 로드
customer_df = pd.read_pickle('customer_processed.pkl')

# 2. 파생변수 생성 및 스케일링
customer_df['transaction_ratio'] = customer_df['total_transaction_amt'] / (customer_df['total_product_amt'] + 1e-5)
customer_df['total_product_amt_log'] = np.log1p(customer_df['total_product_amt'])
customer_df['total_transaction_amt_log'] = np.log1p(customer_df['total_transaction_amt'])

num_features = ['total_product_amt_log', 'total_transaction_amt_log', 'product_count', 'transaction_ratio']
cat_features = ['age', 'income_bracket', 'investment_propensity', 'risk_grade']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

X_prep = preprocessor.fit_transform(customer_df)

# 3. K=3 최종 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_df['cluster'] = kmeans.fit_predict(X_prep)

# 4. 군집별 고객 수 및 비중 출력
print("=" * 80)
print("군집별 고객 수 및 비중")
print("=" * 80)
cluster_counts = customer_df['cluster'].value_counts().sort_index()
for c_id, cnt in cluster_counts.items():
    print(f"Cluster {c_id}: {cnt:,}명 ({cnt / len(customer_df) * 100:.2f}%)")

# 5. 군집별 프로파일링 (KeyError 수정 버전)
print("\n" + "=" * 80)
print("군집별 핵심 피처 평균값 비교")
print("=" * 80)
profile = customer_df.groupby('cluster').agg(
    total_product_amt_mean=('total_product_amt', 'mean'),
    total_transaction_amt_mean=('total_transaction_amt', 'mean'), # 컬럼명 수정 완료
    product_count_mean=('product_count', 'mean'),
    top_age=('age', lambda x: x.mode()[0]),
    top_investment=('investment_propensity', lambda x: x.mode()[0]),
    top_income=('income_bracket', lambda x: x.mode()[0])
).round(1)

print(profile)

# 6. Step 2 연관분석용 최종 데이터 저장
customer_df.to_pickle('customer_clustered_k3.pkl')
print("\n[저장 완료] 'customer_clustered_k3.pkl' 파일 저장 완료.")