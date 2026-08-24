import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 1단계에서 전처리 완료된 데이터 불러오기
print("[1/3] 전처리 데이터(customer_processed.pkl) 로드 중...")
customer_df = pd.read_pickle('customer_processed.pkl')

# 2. 파생변수 생성 및 수치형/범주형 변수 전처리
# 가입금액 대비 거래금액 비율 파생변수
customer_df['transaction_ratio'] = customer_df['total_transaction_amt'] / (customer_df['total_product_amt'] + 1e-5)

# 금액 변수 로그 변환 (로그스케일)
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
print(f"전처리 후 피처 차원: {X_prep.shape}")

# 3. 최적 군집 수(K) 탐색 (K=3~6)
print("[2/3] 최적 군집 수(K) 평가 진행 중...")
inertia_list = []
silhouette_list = []
k_range = range(3, 7)

# 데이터가 크므로 실루엣 계수는 2만 건 샘플링 산출
np.random.seed(42)
sample_idx = np.random.choice(X_prep.shape[0], size=min(20000, X_prep.shape[0]), replace=False)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_prep)
    
    inertia_list.append(kmeans.inertia_)
    sil_score = silhouette_score(X_prep[sample_idx], cluster_labels[sample_idx])
    silhouette_list.append(sil_score)
    print(f"K={k} -> Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {sil_score:.4f}")

# 4. 평가 결과 시각화 및 저장
print("[3/3] 평가 시각화 차트 저장 중...")
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia', color='tab:blue')
ax1.plot(k_range, inertia_list, marker='o', color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='tab:red')
ax2.plot(k_range, silhouette_list, marker='s', color='tab:red')

plt.title('K-Means Optimal K Evaluation')
plt.tight_layout()
plt.savefig('kmeans_k_evaluation.png', dpi=300)
print("[완료] 'kmeans_k_evaluation.png' 파일이 저장되었습니다.")