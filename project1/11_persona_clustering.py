from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")

# 1. 실행 환경 독립적 동적 경로 설정
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"
SAVE_PATH = BASE_DIR / "output" / "customer_clustered_k3.pkl"

if not PROCESSED_PATH.exists():
    print(f"[오류] 전처리 파일이 존재하지 않습니다: {PROCESSED_PATH}")
    sys.exit(1)

# 2. 데이터 불러오기
df = pd.read_pickle(PROCESSED_PATH)

# 3. 전처리 (로그 변환 및 스케일링 파이프라인 일치)
df["log_avg_product_amt"] = np.log1p(df["avg_product_amt"])
df["log_total_transaction_amt"] = np.log1p(df["total_transaction_amt"])

num_cols = ["log_avg_product_amt", "log_total_transaction_amt", "product_count"]
cat_cols = [
    "age",
    "gender",
    "income_bracket",
    "occupation_group",
    "investment_propensity",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ]
)

print("[처리 중] 피처 전처리 및 스케일링을 진행합니다...")
X_processed = preprocessor.fit_transform(df)

# 4. K-Means (K=3) 학습 및 군집 할당
print("[처리 중] K-Means (K=3) 군집화를 실행합니다...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_processed)

# 5. 군집별 프로파일링 요약 집계
cluster_summary = (
    df.groupby("cluster")
    .agg(
        고객수=("customer_id", "count"),
        평균가입금액=("avg_product_amt", "mean"),
        평균총거래금액=("total_transaction_amt", "mean"),
        평균보유상품수=("product_count", "mean"),
    )
    .reset_index()
)

# 고객 비율 계산
cluster_summary["고객비율(%)"] = (
    cluster_summary["고객수"] / len(df) * 100
).round(2)

print("\n" + "=" * 60)
print("[K=3 군집별 기초 요약 통계]")
print("=" * 60)
print(cluster_summary.to_string(index=False))

# 6. 클러스터링 결과 데이터 저장
df.to_pickle(SAVE_PATH)
print(f"\n[완료] 클러스터링 결과 파일 저장 완료: {SAVE_PATH.name}")