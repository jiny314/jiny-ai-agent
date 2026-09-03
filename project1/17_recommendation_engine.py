from pathlib import Path
import sys
import warnings
import platform

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# 1. 실행 환경 독립적 동적 경로 설정
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = OUTPUT_DIR / "customer_processed.pkl"

KMEANS_CANDIDATES = [
    OUTPUT_DIR / "customer_clustered_k3.pkl",
    OUTPUT_DIR / "customer_clustered.pkl",
]
KMEANS_PATH = next((p for p in KMEANS_CANDIDATES if p.exists()), None)

if not PROCESSED_PATH.exists():
    print(f"[오류] 전처리된 파일이 존재하지 않습니다: {PROCESSED_PATH}")
    print("먼저 '1_preprocess.py'를 실행하여 pkl 파일을 생성해 주세요.")
    sys.exit(1)

df = pd.read_pickle(PROCESSED_PATH)

df_kmeans = None
kmeans_col = None
if KMEANS_PATH is not None:
    df_kmeans = pd.read_pickle(KMEANS_PATH)
    kmeans_col = next(
        (c for c in ["cluster", "persona_cluster"] if c in df_kmeans.columns), None
    )
    print(f"[안내] 페르소나 라벨 로드: {KMEANS_PATH.name} (컬럼: {kmeans_col})")
else:
    print("[안내] K-Means 결과 파일이 없어 페르소나 라벨 없이 진행합니다 (추천 자체엔 영향 없음).")

# 2. 고객 x 상품 상호작용 행렬(희소 행렬) 구축
print("[처리 중] 고객 x 상품 상호작용 행렬 구축 중...")

all_products = sorted({p for plist in df["product_list"] for p in plist})
product_to_idx = {p: i for i, p in enumerate(all_products)}
n_products = len(all_products)
n_customers = len(df)

print(f"  - 고객 수: {n_customers:,}명 | 고유 상품 수: {n_products}개")

rows, cols = [], []
for cust_idx, plist in enumerate(df["product_list"]):
    for p in plist:
        rows.append(cust_idx)
        cols.append(product_to_idx[p])

interaction = sparse.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(n_customers, n_products),
)

# 3. 아이템 기반 협업 필터링 (상품 간 코사인 유사도)
print("[처리 중] 아이템 기반 협업 필터링(상품 유사도) 계산 중...")
item_similarity = cosine_similarity(interaction.T)  # (n_products, n_products)

# 4. 행렬분해 (TruncatedSVD)
N_COMPONENTS = 20
print(f"[처리 중] TruncatedSVD 행렬분해 중 (n_components={N_COMPONENTS})...")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
svd.fit(interaction)
item_factors = svd.components_.T  # (n_products, n_components)
explained_var = svd.explained_variance_ratio_.sum()
print(f"  - 설명된 분산 비율: {explained_var:.4f}")

# 5. 인기도 베이스라인
popularity = np.asarray(interaction.sum(axis=0)).flatten()
popularity_rank = np.argsort(-popularity)  # 인기 많은 순 상품 인덱스

# 6. 추천 함수 정의
def recommend_item_cf(known_idx, top_n=10):
    if len(known_idx) == 0:
        return popularity_rank[:top_n]
    scores = item_similarity[known_idx].sum(axis=0)
    scores[known_idx] = -np.inf
    return np.argsort(-scores)[:top_n]


def recommend_svd(customer_row, top_n=10):
    user_vec = svd.transform(customer_row)  # (1, n_components)
    scores = (user_vec @ item_factors.T).flatten()
    known_idx = customer_row.nonzero()[1]
    scores[known_idx] = -np.inf
    return np.argsort(-scores)[:top_n]


def recommend_popularity(known_idx, top_n=10):
    scores = popularity.copy().astype(float)
    scores[known_idx] = -np.inf
    return np.argsort(-scores)[:top_n]


# 7. Leave-one-out 평가 (상품 2개 이상 보유 고객 대상)
eval_candidates = df.index[df["product_count"] >= 2].to_numpy()
N_EVAL_SAMPLE = min(5000, len(eval_candidates))
rng = np.random.RandomState(42)
eval_sample = rng.choice(eval_candidates, size=N_EVAL_SAMPLE, replace=False)

print(
    f"\n[처리 중] Leave-one-out 평가 실행 중 "
    f"(상품 2개 이상 보유 고객 {len(eval_candidates):,}명 중 {N_EVAL_SAMPLE:,}명 샘플)..."
)

results = {"item_cf": {5: 0, 10: 0}, "svd": {5: 0, 10: 0}, "popularity": {5: 0, 10: 0}}

for cust_idx in eval_sample:
    plist = df.loc[cust_idx, "product_list"]
    item_indices = [product_to_idx[p] for p in plist]

    held_out = rng.choice(item_indices)
    known_idx = np.array([i for i in item_indices if i != held_out])

    row = interaction[cust_idx].copy()
    row[0, held_out] = 0

    rec_cf = recommend_item_cf(known_idx, top_n=10)
    rec_svd = recommend_svd(row, top_n=10)
    rec_pop = recommend_popularity(known_idx, top_n=10)

    for k in (5, 10):
        if held_out in rec_cf[:k]:
            results["item_cf"][k] += 1
        if held_out in rec_svd[:k]:
            results["svd"][k] += 1
        if held_out in rec_pop[:k]:
            results["popularity"][k] += 1

print("\n" + "=" * 80)
print("[추천 방법별 평가 결과 (Leave-one-out, Recall@K)]")
print("=" * 80)
eval_summary = []
for method, scores in results.items():
    row = {"방법": method}
    for k, hits in scores.items():
        recall = hits / N_EVAL_SAMPLE
        row[f"Recall@{k}"] = round(recall, 4)
    eval_summary.append(row)

eval_df = pd.DataFrame(eval_summary)
print(eval_df.to_string(index=False))
print("=" * 80)

# 8. 페르소나별 예시 추천 출력 (아이템 기반 CF 기준, 해석이 쉬움)
print("\n[군집(페르소나)별 예시 고객 추천 결과 - 아이템 기반 CF]")
if df_kmeans is not None and kmeans_col is not None:
    merged = df_kmeans[["customer_id", kmeans_col]].merge(
        df[["customer_id", "product_list", "product_count"]], on="customer_id"
    )
    for c_id in sorted(merged[kmeans_col].unique()):
        sample = merged[(merged[kmeans_col] == c_id) & (merged["product_count"] >= 1)]
        if sample.empty:
            continue
        example = sample.iloc[0]
        known_idx = np.array([product_to_idx[p] for p in example["product_list"]])
        rec_idx = recommend_item_cf(known_idx, top_n=5)
        rec_products = [all_products[i] for i in rec_idx]
        print(f"\n■ Cluster {c_id} 예시 고객 (customer_id={example['customer_id']})")
        print(f"  보유 상품: {', '.join(example['product_list'])}")
        print(f"  추천 상품 Top 5: {', '.join(rec_products)}")
else:
    print("  (페르소나 라벨이 없어 생략)")

# 9. 결과 저장
pd.to_pickle(item_similarity, OUTPUT_DIR / "item_similarity_matrix.pkl")
pd.to_pickle(
    {"item_factors": item_factors, "all_products": all_products, "explained_variance": explained_var},
    OUTPUT_DIR / "svd_item_factors.pkl",
)
eval_df.to_pickle(OUTPUT_DIR / "recommendation_evaluation_summary.pkl")
print(f"\n[완료] 결과 저장: item_similarity_matrix.pkl, svd_item_factors.pkl, recommendation_evaluation_summary.pkl")