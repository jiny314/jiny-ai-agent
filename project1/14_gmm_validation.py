from pathlib import Path
import sys
import warnings
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# 1. OS 독립적 한글 폰트 자동 설정
system_os = platform.system()
if system_os == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif system_os == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])

# 2. 실행 환경 독립적 동적 경로 설정 (스크립트 실행 / 인터랙티브 실행 모두 대응)
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

if KMEANS_PATH is None:
    print("[오류] K-Means 군집화 결과 파일이 존재하지 않습니다.")
    print(f"확인한 경로: {[str(p) for p in KMEANS_CANDIDATES]}")
    print("먼저 '11_persona_clustering.py'를 실행해 주세요.")
    sys.exit(1)

print(f"[안내] K-Means 결과 파일 로드: {KMEANS_PATH.name}")

# 3. 데이터 불러오기
df_processed = pd.read_pickle(PROCESSED_PATH)
df_kmeans = pd.read_pickle(KMEANS_PATH)

kmeans_col_candidates = ["cluster", "persona_cluster"]
kmeans_col = next((c for c in kmeans_col_candidates if c in df_kmeans.columns), None)
if kmeans_col is None:
    print(f"[오류] K-Means 군집 컬럼을 찾을 수 없습니다. 확인한 컬럼명: {kmeans_col_candidates}")
    sys.exit(1)

# 4. 11_persona_clustering.py와 동일한 피처 엔지니어링 (동일 조건 비교를 위해 반드시 일치)
feature_cols = ["age", "income_bracket", "total_transaction_amt", "product_count"]
clustering_df = df_processed[feature_cols].copy()
clustering_df["total_transaction_amt"] = np.log1p(clustering_df["total_transaction_amt"])
clustering_df = pd.get_dummies(clustering_df, columns=["age", "income_bracket"], drop_first=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_df)

# 5. 최적 컴포넌트 수 검증 (BIC / AIC)
# covariance_type="diag": 원-핫 인코딩된 더미 변수가 섞인 피처에서는 "full" 공분산이
# 소규모 부분집단에서 특이(singular)해지기 쉬워 BIC/AIC가 비정상적으로 폭주하고
# 사후확률이 전부 1.0으로 쏠리는 현상이 발생함. diag 공분산은 변수별 분산만 추정해
# 이런 불안정성 없이 훨씬 안정적으로 적합됨 (K-Means와 동일 피처로 공정 비교 가능)
n_range = range(2, 9)
bic_scores, aic_scores = [], []

print("[처리 중] GMM 최적 컴포넌트 수 평가를 실행합니다 (covariance_type=diag)...")
for n in n_range:
    gmm_tmp = GaussianMixture(
        n_components=n, covariance_type="diag", random_state=42, n_init=5
    )
    gmm_tmp.fit(scaled_features)
    bic = gmm_tmp.bic(scaled_features)
    aic = gmm_tmp.aic(scaled_features)
    bic_scores.append(bic)
    aic_scores.append(aic)
    print(f"  - n={n} | BIC: {bic:,.1f} | AIC: {aic:,.1f}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(list(n_range), bic_scores, marker="o", label="BIC", color="tab:blue")
ax1.plot(list(n_range), aic_scores, marker="s", label="AIC", color="tab:red")
ax1.axvline(3, color="gray", linestyle="--", alpha=0.7, label="K=3 (K-Means 선정값)")
ax1.set_xlabel("컴포넌트 수 (n_components)", fontweight="bold")
ax1.set_ylabel("Information Criterion (낮을수록 좋음)", fontweight="bold")
ax1.set_title(
    "GMM 최적 컴포넌트 수 평가 (BIC & AIC, covariance_type=diag)",
    fontsize=14,
    fontweight="bold",
)
ax1.legend()
plt.tight_layout()

fig_path = OUTPUT_DIR / "gmm_bic_aic_evaluation.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"[완료] 그래프 저장: {fig_path.name}")

# 6. K=3 기준 최종 GMM 적합 및 K-Means와 비교
gmm = GaussianMixture(
    n_components=3, covariance_type="diag", random_state=42, n_init=10
)
gmm_labels = gmm.fit_predict(scaled_features)
gmm_probs = gmm.predict_proba(scaled_features)
gmm_max_prob = gmm_probs.max(axis=1)

gmm_silhouette = silhouette_score(
    scaled_features, gmm_labels, sample_size=10000, random_state=42
)

kmeans_labels = df_kmeans[kmeans_col].values

# 7. GMM 라벨을 K-Means 라벨과 최대한 일치하도록 정렬 (헝가리안 알고리즘)
n_clusters = 3
cost_matrix = np.zeros((n_clusters, n_clusters))
for i in range(n_clusters):
    for j in range(n_clusters):
        cost_matrix[i, j] = -np.sum((gmm_labels == i) & (kmeans_labels == j))

row_ind, col_ind = linear_sum_assignment(cost_matrix)
label_map = dict(zip(row_ind, col_ind))
gmm_labels_aligned = np.array([label_map[label] for label in gmm_labels])

agreement_rate = (gmm_labels_aligned == kmeans_labels).mean() * 100
ari_score = adjusted_rand_score(kmeans_labels, gmm_labels_aligned)

crosstab = pd.crosstab(
    pd.Series(kmeans_labels, name="K-Means Cluster"),
    pd.Series(gmm_labels_aligned, name="GMM Cluster"),
)

print("\n" + "=" * 80)
print("[GMM vs K-Means 비교 결과 (covariance_type=diag)]")
print("=" * 80)
print(f"GMM Silhouette Score: {gmm_silhouette:.4f}")
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}  (1에 가까울수록 두 군집 결과가 유사)")
print(f"라벨 일치율: {agreement_rate:.2f}%")
print(f"평균 최대 사후확률(경계 확신도): {gmm_max_prob.mean():.4f}")
print(f"경계가 모호한 고객 비율(최대확률 0.6 미만): {(gmm_max_prob < 0.6).mean() * 100:.2f}%")
print("\n[교차표: K-Means Cluster x GMM Cluster]")
print(crosstab.to_string())
print("=" * 80)

# 8. 결과 저장
df_result = df_kmeans.copy()
df_result["gmm_cluster"] = gmm_labels_aligned
df_result["gmm_max_prob"] = gmm_max_prob

SAVE_PATH = OUTPUT_DIR / "customer_clustered_k3_gmm_validated.pkl"
df_result.to_pickle(SAVE_PATH)
print(f"\n[완료] GMM 검증 결과 저장: {SAVE_PATH}")