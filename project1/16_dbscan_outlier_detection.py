from pathlib import Path
import sys
import warnings
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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

# 2. 실행 환경 독립적 동적 경로 설정
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

df_processed = pd.read_pickle(PROCESSED_PATH)
df_kmeans = pd.read_pickle(KMEANS_PATH)

kmeans_col_candidates = ["cluster", "persona_cluster"]
kmeans_col = next((c for c in kmeans_col_candidates if c in df_kmeans.columns), None)
if kmeans_col is None:
    print(f"[오류] K-Means 군집 컬럼을 찾을 수 없습니다. 확인한 컬럼명: {kmeans_col_candidates}")
    sys.exit(1)

# 3. 11/14번과 동일한 피처 엔지니어링 (비교 일관성 유지)
feature_cols = ["age", "income_bracket", "total_transaction_amt", "product_count"]
clustering_df = df_processed[feature_cols].copy()
clustering_df["total_transaction_amt"] = np.log1p(clustering_df["total_transaction_amt"])
clustering_df = pd.get_dummies(clustering_df, columns=["age", "income_bracket"], drop_first=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_df)

n_features = scaled_features.shape[1]
MIN_SAMPLES = max(10, n_features * 2)

# 4. eps 자동 탐지 (k-distance 그래프의 knee point)
print(f"[처리 중] k-distance 계산 중 (min_samples={MIN_SAMPLES})... 데이터 규모(전체 {len(scaled_features):,}건)에 따라 몇 분 소요될 수 있습니다.")
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES, n_jobs=-1)
nn.fit(scaled_features)
distances, _ = nn.kneighbors(scaled_features)
k_distances = np.sort(distances[:, -1])

zero_ratio = (k_distances == 0).mean() * 100
print(f"[안내] k-distance가 0인 포인트 비율: {zero_ratio:.2f}% (동일 프로필 중복 고객이 많을수록 높아짐)")

x = np.arange(len(k_distances))
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min() + 1e-12)

# knee point: 대각선(0,0)->(1,1) 기준 곡선이 아래로 가장 많이 꺼지는 지점
# (y_norm - x_norm)이 가장 "음수"인 지점 = (x_norm - y_norm)이 최대인 지점
diff = x_norm - y_norm
knee_idx = int(np.argmax(diff))
eps_auto = float(k_distances[knee_idx])

# 안전장치: knee가 0으로 잡히는 경우(중복 포인트가 너무 많은 경우) 대체값 사용
if eps_auto <= 0:
    nonzero_distances = k_distances[k_distances > 0]
    if len(nonzero_distances) > 0:
        eps_auto = float(np.percentile(nonzero_distances, 10))
        knee_idx = int(np.searchsorted(k_distances, eps_auto))
        print(f"[안내] knee 탐지 결과가 0이라 대체값 사용: 0이 아닌 거리의 하위 10% 지점 = {eps_auto:.4f}")
    else:
        eps_auto = 0.5
        print("[경고] 모든 k-distance가 0입니다. eps를 임의값(0.5)으로 설정합니다.")

print(f"[완료] 최종 eps: {eps_auto:.4f} (정렬된 {len(k_distances):,}개 포인트 중 {knee_idx:,}번째 지점)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_distances)
ax.axhline(eps_auto, color="red", linestyle="--", label=f"자동 탐지 eps = {eps_auto:.4f}")
ax.axvline(knee_idx, color="red", linestyle=":", alpha=0.5)
ax.set_xlabel("포인트 (거리 오름차순 정렬)", fontweight="bold")
ax.set_ylabel(f"{MIN_SAMPLES}번째 최근접 이웃까지의 거리", fontweight="bold")
ax.set_title("DBSCAN eps 자동 탐지 (k-distance Knee Point)", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()

fig_path = OUTPUT_DIR / "dbscan_eps_selection.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"[완료] 그래프 저장: {fig_path.name}")

# 5. DBSCAN 실행
print(f"\n[처리 중] DBSCAN 실행 중 (eps={eps_auto:.4f}, min_samples={MIN_SAMPLES})...")
dbscan = DBSCAN(eps=eps_auto, min_samples=MIN_SAMPLES, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(scaled_features)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = int((dbscan_labels == -1).sum())
noise_ratio = n_noise / len(dbscan_labels) * 100

print("\n" + "=" * 80)
print("[DBSCAN 결과 요약]")
print("=" * 80)
print(f"발견된 군집 수: {n_clusters}개")
print(f"이상치(노이즈) 포인트 수: {n_noise:,}개 ({noise_ratio:.2f}%)")

cluster_sizes = pd.Series(dbscan_labels).value_counts().sort_index()
print("\n[군집별 크기 (-1 = 노이즈)]")
print(cluster_sizes.to_string())

if n_clusters >= 2:
    mask = dbscan_labels != -1
    if mask.sum() > 10000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(np.where(mask)[0], size=10000, replace=False)
        sil = silhouette_score(scaled_features[sample_idx], dbscan_labels[sample_idx])
    else:
        sil = silhouette_score(scaled_features[mask], dbscan_labels[mask])
    print(f"\nSilhouette Score (노이즈 제외): {sil:.4f}  (참고: K-Means 0.3252 / GMM 0.1555)")
else:
    print("\n[알림] 유효 군집이 2개 미만이라 Silhouette Score 계산이 의미 없음")

crosstab = pd.crosstab(
    df_kmeans[kmeans_col], pd.Series(dbscan_labels, name="DBSCAN_Label")
)
print("\n[교차표: K-Means Cluster x DBSCAN Label (-1=노이즈)]")
print(crosstab.to_string())
print("=" * 80)

# 6. 결과 저장
df_result = df_kmeans.copy()
df_result["dbscan_label"] = dbscan_labels
df_result["is_outlier"] = dbscan_labels == -1

SAVE_PATH = OUTPUT_DIR / "customer_clustered_k3_dbscan_validated.pkl"
df_result.to_pickle(SAVE_PATH)
print(f"\n[완료] DBSCAN 결과 저장: {SAVE_PATH}")

# 7. 이상치 고객 프로파일 요약 (있는 경우)
if n_noise > 0:
    outlier_mask = df_result["is_outlier"]
    outliers = df_processed.loc[df_processed["customer_id"].isin(df_result.loc[outlier_mask, "customer_id"])]
    print("\n[이상치로 분류된 고객 프로파일 요약]")
    print(
        outliers[["avg_product_amt", "total_transaction_amt", "product_count"]]
        .describe()
        .to_string()
    )