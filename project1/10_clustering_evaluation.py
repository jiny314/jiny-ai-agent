from pathlib import Path
import platform
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler

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

# 2. 동적 경로 설정 및 전처리 데이터 로드
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"

if not PROCESSED_PATH.exists():
    print(f"[오류] 전처리된 파일이 존재하지 않습니다: {PROCESSED_PATH}")
    sys.exit(1)

df = pd.read_pickle(PROCESSED_PATH)

# 3. 금액 피처 Log Scale 변환 (이상치 완화 및 스케일링 안정화)
df["log_total_transaction_amt"] = np.log1p(df["total_transaction_amt"])
df["log_avg_product_amt"] = np.log1p(df["avg_product_amt"])

# 4. 파이프라인 구성 (수치형/범주형 변수 분리 처리)
numeric_features = [
    "log_total_transaction_amt",
    "log_avg_product_amt",
    "product_count",
]
categorical_features = [
    "age",
    "gender",
    "income_bracket",
    "investment_propensity",
    "risk_grade",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_processed = preprocessor.fit_transform(df)

# 5. 최적 K 탐색 (Elbow Method & Silhouette Score)
print("[처리 중] 최적 군집 수(K) 산출을 진행합니다...")
sample_size = min(10000, len(df))
np.random.seed(42)
sample_idx = np.random.choice(X_processed.shape[0], sample_size, replace=False)
X_sample = X_processed[sample_idx]

k_range = range(2, 7)
inertia_list = []
silhouette_list = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_processed)

    inertia_list.append(kmeans.inertia_)
    sil_score = silhouette_score(X_sample, labels[sample_idx])
    silhouette_list.append(sil_score)
    print(
        f"K={k} | Inertia: {kmeans.inertia_:.2f} | Silhouette Score: {sil_score:.4f}"
    )

# 6. 평가 지표 시각화
fig, ax1 = plt.subplots(figsize=(10, 5))

color = "tab:blue"
ax1.set_xlabel("군집 수 (K)")
ax1.set_ylabel("Inertia (Elbow Method)", color=color)
ax1.plot(k_range, inertia_list, marker="o", color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Silhouette Score", color=color)
ax2.plot(k_range, silhouette_list, marker="s", linestyle="--", color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("K-Means 최적 군집 수(K) 평가 결과", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 7. 최종 K=3 선정 및 클러스터 할당 결과 저장
OPTIMAL_K = 3
final_kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df["persona_cluster"] = final_kmeans.fit_predict(X_processed)

output_path = BASE_DIR / "output" / "customer_clustered.pkl"
df.to_pickle(output_path)
print(f"\n[완료] 클러스터링 데이터 저장 완료: {output_path.name}")