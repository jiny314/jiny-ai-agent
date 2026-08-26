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

# 1. OS 독립적 한글 폰트 설정
system_os = platform.system()
if system_os == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif system_os == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])

# 2. 동적 경로 설정 및 데이터 검증
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"

if not PROCESSED_PATH.exists():
    print(f"[오류] 전처리된 파일이 존재하지 않습니다: {PROCESSED_PATH}")
    sys.exit(1)

# 3. 데이터 불러오기 및 피처 전처리
df = pd.read_pickle(PROCESSED_PATH)

# 로그 변환 적용 (고액 거래 스케일링 Distortion 방지)
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

print("[처리 중] 피처 변환 및 스케일링을 진행합니다...")
X_processed = preprocessor.fit_transform(df)

# 4. 최적 K 산출 (Elbow Method & Silhouette Score)
k_range = range(2, 9)
inertias = []
silhouette_scores = []

print("[처리 중] 최적 군집 수(K) 평가를 실행합니다...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_processed)

    inertias.append(kmeans.inertia_)

    # 샘플링(10,000건)을 통한 초고속 실루엣 스코어 계산
    score = silhouette_score(
        X_processed, labels, sample_size=10000, random_state=42
    )
    silhouette_scores.append(score)
    print(f"  • K={k} | Inertia: {kmeans.inertia_:,.2f} | Silhouette Score: {score:.4f}")

# 5. 최적 K 평가 시각화
fig, ax1 = plt.subplots(figsize=(10, 5))

color = "tab:blue"
ax1.set_xlabel("군집 수 (K)", fontweight="bold")
ax1.set_ylabel("Inertia (Elbow Method)", color=color, fontweight="bold")
ax1.plot(k_range, inertias, color=color, marker="o", linewidth=2)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Silhouette Score", color=color, fontweight="bold")
ax2.plot(
    k_range,
    silhouette_scores,
    color=color,
    marker="s",
    linestyle="--",
    linewidth=2,
)
ax2.tick_params(axis="y", labelcolor=color)

plt.title(
    "K-Means 최적 군집 수(K) 평가 (Elbow Method & Silhouette)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()