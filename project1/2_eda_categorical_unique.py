from collections import Counter
from pathlib import Path
import platform
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. OS 독립적 한글 폰트 자동 설정 (Windows/Mac/Linux 대응)
system_os = platform.system()
if system_os == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif system_os == "Darwin":  # Mac OS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux 및 기타 OS
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])

# 2. 동적 경로 설정 및 데이터 검증
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"

if not PROCESSED_PATH.exists():
    print(
        f"[오류] 전처리된 파일이 존재하지 않습니다: {PROCESSED_PATH}"
    )
    print("먼저 '1_preprocess.py'를 실행하여 pkl 파일을 생성해 주세요.")
    sys.exit(1)

# 3. 전처리 데이터 불러오기
df_aggregated = pd.read_pickle(PROCESSED_PATH)

# ======================================================================
# 1. 주요 변수 4분할 시각화
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Graph 1: 연령대별 고객 분포
sns.countplot(
    data=df_aggregated,
    x="age",
    ax=axes[0, 0],
    palette="Blues_d",
    order=sorted(df_aggregated["age"].dropna().unique()),
)
axes[0, 0].set_title(
    "1. 연령대별 고객 분포", fontsize=14, fontweight="bold"
)
axes[0, 0].set_xlabel("연령대")
axes[0, 0].set_ylabel("고객 수 (명)")

# Graph 2: 소득 구간별 고객 분포
sns.countplot(
    data=df_aggregated,
    x="income_bracket",
    ax=axes[0, 1],
    palette="Greens_d",
)
axes[0, 1].set_title(
    "2. 소득 구간별 고객 분포", fontsize=14, fontweight="bold"
)
axes[0, 1].set_xlabel("소득 구간")
axes[0, 1].set_ylabel("고객 수 (명)")
axes[0, 1].tick_params(axis="x", rotation=30)

# Graph 3: 총 거래 금액 분포 (Log Scale)
sns.histplot(
    np.log1p(df_aggregated["total_transaction_amt"]),
    kde=True,
    ax=axes[1, 0],
    color="purple",
    bins=30,
)
axes[1, 0].set_title(
    "3. 총 거래 금액 분포 (Log Scale)", fontsize=14, fontweight="bold"
)
axes[1, 0].set_xlabel("log1p(총 거래 금액)")
axes[1, 0].set_ylabel("고객 수 (명)")

# Graph 4: 1인당 보유 상품 수 분포
sns.countplot(
    data=df_aggregated,
    x="product_count",
    ax=axes[1, 1],
    palette="Oranges_d",
)
axes[1, 1].set_title(
    "4. 1인당 보유 상품 수 분포", fontsize=14, fontweight="bold"
)
axes[1, 1].set_xlabel("보유 상품 수 (개)")
axes[1, 1].set_ylabel("고객 수 (명)")

plt.tight_layout()
plt.show()

# ======================================================================
# 2. TOP 10 인기 금융 상품 시각화
# ======================================================================
all_products = [
    item for sublist in df_aggregated["product_list"] for item in sublist
]
product_counts = Counter(all_products)
df_top_products = pd.DataFrame(
    product_counts.most_common(10), columns=["상품명", "보유고객수"]
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_top_products, x="보유고객수", y="상품명", palette="viridis"
)
plt.title(
    "TOP 10 인기 금융 상품 보유 현황", fontsize=15, fontweight="bold"
)
plt.xlabel("보유 고객 수 (명)")
plt.ylabel("상품명")
plt.tight_layout()
plt.show()