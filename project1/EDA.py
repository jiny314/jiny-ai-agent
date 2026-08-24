Python
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")

# ==========================================
# 1. 데이터 개요 및 기초 통계 확인
# ==========================================
print("=== 1. 데이터 개요 ===")
print(f"원본 데이터 크기: {df_raw.shape}")  # (385019, 22)
print(f"고객 단위 집계 데이터 크기: {df_aggregated.shape}")  # (335506, 12)

# ==========================================
# 2. 시각화: 주요 변수 분포 및 관계 파악
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Graph 1: 연령대별 고객 분포
sns.countplot(
    data=df_aggregated,
    x="age",
    ax=axes[0, 0],
    palette="Blues_d",
    order=sorted(df_aggregated["age"].dropna().unique()),
)
axes[0, 0].set_title("1. 연령대별 고객 분포", fontsize=14, fontweight="bold")
axes[0, 0].set_xlabel("연령대")
axes[0, 0].set_ylabel("고객 수 (명)")

# Graph 2: 소득 구간별 고객 분포
sns.countplot(
    data=df_aggregated,
    x="income_bracket",
    ax=axes[0, 1],
    palette="Greens_d",
)
axes[0, 1].set_title("2. 소득 구간별 고객 분포", fontsize=14, fontweight="bold")
axes[0, 1].set_xlabel("소득 구간")
axes[0, 1].set_ylabel("고객 수 (명)")
axes[0, 1].tick_params(axis="x", rotation=30)

# Graph 3: 거래 금액 분포 (Log Scale 적용)
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
sns.countplot(data=df_aggregated, x="product_count", ax=axes[1, 1], palette="Oranges_d")
axes[1, 1].set_title("4. 1인당 보유 상품 수 분포", fontsize=14, fontweight="bold")
axes[1, 1].set_xlabel("보유 상품 수 (개)")
axes[1, 1].set_ylabel("고객 수 (명)")

plt.tight_layout()
plt.show()

# ==========================================
# 3. 주요 상품 상위 TOP 10 보유 현황 시각화
# ==========================================
from collections import Counter

# product_list 펼치기
all_products = [
    item for sublist in df_aggregated["product_list"] for item in sublist
]
product_counts = Counter(all_products)
df_top_products = pd.DataFrame(
    product_counts.most_common(10), columns=["상품명", "보유고객수"]
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_top_products, x="보유고객수", y="상품명", palette="Viridis"
)
plt.title(
    "TOP 10 인기 금융 상품 보유 현황", fontsize=15, fontweight="bold"
)
plt.xlabel("보유 고객 수 (명)")
plt.ylabel("상품명")
plt.show()