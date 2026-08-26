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

# 1. OS 독립적 한글 폰트 설정 (Windows / Mac / Linux)
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
    print(f"[오류] 전처리된 파일이 존재하지 않습니다: {PROCESSED_PATH}")
    print("먼저 '1_preprocess.py'를 실행하여 pkl 파일을 생성해 주세요.")
    sys.exit(1)

# 3. 전처리 데이터 불러오기
df = pd.read_pickle(PROCESSED_PATH)

# ======================================================================
# Chart 1. 상위 20개 금융 상품 보유 현황 및 누적 비율 (파레토 분석)
# ======================================================================
all_products = [item for sublist in df["product_list"] for item in sublist]
product_counts = Counter(all_products)
df_products = pd.DataFrame(
    product_counts.most_common(20), columns=["상품명", "고객수"]
)
df_products["cum_ratio"] = (
    df_products["고객수"].cumsum() / sum(product_counts.values()) * 100
)

fig, ax1 = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=df_products,
    x="상품명",
    y="고객수",
    palette="Blues_r",
    ax=ax1,
    alpha=0.85,
)
ax1.set_ylabel("보유 고객 수 (명)", color="navy", fontweight="bold")
ax1.set_xticklabels(df_products["상품명"], rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.plot(
    df_products["상품명"],
    df_products["cum_ratio"],
    color="red",
    marker="o",
    linewidth=2,
)
ax2.set_ylabel("전체 대비 누적 비율 (%)", color="red", fontweight="bold")
ax2.set_ylim(0, 105)
ax2.axhline(80, color="gray", linestyle="--", alpha=0.7)

plt.title(
    "1. 상위 20개 금융 상품 가입 고객 수 및 누적 비율 (파레토 분석)",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

# ======================================================================
# Chart 2. 수치형 변수 상관관계 히트맵
# ======================================================================
plt.figure(figsize=(8, 6))
num_cols = ["avg_product_amt", "total_transaction_amt", "product_count"]
corr_matrix = df[num_cols].corr()

# 노션 표기용 한글 컬럼명 변환
corr_matrix.columns = ["평균 가입금액", "총 거래금액", "보유 상품수"]
corr_matrix.index = ["평균 가입금액", "총 거래금액", "보유 상품수"]

sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues", cbar=True)
plt.title(
    "2. 수치형 변수 상관관계 히트맵 (다중공선성 확인)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

# ======================================================================
# Chart 3. 연령대 X 투자성향 교차 분석 누적 바 차트
# ======================================================================
age_order = [
    "20대",
    "30대",
    "40대",
    "50대",
    "60대",
    "70대 이상",
    "80대 이상",
]
ct_propensity = (
    pd.crosstab(df["age"], df["investment_propensity"], normalize="index") * 100
)
ct_propensity = ct_propensity.reindex(age_order)

ax = ct_propensity.plot(
    kind="bar", stacked=True, figsize=(12, 6), colormap="Set3"
)
plt.title(
    "3. 연령대별 투자 성향 분포 (비율 %)", fontsize=15, fontweight="bold"
)
plt.xlabel("연령대", fontweight="bold")
plt.ylabel("비율 (%)", fontweight="bold")
plt.legend(title="투자 성향", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()