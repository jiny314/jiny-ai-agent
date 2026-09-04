from pathlib import Path
import platform
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. OS 독립적 한글 폰트 자동 설정
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
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"

if not PROCESSED_PATH.exists():
    print(f"[오류] 전처리된 파일이 존재하지 않습니다: {PROCESSED_PATH}")
    print("먼저 '1_preprocess.py'를 실행하여 pkl 파일을 생성해 주세요.")
    sys.exit(1)

# 3. 전처리 데이터 불러오기
df = pd.read_pickle(PROCESSED_PATH)

# ======================================================================
# 1. 연령대 x 소득 구간 고객 수 히트맵 (Cross-tabulation)
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
income_order = [
    "1000만원미만",
    "3000만원미만",
    "5000만원미만",
    "8000만원미만",
    "10000만원미만",
    "3억미만",
    "10000만원이상",
]

# 교차표 작성
ct = pd.crosstab(df["age"], df["income_bracket"])
ct = ct.reindex(index=age_order, columns=income_order).fillna(0)

plt.figure(figsize=(12, 7))
sns.heatmap(ct, annot=True, fmt=",.0f", cmap="YlGnBu", cbar=True)
plt.title(
    "연령대 및 소득 구간별 고객 분포 교차 분석",
    fontsize=15,
    fontweight="bold",
)
plt.xlabel("소득 구간")
plt.ylabel("연령대")
plt.tight_layout()
plt.show()

# ======================================================================
# 2. 연령대별 평균 가입 금액 및 총 거래 금액 비교
# ======================================================================
df_amt = (
    df.groupby("age")[["avg_product_amt", "total_transaction_amt"]]
    .mean()
    .reindex(age_order)
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=df_amt,
    x="age",
    y="avg_product_amt",
    color="skyblue",
    ax=ax1,
    alpha=0.7,
)
ax1.set_ylabel("평균 가입 금액 (원)", color="blue", fontweight="bold")
ax1.set_title(
    "연령대별 평균 가입 금액 분포", fontsize=15, fontweight="bold"
)

plt.tight_layout()
plt.show()