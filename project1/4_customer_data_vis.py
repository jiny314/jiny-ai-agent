from pathlib import Path
import platform
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. OS 독립적 한글 폰트 자동 설정
system_os = platform.system()
if system_os == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif system_os == "Darwin":  # Mac OS
    plt.rc("font", family="AppleGothic")
else:  # Linux 및 기타 OS
    plt.rc("font", family="NanumGothic")

plt.rc("axes", unicode_minus=False)
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

# 3. 데이터 불러오기
df = pd.read_pickle(PROCESSED_PATH)

# ======================================================================
# 1. 범주형 변수 유니크 값 및 빈도 확인
# ======================================================================
cat_cols = [
    "age",
    "gender",
    "income_bracket",
    "occupation_group",
    "investment_propensity",
    "risk_grade",
]

print("=" * 70)
print("주요 범주형 변수 유니크 값 및 상위 빈도")
print("=" * 70)

for col in cat_cols:
    if col in df.columns:
        unique_vals = df[col].nunique()
        top_vals = df[col].value_counts().head(3).to_dict()
        print(f"[{col}] 유니크 개수: {unique_vals} | 상위 분포: {top_vals}")

# ======================================================================
# 2. 범주형 변수 분포 시각화 (2x3 Subplots)
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(cat_cols):
    if col in df.columns:
        sns.countplot(
            data=df,
            x=col,
            ax=axes[idx],
            palette="Set2",
            order=df[col].value_counts().index,
        )
        axes[idx].set_title(
            f"{col} 분포", fontsize=13, fontweight="bold"
        )
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("고객 수 (명)")
        axes[idx].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()