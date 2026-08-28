from pathlib import Path
import platform
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# 2. 동적 경로 설정 및 군집화 데이터 로드
BASE_DIR = Path(__file__).resolve().parent
CLUSTERED_PATH = BASE_DIR / "output" / "customer_clustered.pkl"

if not CLUSTERED_PATH.exists():
    print(f"[오류] 클러스터링 데이터 파일이 존재하지 않습니다: {CLUSTERED_PATH}")
    print("먼저 '10_clustering_evaluation.py'를 실행해 주세요.")
    sys.exit(1)

df = pd.read_pickle(CLUSTERED_PATH)

# 최빈값 추출 함수
def get_mode(series):
    valid = series.dropna()
    if valid.empty:
        return "미응답"
    m = valid.mode()
    return m.iloc[0] if not m.empty else "미응답"

# 3. 군집별 주요 프로파일링 통계집계
print("[처리 중] 페르소나 군집별 프로파일링 분석을 진행합니다...")

persona_summary = df.groupby("persona_cluster").agg(
    customer_count=("customer_id", "count"),
    avg_total_amt=("total_transaction_amt", "mean"),
    avg_product_amt=("avg_product_amt", "mean"),
    avg_product_cnt=("product_count", "mean"),
    main_age=("age", get_mode),
    main_income=("income_bracket", get_mode),
    main_propensity=("investment_propensity", get_mode),
    main_risk=("risk_grade", get_mode)
).reset_index()

# 비율 계산 (고객 비중)
total_customers = len(df)
persona_summary["customer_ratio(%)"] = (persona_summary["customer_count"] / total_customers * 100).round(2)

# 4. 프로파일링 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 그래프 1: 군집별 고객 수 및 비율
sns.barplot(data=persona_summary, x="persona_cluster", y="customer_count", ax=axes[0, 0], palette="Blues_d")
axes[0, 0].set_title("1. 군집별 고객 수 분포", fontsize=14, fontweight="bold")
axes[0, 0].set_xlabel("페르소나 군집 (Cluster)")
axes[0, 0].set_ylabel("고객 수 (명)")

# 그래프 2: 군집별 평균 총 거래 금액
sns.barplot(data=persona_summary, x="persona_cluster", y="avg_total_amt", ax=axes[0, 1], palette="Greens_d")
axes[0, 1].set_title("2. 군집별 평균 총 거래 금액", fontsize=14, fontweight="bold")
axes[0, 1].set_xlabel("페르소나 군집 (Cluster)")
axes[0, 1].set_ylabel("평균 금액 (원)")

# 그래프 3: 군집별 평균 보유 상품 수
sns.barplot(data=persona_summary, x="persona_cluster", y="avg_product_cnt", ax=axes[1, 0], palette="Oranges_d")
axes[1, 0].set_title("3. 군집별 평균 보유 상품 수", fontsize=14, fontweight="bold")
axes[1, 0].set_xlabel("페르소나 군집 (Cluster)")
axes[1, 0].set_ylabel("보유 상품 수 (개)")

# 그래프 4: 군집별 주요 연령대 분포
sns.countplot(data=df, x="persona_cluster", hue="age", ax=axes[1, 1], palette="Purples_d")
axes[1, 1].set_title("4. 군집별 연령대 세부 구성", fontsize=14, fontweight="bold")
axes[1, 1].set_xlabel("페르소나 군집 (Cluster)")
axes[1, 1].set_ylabel("고객 수 (명)")

plt.tight_layout()
plt.show()

# 5. 콘솔 요약 출력
print("\n" + "="*80)
print("[페르소나 군집별 대표 특성 요약]")
print("="*80)
for idx, row in persona_summary.iterrows():
    print(f"■ 군집 {row['persona_cluster']} (비중: {row['customer_ratio(%)']}%, {row['customer_count']:,}명)")
    print(f"  - 대표 연령/소득: {row['main_age']} / {row['main_income']}")
    print(f"  - 투자성향/위험등급: {row['main_propensity']} / {row['main_risk']}")
    print(f"  - 평균 거래금액: {int(row['avg_total_amt']):,}원 | 평균 보유상품 수: {row['avg_product_cnt']:.2f}개")
    print("-" * 80)