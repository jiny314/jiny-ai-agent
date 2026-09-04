from pathlib import Path
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# 1. 동적 경로 설정 및 예외 처리
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"
CSV_PATH = BASE_DIR / "data" / "소비자.csv"

# 전처리 데이터가 있으면 전처리 데이터를, 없으면 원본 csv 사용
if PROCESSED_PATH.exists():
    print("[안내] 전처리 완료된 고객 단위 데이터를 불러옵니다.")
    df = pd.read_pickle(PROCESSED_PATH)
    num_cols = ["avg_product_amt", "total_transaction_amt", "product_count"]
elif CSV_PATH.exists():
    print("[안내] 원본 소비자.csv 데이터를 불러옵니다.")
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(CSV_PATH, encoding="cp949")
        except Exception as e:
            print(f"[오류] 파일 읽기 실패: {e}")
            sys.exit(1)
    num_cols = ["product_amt", "transaction_amt", "product_period"]
else:
    print(f"[오류] 요약할 데이터 파일이 존재하지 않습니다.")
    sys.exit(1)

# 2. 요약 통계량 가공
summary_data = []

for col in num_cols:
    if col in df.columns:
        summary_data.append(
            {
                "수치 항목": col,
                "평균": f"{df[col].mean():,.2f}",
                "표준편차": f"{df[col].std():,.2f}",
                "최소값": f"{df[col].min():,.2f}",
                "25% (Q1)": f"{df[col].quantile(0.25):,.2f}",
                "중앙값 (50%)": f"{df[col].median():,.2f}",
                "75% (Q3)": f"{df[col].quantile(0.75):,.2f}",
                "최대값": f"{df[col].max():,.2f}",
            }
        )

summary_df = pd.DataFrame(summary_data)

print("=" * 80)
print("수치형 변수 요약 통계량")
print("=" * 80)
print(summary_df.to_string(index=False))