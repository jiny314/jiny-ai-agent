from collections import Counter
from pathlib import Path
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# 1. 동적 경로 설정 및 예외 처리
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "output" / "customer_processed.pkl"
CSV_PATH = BASE_DIR / "data" / "소비자.csv"

# 2. 데이터 불러오기
if PROCESSED_PATH.exists():
    print("[안내] 고객 단위 집계 데이터(customer_processed.pkl)를 불러옵니다.")
    df_cust = pd.read_pickle(PROCESSED_PATH)
    all_products = [
        item for sublist in df_cust["product_list"] for item in sublist
    ]
    product_counts = pd.Series(Counter(all_products))
    total_len = len(df_cust)
    unit_label = "고객 수 (명)"
elif CSV_PATH.exists():
    print("[안내] 원본 거래 데이터(소비자.csv)를 불러옵니다.")
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(CSV_PATH, encoding="cp949")
        except Exception as e:
            print(f"[오류] 파일 읽기 실패: {e}")
            sys.exit(1)
    product_counts = df["product_name"].value_counts()
    total_len = len(df)
    unit_label = "거래 건수 (건)"
else:
    print("[오류] 데이터 파일이 존재하지 않습니다.")
    sys.exit(1)

# 3. 금융 상품 분포 및 롱테일(Long-tail) 집계
top_10 = product_counts.head(10)
top_10_ratio = (top_10.sum() / total_len) * 100

print("=" * 70)
print(f"TOP 10 인기 금융 상품 가입 현황 ({unit_label} 기준)")
print("=" * 70)
print(top_10)
print("-" * 70)
print(
    f"전체 금융 상품 종류 수: {len(product_counts)}개"
)
print(
    f"상위 10개 상품의 집계 점유율: {top_10_ratio:.2f}%"
)
print("=" * 70)