from pathlib import Path
import pandas as pd
import sys

# 1. 동적 경로 설정 (OS 및 타 환경 호환)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = DATA_DIR / "소비자.csv"

if not csv_path.exists():
    print(f"[오류] 데이터 파일이 존재하지 않습니다: {csv_path}")
    print("data/ 폴더에 '소비자.csv' 파일을 위치시킨 후 다시 실행해 주세요.")
    sys.exit(1)

# 2. 데이터 불러오기 (인코딩 대응)
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except (UnicodeDecodeError, Exception):
    try:
        df = pd.read_csv(csv_path, encoding='cp949')
    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        sys.exit(1)

# 3. 데이터 전처리 및 이상치 정제
df['age'] = df['age'].replace({'70대': '70대 이상'})
df['investment_propensity'] = df['investment_propensity'].replace({'-': '미응답'})
df['income_bracket'] = df['income_bracket'].replace({'30000만원미만': '3억미만'})

# 최빈값 추출 함수
def safe_mode(series):
    valid_series = series.dropna()
    if valid_series.empty:
        return "미응답"
    m = valid_series.mode()
    return m.iloc[0] if not m.empty else "미응답"

# 4. 고객(customer_id) 기준 요약 집계 (초고속 연산)
print("[처리 중] 고객 단위 데이터 요약 및 집계를 진행합니다...")

# 상품 리스트 고속 추출
product_series = df.groupby('customer_id')['product_name'].unique().apply(list)

# 메인 집계
customer_df = df.groupby('customer_id').agg(
    age=('age', safe_mode),
    gender=('gender', safe_mode),
    income_bracket=('income_bracket', safe_mode),
    occupation_group=('occupation_group', safe_mode),
    investment_propensity=('investment_propensity', safe_mode),
    risk_grade=('risk_grade', safe_mode),
    total_product_amt=('product_amt', 'sum'),
    avg_product_amt=('product_amt', 'mean'),
    total_transaction_amt=('transaction_amt', 'sum'),
    product_count=('product_name', 'nunique')
).reset_index()

customer_df['product_list'] = customer_df['customer_id'].map(product_series)

# 5. pkl 파일 저장
save_path = OUTPUT_DIR / 'customer_processed.pkl'
customer_df.to_pickle(save_path)

# 6. 실행 결과 및 상위 5개 데이터(a, b, c, d, e) 출력
print(f"\n[완료] {save_path.name} 저장 완료\n")
print("=" * 70)
print(f"원본 데이터 크기: {df.shape}")
print(f"고객 단위 집계 후 데이터 크기: {customer_df.shape}")
print("=" * 70)

# 노션 양식에 맞춘 상위 5개 미리보기 가공 및 출력
display_cols = [
    'customer_id', 'age', 'gender', 'income_bracket',
    'avg_product_amt', 'total_transaction_amt', 'product_count', 'product_list'
]
sample_df = customer_df.head(5)[display_cols].copy()

# customer_id를 a, b, c, d, e로 식별 변환
sample_df['customer_id'] = ['a', 'b', 'c', 'd', 'e']

# 금액 단위 포맷팅 (원 단위)
sample_df['avg_product_amt'] = sample_df['avg_product_amt'].apply(lambda x: f"{int(x):,}원")
sample_df['total_transaction_amt'] = sample_df['total_transaction_amt'].apply(lambda x: f"{int(x):,}원")
sample_df['product_count'] = sample_df['product_count'].apply(lambda x: f"{x}개")

print("\n[상위 5개 데이터 미리보기]")
print(sample_df.to_string(index=False))
print("=" * 70)