from pathlib import Path
import sys
import warnings
import platform

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prefixspan import PrefixSpan
except ImportError:
    print("[오류] prefixspan 패키지가 설치되어 있지 않습니다.")
    print("터미널에서 아래 명령어를 실행한 뒤 다시 실행해 주세요.")
    print("  pip install prefixspan")
    sys.exit(1)

# 1. 실행 환경 독립적 동적 경로 설정
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "data" / "소비자.csv"

KMEANS_CANDIDATES = [
    OUTPUT_DIR / "customer_clustered_k3.pkl",
    OUTPUT_DIR / "customer_clustered.pkl",
]
KMEANS_PATH = next((p for p in KMEANS_CANDIDATES if p.exists()), None)

if not CSV_PATH.exists():
    print(f"[오류] 원본 데이터 파일이 존재하지 않습니다: {CSV_PATH}")
    print("data/ 폴더에 '소비자.csv' 파일을 위치시킨 후 다시 실행해 주세요.")
    sys.exit(1)

if KMEANS_PATH is None:
    print("[오류] K-Means 군집화 결과 파일이 존재하지 않습니다.")
    print(f"확인한 경로: {[str(p) for p in KMEANS_CANDIDATES]}")
    print("먼저 '11_persona_clustering.py'를 실행해 주세요.")
    sys.exit(1)

print(f"[안내] 페르소나 라벨 로드: {KMEANS_PATH.name}")
df_kmeans = pd.read_pickle(KMEANS_PATH)
kmeans_col = next(
    (c for c in ["cluster", "persona_cluster"] if c in df_kmeans.columns), None
)
if kmeans_col is None:
    print("[오류] K-Means 군집 컬럼을 찾을 수 없습니다.")
    sys.exit(1)

# 2. 원본 거래 데이터 불러오기 (인코딩 대응)
print("[처리 중] 원본 거래 데이터(소비자.csv)를 불러옵니다...")
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except (UnicodeDecodeError, Exception):
    try:
        df = pd.read_csv(CSV_PATH, encoding="cp949")
    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        sys.exit(1)

required_cols = ["customer_id", "product_name", "transaction_start_year"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"[오류] 필요한 컬럼이 없습니다: {missing_cols}")
    print(f"실제 컬럼 목록: {list(df.columns)}")
    sys.exit(1)

# 3. 거래 시점 데이터 정제
df["transaction_start_year"] = pd.to_numeric(
    df["transaction_start_year"], errors="coerce"
)

CURRENT_YEAR = pd.Timestamp.now().year
before_count = len(df)
df = df[
    df["transaction_start_year"].notna()
    & (df["transaction_start_year"] >= 1990)
    & (df["transaction_start_year"] <= CURRENT_YEAR)
]
after_count = len(df)
print(
    f"[안내] transaction_start_year 결측/이상치 제외: "
    f"{before_count - after_count:,}건 제외 ({(before_count - after_count) / before_count * 100:.2f}%), "
    f"{after_count:,}건 사용"
)

# 4. 페르소나 라벨 병합
df = df.merge(df_kmeans[["customer_id", kmeans_col]], on="customer_id", how="inner")

# 5. 고객별 시간순 상품 시퀀스 구축 (동일 상품 중복은 최초 등장만 유지)
print("[처리 중] 고객별 시간순 상품 시퀀스 구축 중...")


def build_sequence(group):
    ordered = group.sort_values(["transaction_start_year", "product_name"])
    seq = []
    for p in ordered["product_name"]:
        if not seq or seq[-1] != p:
            if p not in seq:
                seq.append(p)
    return seq


MAX_SEQUENCES_PER_CLUSTER = 50000
MIN_SUPPORT_RATIO = 0.01  # 군집 내 시퀀스 수의 1%
MIN_SUPPORT_FLOOR = 10

all_pattern_results = {}

for c_id in sorted(df[kmeans_col].unique()):
    cluster_df = df[df[kmeans_col] == c_id]

    sequences = (
        cluster_df.groupby("customer_id")
        .apply(build_sequence)
        .loc[lambda s: s.apply(len) >= 2]
        .tolist()
    )

    n_customers_total = cluster_df["customer_id"].nunique()
    n_sequences = len(sequences)
    print(
        f"\n[Cluster {c_id}] 전체 고객 {n_customers_total:,}명 중 "
        f"길이 2 이상 시퀀스 보유 고객: {n_sequences:,}명 "
        f"({n_sequences / n_customers_total * 100:.2f}%)"
    )

    if n_sequences < MIN_SUPPORT_FLOOR:
        print(f"  - 시퀀스가 너무 적어 마이닝을 건너뜁니다.")
        continue

    if n_sequences > MAX_SEQUENCES_PER_CLUSTER:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_sequences, size=MAX_SEQUENCES_PER_CLUSTER, replace=False)
        sequences = [sequences[i] for i in idx]
        print(f"  - 연산 속도를 위해 {MAX_SEQUENCES_PER_CLUSTER:,}개로 샘플링")

    min_support = max(MIN_SUPPORT_FLOOR, int(len(sequences) * MIN_SUPPORT_RATIO))
    print(f"  - PrefixSpan 실행 중 (min_support={min_support})...")

    ps = PrefixSpan(sequences)
    patterns = ps.frequent(min_support)

    # 길이 2 이상(실제 순서 정보가 있는) 패턴만 필터링, 지지도 내림차순 정렬
    multi_item_patterns = [
        (support, pattern) for support, pattern in patterns if len(pattern) >= 2
    ]
    multi_item_patterns.sort(key=lambda x: -x[0])

    print(f"  - 길이 2 이상 시퀀셜 패턴 {len(multi_item_patterns)}개 발견")

    top_patterns = []
    for support, pattern in multi_item_patterns[:10]:
        top_patterns.append(
            {
                "pattern": pattern,
                "support_count": support,
                "support_ratio": round(support / len(sequences), 4),
            }
        )
        print(f"    {' → '.join(pattern)}  (지지도: {support}/{len(sequences)} = {support / len(sequences):.4f})")

    all_pattern_results[c_id] = {
        "n_customers_total": n_customers_total,
        "n_sequences": n_sequences,
        "min_support": min_support,
        "top_patterns": top_patterns,
    }

# 6. 결과 저장
SAVE_PATH = OUTPUT_DIR / "sequential_patterns_by_persona.pkl"
pd.to_pickle(all_pattern_results, SAVE_PATH)
print(f"\n[완료] 시퀀셜 패턴 마이닝 결과 저장: {SAVE_PATH}")