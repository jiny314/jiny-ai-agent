from pathlib import Path
import sys
import warnings
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

warnings.filterwarnings("ignore")

# 1. 실행 환경 독립적 동적 상대 경로 설정
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "customer_clustered_k3.pkl"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 클러스터링 데이터 로드 및 예외 처리
print("[1/4] 클러스터링 데이터(customer_clustered_k3.pkl)를 로드합니다...")
if not INPUT_PATH.exists():
    print(
        f"[오류] 데이터 파일이 존재하지 않습니다: {INPUT_PATH}\n'11_persona_clustering.py'를 먼저 실행해 주세요."
    )
    sys.exit(1)

df = pd.read_pickle(INPUT_PATH)
print(f"데이터 로드 완료! (전체 고객 수: {len(df):,}명)")

# 3. 페르소나(Cluster)별 FP-Growth 연관 규칙 도출 함수
results_all = []


def run_association_rules_by_cluster(
    cluster_df, cluster_id, min_support=0.01, min_threshold=0.1
):
    print(
        f"\n--- [Cluster {cluster_id}] 연관 분석 진행 중 (고객 수: {len(cluster_df):,}명) ---"
    )

    # 고객별 상품 리스트 추출
    transactions = cluster_df["product_list"].dropna().tolist()

    # TransactionEncoder 변환 (One-Hot DataFrame 생성)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    # FP-Growth로 Frequent Itemsets 추출 (Apriori 대비 고속 연산)
    frequent_itemsets = fpgrowth(
        df_tf, min_support=min_support, use_colnames=True
    )

    if frequent_itemsets.empty:
        print(f"  • Cluster {cluster_id}: 설정한 min_support({min_support}) 기준을 만족하는 빈발 항목이 없습니다.")
        return None

    # 연관 규칙 도출 (신뢰도 Confidence 기준)
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_threshold
    )

    if rules.empty:
        print(f"  • Cluster {cluster_id}: 유의미한 연관 규칙이 도출되지 않았습니다.")
        return None

    rules["cluster"] = cluster_id
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x))

    # 향상도(Lift) 기준 내림차순 정렬
    rules = rules.sort_values(by="lift", ascending=False)
    print(f"  • Cluster {cluster_id}: 총 {len(rules)}개의 연관 규칙 발굴 완료!")
    return rules


# 4. 전체 Cluster 대상 연관 규칙 도출 파이프라인
for c_id in sorted(df["cluster"].unique()):
    sub_df = df[df["cluster"] == c_id]
    rules_c = run_association_rules_by_cluster(
        sub_df, cluster_id=c_id, min_support=0.005, min_threshold=0.1
    )
    if rules_c is not None:
        results_all.append(rules_c)

# 5. 결과 통합 및 저장
if results_all:
    final_rules_df = pd.concat(results_all, ignore_index=True)
    save_csv_path = OUTPUT_DIR / "association_rules_k3.csv"
    final_rules_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 전체 페르소나 연관 규칙 추출 및 CSV 저장 완료: {save_csv_path.name}")

    # 상위 5개 주요 규칙 미리보기 출력
    preview_cols = [
        "cluster",
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
    ]
    print("\n[발굴된 대표 연관 규칙 상위 5선 (Lift 기준)]")
    print(final_rules_df[preview_cols].head(5).to_string(index=False))