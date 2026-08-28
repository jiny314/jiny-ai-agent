from pathlib import Path
import platform
import sys
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

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
INPUT_PATH = BASE_DIR / "output" / "customer_clustered.pkl"

if not INPUT_PATH.exists():
    print(f"[오류] 군집화 데이터 파일이 존재하지 않습니다: {INPUT_PATH}")
    print("먼저 '10_clustering_evaluation.py'를 실행해 주세요.")
    sys.exit(1)

df = pd.read_pickle(INPUT_PATH)


# 3. 군집별 FP-Growth 연관 분석 함수 정의 (임계값 조정)
def analyze_cluster_association(
    cluster_df, cluster_id, min_support=0.0005, min_lift=1.01
):
    print(
        f"\n[처리 중] Cluster {cluster_id} 연관 분석 진행 (고객 수: {len(cluster_df):,}명)..."
    )

    # 2개 이상의 상품을 보유한 고객만 타겟팅하여 연관 규칙 도출 정밀도 향상
    multi_product_df = cluster_df[cluster_df["product_count"] >= 2]
    if len(multi_product_df) == 0:
        print(f"  - Cluster {cluster_id}: 다지점 가입 고객이 없습니다.")
        return None

    transactions = multi_product_df["product_list"].tolist()

    # 원-핫 인코딩 변환
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # FP-Growth 알고리즘 적용
    frequent_itemsets = fpgrowth(
        df_encoded, min_support=min_support, use_colnames=True
    )

    if frequent_itemsets.empty:
        print(
            f"  - Cluster {cluster_id}: 조건(min_support={min_support})을 만족하는 빈발 항목집합이 없습니다."
        )
        return None

    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=min_lift
    )

    if rules.empty:
        print(
            f"  - Cluster {cluster_id}: 조건(min_lift={min_lift})을 만족하는 연관 규칙이 없습니다."
        )
        return None

    # 가공
    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )
    rules["cluster"] = cluster_id

    # 전체 고객 기준 지지도(Support)로 재계산
    rules["support_total"] = rules["support"] * (
        len(multi_product_df) / len(cluster_df)
    )

    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    return rules


# 4. 전체 군집 대상 연관 분석 실행
all_rules_list = []
for c_id in sorted(df["persona_cluster"].unique()):
    c_df = df[df["persona_cluster"] == c_id]
    rules_df = analyze_cluster_association(
        c_df, cluster_id=c_id, min_support=0.0005, min_lift=1.01
    )
    if rules_df is not None:
        all_rules_list.append(rules_df)

if all_rules_list:
    final_rules = pd.concat(all_rules_list, ignore_index=True)

    # 5. 결과 저장
    output_path = BASE_DIR / "output" / "association_rules_by_persona.pkl"
    final_rules.to_pickle(output_path)
    print(f"\n[완료] 연관성 분석 결과 저장 완료: {output_path.name}")

    # 콘솔 요약 출력
    print("\n" + "=" * 80)
    print("[군집별 상위 핵심 교차판매 상품 패키지 (Lift TOP 1)]")
    print("=" * 80)
    for c_id in sorted(final_rules["cluster"].unique()):
        c_rules = final_rules[final_rules["cluster"] == c_id]
        if not c_rules.empty:
            c_top = c_rules.iloc[0]
            print(f"■ Cluster {c_id}")
            print(
                f"  - 추천 상품 조합: [{c_top['antecedents']}] -> [{c_top['consequents']}]"
            )
            print(
                f"  - 지지도(Support): {c_top['support_total']:.5f} | 신뢰도(Confidence): {c_top['confidence']:.4f} | 향상도(Lift): {c_top['lift']:.2f}"
            )
            print("-" * 80)
else:
    print(
        "\n[알림] 여전히 규칙이 도출되지 않으면 min_support를 0.0001 로 낮추어 시도하세요."
    )