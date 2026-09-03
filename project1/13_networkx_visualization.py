from pathlib import Path
import platform
import sys
import warnings
import matplotlib.pyplot as plt
import networkx as nx
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

# 2. 실행 환경 독립적 동적 경로 설정 (스크립트 실행 / 인터랙티브 실행 모두 대응)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

RULES_PATH = BASE_DIR / "output" / "association_rules_by_persona.pkl"
OUTPUT_DIR = BASE_DIR / "output"

if not RULES_PATH.exists():
    print(f"[오류] 연관 규칙 결과 파일이 존재하지 않습니다: {RULES_PATH}")
    print("먼저 '12_association_rules.py'를 실행해 주세요.")
    sys.exit(1)

# 3. 최신 연관 규칙 데이터 로드 (pkl)
rules_df = pd.read_pickle(RULES_PATH)

# 4. 클러스터별로 각각 독립된 이미지 1장씩 생성
cluster_ids = sorted(rules_df["cluster"].unique())

# 클러스터별 중심성 결과를 뒤 요약 출력에서도 재사용할 수 있도록 저장
centrality_summary = {}

for c_id in cluster_ids:
    cluster_rules = rules_df[rules_df["cluster"] == c_id].copy()
    total_rule_count = len(cluster_rules)  # 페르소나별 전체 연관 규칙 수(리포트 상단 표 수치)

    # 대표성 없는 "기타 상품" 카테고리 노드는 허브/브릿지 해석(그래프)에서만 제외
    sub_rules = cluster_rules[
        (cluster_rules["antecedents"] != "기타 상품")
        & (cluster_rules["consequents"] != "기타 상품")
    ]

    fig, ax = plt.subplots(figsize=(11, 9))

    if sub_rules.empty:
        ax.set_title(f"Cluster {c_id}: 도출된 규칙 없음", fontsize=14)
        ax.axis("off")
        fig_path = OUTPUT_DIR / f"network_graph_cluster{c_id}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[완료] Cluster {c_id} 그래프 저장: {fig_path.name} (규칙 없음)")
        continue

    # NetworkX 방향성 그래프 생성
    G = nx.DiGraph()

    for _, row in sub_rules.iterrows():
        # 12_association_rules.py에서 이미 ", ".join(list(x))로 문자열 변환되어
        # 저장되므로 별도의 eval() 파싱 없이 그대로 노드명으로 사용
        ant = row["antecedents"]
        con = row["consequents"]
        lift = row["lift"]

        G.add_edge(ant, con, weight=lift)

    # 중심성 지표 계산 (허브 = Degree Centrality, 브릿지 = Betweenness Centrality)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight=None)

    hub_node = max(degree_centrality, key=degree_centrality.get)
    hub_score = degree_centrality[hub_node]

    # [추가] 클러스터별 연결 중심성(Degree Centrality) 상위 5개 상품 저장
    top5_centrality = sorted(
        degree_centrality.items(), key=lambda x: -x[1]
    )[:5]

    # 브릿지(Bridge)는 "서로 다른 상품군을 이어주는 별도의 연결고리"를 찾는 지표이므로,
    # 허브 노드 자신(연결이 많아 매개 중심성도 함께 높게 나오는 경우가 흔함)을 제외한
    # 나머지 노드 중 매개 중심성이 가장 높은 노드를 브릿지로 정의한다.
    non_hub_betweenness = {
        n: v for n, v in betweenness_centrality.items() if n != hub_node
    }
    if non_hub_betweenness and max(non_hub_betweenness.values()) > 0:
        bridge_node = max(non_hub_betweenness, key=non_hub_betweenness.get)
        bridge_score = non_hub_betweenness[bridge_node]
    else:
        # 허브를 제외하면 매개 역할을 하는 노드가 없음 = 허브가 허브·브릿지를 겸함
        bridge_node = hub_node
        bridge_score = betweenness_centrality[hub_node]

    centrality_summary[c_id] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "rules": total_rule_count,
        "hub": hub_node,
        "hub_score": hub_score,
        "bridge": bridge_node,
        "bridge_score": bridge_score,
        "hub_is_bridge": bridge_node == hub_node,
        "top5_centrality": top5_centrality,  # [추가]
    }

    # Spring Layout 적용 (노드 간격 조정)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # 허브(연결 중심성 최상위)·브릿지(매개 중심성 최상위) 노드를 시각적으로 강조
    def _node_color(n):
        if n == hub_node:
            return "orange"  # 허브 (허브·브릿지 겸함인 경우도 포함)
        if n == bridge_node:
            return "mediumseagreen"  # 브릿지
        return "skyblue"

    node_colors = [_node_color(n) for n in G.nodes()]

    # 노드 및 엣지 그리기
    nx.draw_networkx_nodes(
        G, pos, node_size=1500, node_color=node_colors, alpha=0.9, ax=ax
    )
    nx.draw_networkx_labels(
        G,
        pos,
        font_family=plt.rcParams["font.family"],
        font_size=9,
        font_weight="bold",
        ax=ax,
    )

    edges = G.edges(data=True)
    weights = [min(d["weight"], 30) * 0.2 for u, v, d in edges]

    nx.draw_networkx_edges(
        G,
        pos,
        width=weights,
        edge_color="gray",
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    ax.set_title(
        f"Cluster {c_id} 상품 연관 네트워크 (노드: 상품, 두께: Lift, 주황: 허브, 초록: 브릿지)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"network_graph_cluster{c_id}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[완료] Cluster {c_id} 그래프 저장: {fig_path.name}")

# 5. 클러스터별 허브(Hub) / 브릿지(Bridge) 요약 콘솔 출력
print("\n" + "=" * 80)
print("[군집별 네트워크 허브(Hub) / 브릿지(Bridge) 요약]")
print("=" * 80)
for c_id, s in centrality_summary.items():
    print(f"■ Cluster {c_id} (노드 {s['nodes']}개 | 엣지 {s['edges']}개 | 연관 규칙 {s['rules']}개)")
    print(f"   - 허브(Hub, 연결 중심성 최고): {s['hub']} ({s['hub_score']:.4f})")
    if s["hub_is_bridge"]:
        print(f"     └ 브릿지(매개 중심성) 역할도 동시 수행 ({s['bridge_score']:.4f}) — 별도 브릿지 상품 없음")
    else:
        print(f"   - 브릿지(Bridge, 허브 제외 매개 중심성 최고): {s['bridge']} ({s['bridge_score']:.4f})")

    # [추가] 연결 중심성(Degree Centrality) 상위 5개 상품 출력
    print(f"   - [연결 중심성(Degree Centrality) TOP 5]")
    for rank, (node, score) in enumerate(s["top5_centrality"], start=1):
        print(f"     {rank}. {node}: {score:.4f}")
    print("-" * 80)