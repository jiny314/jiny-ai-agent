from pathlib import Path
import sys
import warnings
import platform

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])

# 2. 실행 환경 독립적 동적 경로 설정
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RULES_PATH = OUTPUT_DIR / "association_rules_by_persona.pkl"

if not RULES_PATH.exists():
    print(f"[오류] 연관 규칙 결과 파일이 존재하지 않습니다: {RULES_PATH}")
    print("먼저 '12_association_rules.py'를 실행해 주세요.")
    sys.exit(1)

rules_df = pd.read_pickle(RULES_PATH)

# 3. networkx 버전에 따라 louvain 함수 위치가 다를 수 있어 안전하게 로드
LOUVAIN_SOURCE = None
try:
    if hasattr(nx, "community") and hasattr(nx.community, "louvain_communities"):
        LOUVAIN_SOURCE = "networkx"
except Exception:
    pass

if LOUVAIN_SOURCE is None:
    try:
        import community as community_louvain  # python-louvain 패키지
        LOUVAIN_SOURCE = "python-louvain"
    except ImportError:
        print("[오류] 커뮤니티 탐지 라이브러리를 찾을 수 없습니다.")
        print("아래 중 하나를 설치해 주세요.")
        print("  pip install networkx --upgrade   (networkx>=2.8 권장, 별도 설치 불필요)")
        print("  또는 pip install python-louvain")
        sys.exit(1)

print(f"[안내] 커뮤니티 탐지 라이브러리: {LOUVAIN_SOURCE}")


def _get_cmap(name, n):
    """matplotlib 버전에 상관없이 컬러맵을 안전하게 가져오는 헬퍼"""
    n = max(n, 1)
    try:
        return plt.colormaps.get_cmap(name).resampled(n)
    except Exception:
        try:
            return plt.get_cmap(name, n)
        except Exception:
            return cm.get_cmap(name, n)


cluster_ids = sorted(rules_df["cluster"].unique())
community_summary = {}

for c_id in cluster_ids:
    cluster_rules = rules_df[rules_df["cluster"] == c_id].copy()

    sub_rules = cluster_rules[
        (cluster_rules["antecedents"] != "기타 상품")
        & (cluster_rules["consequents"] != "기타 상품")
    ]

    if sub_rules.empty:
        print(f"[알림] Cluster {c_id}: 커뮤니티 탐지에 사용할 규칙이 없습니다.")
        continue

    # 4. 방향성 그래프 생성 (13번과 동일한 구조 유지 - 조합 상품도 하나의 노드로 취급)
    G = nx.DiGraph()
    for _, row in sub_rules.iterrows():
        ant, con, lift = row["antecedents"], row["consequents"], row["lift"]
        if G.has_edge(ant, con):
            G[ant][con]["weight"] = max(G[ant][con]["weight"], lift)
        else:
            G.add_edge(ant, con, weight=lift)

    G_undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]["weight"] = max(G_undirected[u][v]["weight"], weight)
        else:
            G_undirected.add_edge(u, v, weight=weight)

    if LOUVAIN_SOURCE == "networkx":
        communities = nx.community.louvain_communities(
            G_undirected, weight="weight", seed=42
        )
        mod_score = nx.community.modularity(G_undirected, communities, weight="weight")
    else:
        partition = community_louvain.best_partition(
            G_undirected, weight="weight", random_state=42
        )
        grouped = {}
        for node, comm_id in partition.items():
            grouped.setdefault(comm_id, set()).add(node)
        communities = list(grouped.values())
        mod_score = community_louvain.modularity(partition, G_undirected, weight="weight")

    node_to_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = idx

    # 5. 시각화 (그래프 구조는 원본 노드 그대로 사용 - 13번과 동일)
    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(G_undirected, k=0.8, seed=42)

    palette = _get_cmap("tab10", len(communities))
    node_colors = [palette(node_to_community[n]) for n in G_undirected.nodes()]

    nx.draw_networkx_nodes(
        G_undirected, pos, node_size=1500, node_color=node_colors, alpha=0.9, ax=ax
    )
    nx.draw_networkx_labels(
        G_undirected,
        pos,
        font_family=plt.rcParams["font.family"],
        font_size=9,
        font_weight="bold",
        ax=ax,
    )
    weights = [min(d["weight"], 30) * 0.2 for _, _, d in G_undirected.edges(data=True)]
    nx.draw_networkx_edges(G_undirected, pos, width=weights, edge_color="gray", ax=ax)

    ax.set_title(
        f"Cluster {c_id} 상품 커뮤니티 탐지 (노드: 상품, 색상: 커뮤니티, Modularity: {mod_score:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    fig_path = OUTPUT_DIR / f"community_graph_cluster{c_id}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[완료] Cluster {c_id} 커뮤니티 그래프 저장: {fig_path.name}")

    # 6. 커뮤니티 요약: 조합 상품 노드를 개별 상품으로 분해 후 중복 제거
    #    (그래프 구조는 그대로 두고, 사람이 읽는 요약만 상품 단위로 정리)
    community_records = []
    for idx, comm in enumerate(communities):
        individual_products = set()
        for node in comm:
            for p in node.split(", "):
                individual_products.add(p.strip())
        community_records.append(
            {
                "community_id": idx,
                "n_nodes": len(comm),
                "nodes": sorted(comm),
                "n_unique_products": len(individual_products),
                "products": sorted(individual_products),
            }
        )

    community_summary[c_id] = {
        "n_communities": len(communities),
        "modularity": mod_score,
        "communities": community_records,
    }

# 7. 결과 저장 및 콘솔 요약 출력
SAVE_PATH = OUTPUT_DIR / "community_detection_summary.pkl"
pd.to_pickle(community_summary, SAVE_PATH)
print(f"\n[완료] 커뮤니티 탐지 결과 저장: {SAVE_PATH}")

print("\n" + "=" * 80)
print("[군집별 상품 커뮤니티 탐지 요약]")
print("=" * 80)
for c_id, summary in community_summary.items():
    print(
        f"■ Cluster {c_id} | 커뮤니티 수: {summary['n_communities']}개 | "
        f"Modularity: {summary['modularity']:.4f}"
    )
    for comm in summary["communities"]:
        preview = ", ".join(comm["products"][:5])
        more = (
            f" 외 {comm['n_unique_products'] - 5}개"
            if comm["n_unique_products"] > 5
            else ""
        )
        print(
            f"  - Community {comm['community_id']} "
            f"(그래프 노드 {comm['n_nodes']}개 / 개별 상품 {comm['n_unique_products']}개): "
            f"{preview}{more}"
        )
    print("-" * 80)