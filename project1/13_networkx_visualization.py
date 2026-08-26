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

# 2. 동적 상대 경로 설정 및 예외 처리
BASE_DIR = Path(__file__).resolve().parent
RULES_PATH = BASE_DIR / "output" / "association_rules_k3.csv"

if not RULES_PATH.exists():
    print(f"[오류] 연관 규칙 결과 파일이 존재하지 않습니다: {RULES_PATH}")
    print("먼저 '12_association_rules.py'를 실행해 주세요.")
    sys.exit(1)

# 3. CSV 데이터 로드
rules_df = pd.read_csv(RULES_PATH)

# 4. Cluster 0 & Cluster 2 네트워크 그래프 시각화 (2개 서브플롯)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for idx, c_id in enumerate([0, 2]):
    ax = axes[idx]
    sub_rules = rules_df[rules_df["cluster"] == c_id].copy()

    if sub_rules.empty:
        ax.set_title(f"Cluster {c_id}: 도출된 규칙 없음", fontsize=14)
        ax.axis("off")
        continue

    # NetworkX 멀티 디렉티드 그래프 생성
    G = nx.DiGraph()

    for _, row in sub_rules.iterrows():
        # 문자열 형태의 리스트 복원
        ant = (
            eval(row["antecedents"])[0]
            if isinstance(row["antecedents"], str)
            else row["antecedents"][0]
        )
        con = (
            eval(row["consequents"])[0]
            if isinstance(row["consequents"], str)
            else row["consequents"][0]
        )
        lift = row["lift"]

        G.add_edge(ant, con, weight=lift)

    # Spring Layout 적용 (노드 간격 조정)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # 노드 및 엣지 그리기
    nx.draw_networkx_nodes(
        G, pos, node_size=2500, node_color="skyblue", alpha=0.9, ax=ax
    )
    nx.draw_networkx_labels(
        G,
        pos,
        font_family=plt.rcParams["font.family"],
        font_size=10,
        font_weight="bold",
        ax=ax,
    )

    edges = G.edges(data=True)
    weights = [d["weight"] * 2 for u, v, d in edges]

    nx.draw_networkx_edges(
        G,
        pos,
        width=weights,
        edge_color="gray",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    ax.set_title(
        f"Cluster {c_id} 상품 연관 네트워크 (노드: 상품, 두께: Lift)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

plt.suptitle(
    "페르소나별 상품 교차 판매 네트워크 그래프 시각화",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout()
plt.show()