import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 운영체제별 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, Linux: NanumGothic)
system_os = platform.system()
if system_os == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_os == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DATA_DIR = os.path.join(".", "data", "processed")
REPORTS_DIR = os.path.join(".", "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "clustered_population_5years.csv")

def generate_visualizations():
    print("[STEP 1] 시각화용 데이터 로드 및 범례 순서 정렬 중...")
    if not os.path.exists(INPUT_PATH):
        print(f"[오류] 파일이 존재하지 않습니다: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    # 상권명 컬럼 예외 처리
    name_col = '상권_명_마스터' if '상권_명_마스터' in df.columns else ('상권_코드_명' if '상권_코드_명' in df.columns else '상권_코드')
    
    market_summary = df.groupby(['상권_코드', name_col, 'matrix_group']).agg({
        'transfer_rate': 'mean',
        'ghost_risk_score': 'mean',
        'time_gap_index': 'mean'
    }).reset_index()

    # 상권 그룹 순서를 A -> B -> C -> D 순으로 고정 (Categorical 정렬)
    group_order = [
        'A그룹 (고효율 상권)',
        'B그룹 (배후지 소진 상권)',
        'C그룹 (유령 상권 - 잠재 단절)',
        'D그룹 (침체 상권)'
    ]
    market_summary['matrix_group'] = pd.Categorical(market_summary['matrix_group'], categories=group_order, ordered=True)

    palette = {
        'A그룹 (고효율 상권)': '#2ecc71',
        'B그룹 (배후지 소진 상권)': '#3498db',
        'C그룹 (유령 상권 - 잠재 단절)': '#e74c3c',
        'D그룹 (침체 상권)': '#95a5a6'
    }

    print("\n[STEP 2] 2x2 매트릭스 사분면 산점도 차트 생성 중 (A-B-C-D 정렬)...")
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        data=market_summary,
        x='transfer_rate',
        y='ghost_risk_score',
        hue='matrix_group',
        hue_order=group_order,
        palette=palette,
        alpha=0.7,
        s=60
    )

    # 기준선 (전이율 중앙값 및 위험도 스코어 0)
    median_tr = market_summary['transfer_rate'].median()
    plt.axvline(x=median_tr, color='black', linestyle='--', linewidth=1, label=f'전이율 중앙값 ({median_tr:.1f}%)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='위험도 스코어 0')

    plt.title('상권 유동인구 전이율 vs 유령 상권 위험도 2x2 매트릭스', fontsize=14, pad=15)
    plt.xlabel('유동인구 전이율 (%)', fontsize=12)
    plt.ylabel('유령 상권 위험도 스코어 (Ghost Risk Score)', fontsize=12)
    plt.legend(title='상권 유형 그룹', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fig_path1 = os.path.join(REPORTS_DIR, "matrix_4quadrant.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"[완료] 2x2 매트릭스 산점도 저장 완료: {fig_path1}")

    print("\n[STEP 3] 상권 그룹별 시간대 갭 지수 분포 박스플롯 생성 중 (A-B-C-D 정렬)...")
    plt.figure(figsize=(9, 6))
    
    sns.boxplot(
        data=market_summary,
        x='matrix_group',
        y='time_gap_index',
        order=group_order,
        hue='matrix_group',
        hue_order=group_order,
        palette=palette,
        legend=False
    )
    plt.title('상권 그룹별 시간대 갭 비대칭 지수 분포', fontsize=14, pad=15)
    plt.xlabel('상권 그룹', fontsize=12)
    plt.ylabel('시간대 갭 지수 (야간/주간 비율)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fig_path2 = os.path.join(REPORTS_DIR, "time_gap_boxplot.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[완료] 시간대 갭 박스플롯 저장 완료: {fig_path2}")

    print("\n[STEP 4] 주요 타깃 C그룹 상권 시계열 추이 차트 생성 중...")
    c_group_markets = market_summary[market_summary['matrix_group'] == 'C그룹 (유령 상권 - 잠재 단절)']['상권_코드'].head(5)
    df_c_group = df[df['상권_코드'].isin(c_group_markets)]

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_c_group,
        x='기준_년분기_코드',
        y='transfer_rate',
        hue=name_col,
        marker='o'
    )
    plt.title('주요 유령 상권(C그룹) 5개년 전이율 시계열 추이', fontsize=14, pad=15)
    plt.xlabel('기준 년분기', fontsize=12)
    plt.ylabel('유동인구 전이율 (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='상권명', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    fig_path3 = os.path.join(REPORTS_DIR, "c_group_time_series.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"[완료] C그룹 시계열 차트 저장 완료: {fig_path3}")

if __name__ == "__main__":
    generate_visualizations()