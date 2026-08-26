import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# [설정] 경로 지정 및 OS별 크로스 플랫폼 한글 폰트 설정
# ---------------------------------------------------------
PROCESSED_DATA_DIR = os.path.join(".", "data", "processed")
REPORT_DIR = os.path.join(".", "reports")
INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "clustered_population_5years.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

# OS별 한글 폰트 자동 설정 (Windows / Mac / Linux)
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':  # Mac OS
    plt.rc('font', family='AppleGothic')
else:  # Linux 및 기타
    plt.rc('font', family='NanumGothic')

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def create_visualizations():
    print("[STEP 1] 클러스터링 데이터 로드 중...")
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    
    # 상권별 요약 데이터 생성
    market_summary = df.groupby(['상권_코드', '상권_명_마스터', 'matrix_group']).agg({
        'transfer_rate': 'mean',
        'ghost_risk_score': 'last',
        'time_gap_index': 'mean'
    }).reset_index()

    # X축 그룹 정렬 순서 고정 (A -> B -> C -> D)
    group_order = [
        'A그룹 (고효율 상권)',
        'B그룹 (배후지 소진 상권)',
        'C그룹 (유령 상권 - 타깃)',
        'D그룹 (침체 상권)'
    ]

    palette = {
        'A그룹 (고효율 상권)': 'green',
        'B그룹 (배후지 소진 상권)': 'orange',
        'C그룹 (유령 상권 - 타깃)': 'red',
        'D그룹 (침체 상권)': 'gray'
    }

    print("\n[STEP 2] 시각화 차트 생성 및 저장 시작...")

    # 1. 상권 4분면 매트릭스 산점도 (Scatter Plot)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=market_summary,
        x='transfer_rate',
        y='ghost_risk_score',
        hue='matrix_group',
        hue_order=group_order,  # 범례 순서 고정
        palette=palette,
        alpha=0.7,
        s=60
    )
    
    # 기준선 추가 (전이율 중앙값, 위험도 0)
    plt.axvline(x=market_summary['transfer_rate'].median(), color='blue', linestyle='--', label='전이율 중앙값')
    plt.axhline(y=0, color='black', linestyle='--', label='위험도 기준선(0)')
    
    plt.title('서울시 상권 4분면 매트릭스 분류 (전이율 vs 위험도)', fontsize=14, pad=15)
    plt.xlabel('유동인구 전이율 (%)', fontsize=12)
    plt.ylabel('유령 상권 위험도 스코어', fontsize=12)
    plt.legend(title='상권 그룹', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    save_path_1 = os.path.join(REPORT_DIR, "01_market_matrix_scatterplot.png")
    plt.savefig(save_path_1, dpi=300)
    plt.close()
    print(f"- 1. 4분면 매트릭스 차트 저장 완료: {save_path_1}")

    # 2. C그룹 (유령 상권) 5개년 시계열 전이율 추세 그래프
    plt.figure(figsize=(12, 6))
    
    timeline_df = df.groupby(['기준_년분기_코드', 'matrix_group'])['transfer_rate'].mean().reset_index()
    
    sns.lineplot(
        data=timeline_df,
        x='기준_년분기_코드',
        y='transfer_rate',
        hue='matrix_group',
        hue_order=group_order,  # 범례 순서 고정
        palette=palette,
        marker='o',
        linewidth=2
    )
    
    plt.title('5개년(2021-2026) 상권 그룹별 평균 유동인구 전이율 추세', fontsize=14, pad=15)
    plt.xlabel('기준 년분기 코드', fontsize=12)
    plt.ylabel('평균 유동인구 전이율 (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='상권 그룹', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path_2 = os.path.join(REPORT_DIR, "02_time_series_transfer_trend.png")
    plt.savefig(save_path_2, dpi=300)
    plt.close()
    print(f"- 2. 시계열 추세 차트 저장 완료: {save_path_2}")

    # 3. 그룹별 시간대 갭 비대칭 지수 상자 그림 (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=market_summary,
        x='matrix_group',
        y='time_gap_index',
        order=group_order,      # X축 그룹 순서 (A -> B -> C -> D) 고정
        hue='matrix_group',     # 최신 Seaborn 규격 (FutureWarning 방지)
        hue_order=group_order,  # hue 순서 동일 적용
        palette=palette,
        legend=False            # 중복 범례 숨김
    )
    plt.title('상권 그룹별 시간대 갭 비대칭 지수 분포 (주간 - 야간)', fontsize=14, pad=15)
    plt.xlabel('상권 매트릭스 그룹', fontsize=12)
    plt.ylabel('시간대 갭 비대칭 지수', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    save_path_3 = os.path.join(REPORT_DIR, "03_time_gap_index_boxplot.png")
    plt.savefig(save_path_3, dpi=300)
    plt.close()
    print(f"- 3. 시간대 갭 분포 차트 저장 완료: {save_path_3}")

    print("\n" + "=" * 80)
    print(f"[성공] 모든 시각화 리포트 이미지 저장 완료! 저장 폴더: {REPORT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    create_visualizations()