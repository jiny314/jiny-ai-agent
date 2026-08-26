import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PROCESSED_DATA_DIR = os.path.join(".", "data", "processed")
INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "featured_population_5years.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "clustered_population_5years.csv")

def perform_clustering():
    print("[STEP 1] 지표 산출 데이터 로드 중...")
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    print("\n[STEP 2] 상권 단위 지표 집계 및 표준화 중...")
    market_summary = df.groupby(['상권_코드', '상권_명_마스터', '자치구명']).agg({
        'transfer_rate': 'mean',
        'ghost_risk_score': 'last',
        'time_gap_index': 'mean',
        '상권_총_유동인구_수': 'mean',
        '배후지_총_유동인구_수': 'mean'
    }).reset_index()

    feature_cols = ['transfer_rate', 'ghost_risk_score']
    X = market_summary[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[STEP 3] K-Means 클러스터링 및 2x2 매트릭스 그룹핑 진행...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    market_summary['cluster_id'] = kmeans.fit_predict(X_scaled)

    transfer_median = market_summary['transfer_rate'].median()
    risk_threshold = 0.0

    def assign_group(row):
        is_high_transfer = row['transfer_rate'] >= transfer_median
        is_high_risk = row['ghost_risk_score'] > risk_threshold

        if is_high_transfer and not is_high_risk:
            return 'A그룹 (고효율 상권)'
        elif is_high_transfer and is_high_risk:
            return 'B그룹 (배후지 소진 상권)'
        elif not is_high_transfer and is_high_risk:
            return 'C그룹 (유령 상권 - 타깃)'
        else:
            return 'D그룹 (침체 상권)'

    market_summary['matrix_group'] = market_summary.apply(assign_group, axis=1)

    group_map = market_summary.set_index('상권_코드')['matrix_group'].to_dict()
    df['matrix_group'] = df['상권_코드'].map(group_map)

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 80)
    print(f"[성공] 클러스터링 및 매트릭스 분류 완료! 저장 경로: {OUTPUT_PATH}")
    print(f"- 최종 데이터 크기 (행, 열): {df.shape}")
    print("=" * 80)

if __name__ == "__main__":
    perform_clustering()