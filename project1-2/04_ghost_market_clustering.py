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
    if not os.path.exists(INPUT_PATH):
        print(f"[오류] 파일이 존재하지 않습니다: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    print("\n[STEP 2] 위험도 스코어 컬럼 보완 및 집계 준비 중...")
    
    # ghost_risk_score 컬럼 생성 (transfer_rate_slope 기반)
    if 'ghost_risk_score' not in df.columns:
        if 'transfer_rate_slope' in df.columns:
            # 추세 기울기가 음수일수록(유입 하락) 위험도가 높으므로 -1을 곱함
            df['ghost_risk_score'] = df['transfer_rate_slope'] * -1
        else:
            df['ghost_risk_score'] = 0.0

    # 상권명 및 자치구명 컬럼 존재 여부 체크
    name_col = '상권_명_마스터' if '상권_명_마스터' in df.columns else ('상권_코드_명' if '상권_코드_명' in df.columns else None)
    group_cols = ['상권_코드']
    if name_col:
        group_cols.append(name_col)
    if '자치구명' in df.columns:
        group_cols.append('자치구명')

    # 집계 항목 정의
    agg_dict = {
        'transfer_rate': 'mean',
        'ghost_risk_score': 'last',
        'time_gap_index': 'mean'
    }
    if '배후지_총_유동인구_수' in df.columns:
        agg_dict['배후지_총_유동인구_수'] = 'mean'
    if '상권_총_유동인구_수' in df.columns:
        agg_dict['상권_총_유동인구_수'] = 'mean'

    market_summary = df.groupby(group_cols).agg(agg_dict).reset_index()

    print("\n[STEP 3] K-Means 클러스터링 수행 (K=4)...")
    features = ['transfer_rate', 'ghost_risk_score']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(market_summary[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    market_summary['cluster_id'] = kmeans.fit_predict(scaled_features)

    print("\n[STEP 4] 2x2 매트릭스 기반 상권 그룹 매핑 중...")
    median_transfer = market_summary['transfer_rate'].median()
    
    def map_matrix_group(row):
        tr = row['transfer_rate']
        risk = row['ghost_risk_score']
        
        if tr >= median_transfer and risk < 0:
            return 'A그룹 (고효율 상권)'
        elif tr >= median_transfer and risk >= 0:
            return 'B그룹 (배후지 소진 상권)'
        elif tr < median_transfer and risk >= 0:
            return 'C그룹 (유령 상권 - 잠재 단절)'
        else:
            return 'D그룹 (침체 상권)'

    market_summary['matrix_group'] = market_summary.apply(map_matrix_group, axis=1)

    # 원본 분기별 데이터에 클러스터링 결과 매핑
    df_result = pd.merge(
        df, 
        market_summary[['상권_코드', 'cluster_id', 'matrix_group']], 
        on='상권_코드', 
        how='left'
    )

    print("\n[STEP 5] 군집화 결과 저장 중...")
    df_result.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"[완료] 클러스터링 데이터 저장 성공: {OUTPUT_PATH}")
    
    print("\n[그룹별 상권 분포 현황]")
    print(market_summary['matrix_group'].value_counts())

if __name__ == "__main__":
    perform_clustering()