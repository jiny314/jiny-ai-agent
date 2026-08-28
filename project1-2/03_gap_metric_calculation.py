import os
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = os.path.join(".", "data", "processed")
INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "merged_population_5years.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "featured_population_5years.csv")

def calculate_slope(series):
    y = series.values
    if len(y) < 2 or np.isnan(y).any():
        return 0.0
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

def calculate_metrics():
    print("[STEP 1] 정제 데이터 로드 중...")
    if not os.path.exists(INPUT_PATH):
        print(f"[오류] 파일이 존재하지 않습니다: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    print("\n[STEP 2] 핵심 지표 산출 진행 중...")

    # 1. 유동인구 전이율 (Transfer Rate, %)
    df['transfer_rate'] = np.where(
        df['배후지_총_유동인구_수'] > 0,
        (df['상권_총_유동인구_수'] / df['배후지_총_유동인구_수']) * 100,
        0
    )

    # 2. 시간대별 갭 비대칭 지수 (Time Gap Index)
    # 주간 시간대(11~17시) 및 야간 시간대(17~24시) 컬럼 결합
    df['상권_주간_유동인구'] = df['상권_시간대_11_14_유동인구_수'] + df['상권_시간대_14_17_유동인구_수']
    df['상권_야간_유동인구'] = df['상권_시간대_17_21_유동인구_수'] + df['상권_시간대_21_24_유동인구_수']

    df['time_gap_index'] = np.where(
        df['상권_주간_유동인구'] > 0,
        (df['상권_야간_유동인구'] / df['상권_주간_유동인구']),
        0
    )

    # 3. 5개년 시계열 변화율 및 추세 기울기
    df = df.sort_values(by=['상권_코드', '기준_년분기_코드']).reset_index(drop=True)
    
    # 분기별 전이율 변동성(%)
    df['transfer_rate_pct_change'] = df.groupby('상권_코드')['transfer_rate'].pct_change() * 100

    # 상권별 전이율 추세 기울기(Slope) 계산
    slopes = df.groupby('상권_코드')['transfer_rate'].apply(calculate_slope).reset_index()
    slopes.rename(columns={'transfer_rate': 'transfer_rate_slope'}, inplace=True)
    df = pd.merge(df, slopes, on='상권_코드', how='left')

    # 이상치 및 무한대(Inf) 치환
    fill_cols = ['transfer_rate', 'time_gap_index', 'transfer_rate_pct_change', 'transfer_rate_slope']
    for col in fill_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("\n[STEP 3] 지표 산출 결과 저장 중...")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"[완료] 파생 지표 데이터 저장 성공: {OUTPUT_PATH}")
    print(f"- 최종 데이터 크기: {df.shape}")

if __name__ == "__main__":
    calculate_metrics()