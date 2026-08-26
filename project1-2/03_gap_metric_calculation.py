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
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    print("\n[STEP 2] 핵심 지표 산출 진행 중...")

    # 1. 유동인구 전이율
    df['transfer_rate'] = np.where(
        df['배후지_총_유동인구_수'] > 0,
        (df['상권_총_유동인구_수'] / df['배후지_총_유동인구_수']) * 100,
        0
    )

    # 2. 시간대별 갭 비대칭 지수
    df['상권_주간_유동인구'] = df['상권_시간대_11_14_유동인구_수'] + df['상권_시간대_14_17_유동인구_수']
    df['배후지_주간_유동인구'] = df['배후지_시간대_11_14_유동인구_수'] + df['배후지_시간대_14_17_유동인구_수']
    
    df['상권_야간_유동인구'] = df['상권_시간대_17_21_유동인구_수']
    df['배후지_야간_유동인구'] = df['배후지_시간대_17_21_유동인구_수']

    df['day_transfer_rate'] = np.where(df['배후지_주간_유동인구'] > 0, (df['상권_주간_유동인구'] / df['배후지_주간_유동인구']) * 100, 0)
    df['night_transfer_rate'] = np.where(df['배후지_야간_유동인구'] > 0, (df['상권_야간_유동인구'] / df['배후지_야간_유동인구']) * 100, 0)

    df['time_gap_index'] = df['day_transfer_rate'] - df['night_transfer_rate']

    # 3. 유령 상권 위험도 스코어
    print("- 상권별 5개년 시계열 전이율 추세 기울기 산출 중...")
    df = df.sort_values(by=['상권_코드', '기준_년분기_코드']).reset_index(drop=True)
    
    slope_series = df.groupby('상권_코드')['transfer_rate'].transform(calculate_slope)
    df['ghost_risk_score'] = -1 * slope_series

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 80)
    print(f"[성공] 파생 지표 산출 완료! 데이터 저장 경로: {OUTPUT_PATH}")
    print(f"- 최종 데이터 크기 (행, 열): {df.shape}")
    print("=" * 80)

if __name__ == "__main__":
    calculate_metrics()