import os
import sys
import subprocess
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# [1] 필수 폴더 구조 및 샘플 데이터 자동 생성 (타 환경 오류 방지)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SHAPE_DIR = os.path.join(BASE_DIR, "data", "shape")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for directory in [RAW_DIR, SHAPE_DIR, PROCESSED_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

def create_sample_data_if_missing():
    """원천 CSV 데이터가 없는 타 환경 사용자를 위해 더미 데이터를 자동 생성합니다."""
    store_csv = os.path.join(RAW_DIR, "서울시 상권분석서비스(길단위인구-상권).csv")
    hinterland_csv = os.path.join(RAW_DIR, "서울시 상권분석서비스(길단위인구-상권배후지).csv")
    
    if not os.path.exists(store_csv) or not os.path.exists(hinterland_csv):
        print("\n[안내] 원천 CSV 데이터가 존재하지 않아 샘플(Dummy) 데이터를 자동 생성합니다...")
        
        # 더미 데이터 생성 (21개 분기 x 5개 상권)
        quarters = [20211, 20212, 20213, 20214, 20221, 20222, 20223, 20224, 
                    20231, 20232, 20233, 20234, 20241, 20242, 20243, 20244, 
                    20251, 20252, 20253, 20254, 20261]
        market_codes = [3110001, 3110002, 3110003, 3110004, 3110005]
        
        rows_store = []
        rows_hinterland = []
        
        for q in quarters:
            for code in market_codes:
                # 상권 더미
                rows_store.append({
                    '기준_년분기_코드': q,
                    '상권_구분_코드': 'A',
                    '상권_구분_코드_명': '골목상권',
                    '상권_코드': code,
                    '상권_코드_명': f'테스트상권_{code}',
                    '총_유동인구_수': np.random.randint(100000, 500000),
                    '남성_유동인구_수': 100000, '여성_유동인구_수': 100000,
                    '연령대_10_유동인구_수': 20000, '연령대_20_유동인구_수': 40000,
                    '연령대_30_유동인구_수': 40000, '연령대_40_유동인구_수': 40000,
                    '연령대_50_유동인구_수': 30000, '연령대_60_이상_유동인구_수': 30000,
                    '시간대_00_06_유동인구_수': 10000, '시간대_06_11_유동인구_수': 30000,
                    '시간대_11_14_유동인구_수': 50000, '시간대_14_17_유동인구_수': 50000,
                    '시간대_17_21_유동인구_수': 40000, '시간대_21_24_유동인구_수': 20000
                })
                # 배후지 더미
                rows_hinterland.append({
                    '기준_년분기_코드': q,
                    '상권_구분_코드': 'A',
                    '상권_구분_코드_명': '골목상권',
                    '상권배후지_코드': code,
                    '상권배후지_코드_명': f'테스트상권_{code}',
                    '총_유동인구_수': np.random.randint(1000000, 3000000),
                    '남성_유동인구_수': 500000, '여성_유동인구_수': 500000,
                    '연령대_10_유동인구_수': 100000, '연령대_20_유동인구_수': 200000,
                    '연령대_30_유동인구_수': 200000, '연령대_40_유동인구_수': 200000,
                    '연령대_50_유동인구_수': 150000, '연령대_60_이상_유동인구_수': 150000,
                    '시간대_00_06_유동인구_수': 50000, '시간대_06_11_유동인구_수': 150000,
                    '시간대_11_14_유동인구_수': 250000, '시간대_14_17_유동인구_수': 250000,
                    '시간대_17_21_유동인구_수': 200000, '시간대_21_24_유동인구_수': 100000
                })
        
        pd.DataFrame(rows_store).to_csv(store_csv, index=False, encoding='utf-8-sig')
        pd.DataFrame(rows_hinterland).to_csv(hinterland_csv, index=False, encoding='utf-8-sig')
        print(f"[완료] 샘플 CSV 데이터 생성 완료: {RAW_DIR}\n")

# ---------------------------------------------------------
# [2] 파이프라인 스크립트 순차 실행
# ---------------------------------------------------------
scripts = [
    "01_eda_inspection.py",
    "02_merge_pipeline.py",
    "03_gap_metric_calculation.py",
    "04_ghost_market_clustering.py",
    "05_visualization_report.py"
]

def run_pipeline():
    print("=" * 80)
    print("🚀 SEOUL-GHOST-STORE 파이프라인 전체 실행을 시작합니다.")
    print("=" * 80)
    
    create_sample_data_if_missing()

    for idx, script in enumerate(scripts, 1):
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(script_path):
            print(f"❌ [오류] {script} 파일을 찾을 수 없습니다.")
            sys.exit(1)
            
        print(f"\n▶ [{idx}/{len(scripts)}] {script} 실행 중...")
        result = subprocess.run([sys.executable, script_path], check=False)
        
        if result.returncode != 0:
            print(f"\n❌ [{script}] 실행 중 에러가 발생하여 전체 공정이 중단되었습니다.")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 모든 파이프라인 스크립트가 성공적으로 실행되었습니다!")
    print(f"- 생성된 데이터 파일: {PROCESSED_DIR}")
    print(f"- 생성된 차트 이미지: {REPORT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()