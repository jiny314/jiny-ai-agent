import sqlite3
from pathlib import Path
import pandas as pd

# 1. OS 독립적 경로 설정 (pathlib 적용)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "ott_data_mart.db"


def generate_final_summary_report():
    conn = sqlite3.connect(DB_PATH)

    # 전체 데이터 요약 수치 추출
    query_summary = """
    SELECT 
        COUNT(DISTINCT show_title) as total_shows,
        MIN(week) as start_week,
        MAX(week) as end_week,
        SUM(weekly_hours_viewed) as total_viewed_hours
    FROM fact_global_weekly;
    """

    df_sum = pd.read_sql_query(query_summary, conn)
    conn.close()

    total_shows = df_sum["total_shows"].iloc[0] if not df_sum.empty else 0
    start_week = df_sum["start_week"].iloc[0] if not df_sum.empty else "N/A"
    end_week = df_sum["end_week"].iloc[0] if not df_sum.empty else "N/A"

    print("=== [STEP 6 최종 종합 분석 리포트 요약] ===")
    print(f"1. 데이터 분석 기간: {start_week} ~ {end_week}")
    print(f"2. 모니터링 대상 콘텐츠: 총 {total_shows:,}개 작품")
    print(
        "3. 권장 예측 모델: Logistic Regression (F1-Score: 0.7542, Accuracy: 0.6448)"
    )
    print(
        "4. 핵심 수급 제언: 3주 이동평균 시청 시간 및 상승 모멘텀 기반 최적 판권 계약 시점 포착"
    )


if __name__ == "__main__":
    generate_final_summary_report()