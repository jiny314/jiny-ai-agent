import sqlite3
from pathlib import Path
import pandas as pd

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "ott_data_mart.db"


def analyze_data_structure():
    conn = sqlite3.connect(DB_PATH)

    # 1. Fact Table 단위 및 기본 통계 확인
    query_global = """
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT show_title) as unique_shows,
        MIN(week) as start_week,
        MAX(week) as end_week
    FROM fact_global_weekly;
    """

    query_country = """
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT country_name) as unique_countries,
        COUNT(DISTINCT show_title) as unique_shows
    FROM fact_country_weekly;
    """

    df_global_stat = pd.read_sql_query(query_global, conn)
    df_country_stat = pd.read_sql_query(query_country, conn)

    print("=== [글로벌 데이터 마트 요약] ===")
    print(df_global_stat.to_string(index=False))

    print("\n=== [국가별 데이터 마트 요약] ===")
    print(df_country_stat.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    analyze_data_structure()