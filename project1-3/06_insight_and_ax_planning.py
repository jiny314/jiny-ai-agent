import sqlite3
from pathlib import Path
import pandas as pd

# 1. 경로 설정 (pathlib 적용)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "ott_data_mart.db"


def generate_insights_and_ax():
    conn = sqlite3.connect(DB_PATH)

    # 1. 콘텐츠 흥행 패턴 세그먼트 분석 (SQL 쿼리)
    query = """
    WITH show_summary AS (
        SELECT 
            show_title,
            category,
            COUNT(DISTINCT week) as weeks_on_chart,
            MAX(weekly_hours_viewed) as max_hours,
            AVG(weekly_hours_viewed) as avg_hours
        FROM fact_global_weekly
        GROUP BY show_title, category
    )
    SELECT 
        show_title,
        category,
        weeks_on_chart,
        max_hours,
        CASE 
            WHEN weeks_on_chart >= 4 THEN '역주행 롱테일형'
            WHEN max_hours >= 10000000 THEN '초기 폭발형'
            ELSE '단기 휘발형'
        END as hit_pattern
    FROM show_summary
    ORDER BY weeks_on_chart DESC, max_hours DESC;
    """

    df_patterns = pd.read_sql_query(query, conn)
    conn.close()

    print("=== [STEP 5 콘텐츠 흥행 패턴 세그먼트 분석 결과 (상위 10개)] ===")
    # 가독성을 위해 상위 10개 주요 작품만 요약 출력
    print(df_patterns.head(10).to_string(index=False))

    # 2. AX AI Assistant 수급 가이드 모의 응답 출력
    print("\n=== [AX AI Assistant 글로벌 수급 가이드 모의 리포트] ===")
    sample_show = (
        df_patterns["show_title"].iloc[0]
        if not df_patterns.empty
        else "Squid Game"
    )
    print(
        f"질문: '{sample_show}' 작품의 차트 잔류 모멘텀과 글로벌 수급 제언은?"
    )
    print(
        f"AI 답변: [{sample_show}] 작품은 Logistic Regression 모델 분석 결과(F1: 0.7542, Acc: 0.6448) "
        f"3주 이동평균 시청 시간 및 전주 대비 상승폭 모멘텀이 최상위권에 속해 다음 주 Top 10 잔류 확률이 매우 높습니다. "
        f"글로벌 판권 계약 연장 및 주요 지역 대상 차별화 마케팅 집행을 추천합니다."
    )


if __name__ == "__main__":
    generate_insights_and_ax()