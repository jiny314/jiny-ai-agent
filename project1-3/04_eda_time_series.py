import os
import platform
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "ott_data_mart.db"

# 2. OS별 한글 폰트 설정
def setup_korean_font():
    os_name = platform.system()
    if os_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    elif os_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False

def run_feature_importance_and_time_series():
    setup_korean_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 3. SQL 윈도우 함수를 활용한 파생 변수(Feature) 및 타겟 변수(Target) 생성
    query = """
    WITH weekly_features AS (
        SELECT 
            show_title,
            week,
            weekly_hours_viewed,
            LAG(weekly_hours_viewed, 1) OVER (PARTITION BY show_title ORDER BY week) as prev_hours,
            LEAD(weekly_hours_viewed, 1) OVER (PARTITION BY show_title ORDER BY week) as next_hours,
            AVG(weekly_hours_viewed) OVER (
                PARTITION BY show_title 
                ORDER BY week 
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) as ma_3weeks_hours
        FROM fact_global_weekly
    )
    SELECT 
        weekly_hours_viewed,
        COALESCE(prev_hours, 0) as prev_hours,
        (weekly_hours_viewed - COALESCE(prev_hours, weekly_hours_viewed)) as hours_diff,
        ROUND(ma_3weeks_hours, 2) as ma_3weeks_hours,
        CASE WHEN next_hours > 0 THEN 1 ELSE 0 END as is_retained_next_week
    FROM weekly_features;
    """
    
    df = pd.read_sql_query(query, conn)
    
    feature_cols = ["weekly_hours_viewed", "prev_hours", "hours_diff", "ma_3weeks_hours"]
    feature_names_kr = ["당주 시청 시간", "전주 시청 시간", "전주 대비 증감폭", "3주 이동평균 시청 시간"]
    
    X = df[feature_cols]
    y = df["is_retained_next_week"]
    
    print("=== [STEP 3-1] SQL 시계열 윈도우 함수 변수 추출 결과 ===")
    print(df.head())
    
    # 4. Random Forest 기반 Feature Importance 산출
    # (데이터 클래스가 1개뿐이거나 샘플 수가 적을 때 발생하는 에러 방지 예외 처리)
    if len(np.unique(y)) > 1 and len(df) >= 5:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        df_imp = pd.DataFrame({
            "Feature": feature_names_kr,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        print("\n=== [STEP 3-2] Feature Importance 산출 결과 ===")
        print(df_imp.to_string(index=False))
        
        # 시각화 차트 생성 및 저장
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_imp, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title("글로벌 Top 10 차트 잔류 예측 주요 변수 중요도 (Feature Importance)")
        plt.xlabel("중요도 (Importance)")
        plt.ylabel("변수명")
        plt.tight_layout()
        
        chart_path = OUTPUT_DIR / "feature_importance.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"\n[SUCCESS] Feature Importance 차트 저장 완료: {chart_path}")
    else:
        print("\n[WARNING] 학습을 위한 데이터 샘플 수 또는 타겟 클래스가 부족하여 Feature Importance 학습을 스킵합니다.")
        print("(다음 단계인 [보안점 2]에서 데이터 수량을 늘린 후 정상 학습됩니다.)")

    # 5. 국가별 순위 변동 모멘텀(Acceleration) SQL 조회
    query_country_momentum = """
    SELECT 
        country_name,
        show_title,
        week,
        weekly_rank,
        LAG(weekly_rank, 1) OVER (PARTITION BY country_name, show_title ORDER BY week) - weekly_rank as rank_momentum
    FROM fact_country_weekly
    ORDER BY country_name, show_title, week;
    """
    df_momentum = pd.read_sql_query(query_country_momentum, conn)
    print("\n=== [STEP 3-3] 국가별 순위 상승 모멘텀(Rank Momentum) 샘플 ===")
    print(df_momentum.head())
    
    conn.close()

if __name__ == "__main__":
    run_feature_importance_and_time_series()