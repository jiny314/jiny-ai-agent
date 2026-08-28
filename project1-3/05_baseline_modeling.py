import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# 타 환경 패키지 미설치 및 로딩 예외 처리
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    print("[ERROR] scikit-learn 패키지가 설치되지 않았거나 손상되었습니다.")
    print(
        "[GUIDE] 터미널에서 'python -m pip install scikit-learn --no-cache-dir' 명령어를 실행해 주세요."
    )

# 1. OS 독립적 절대 경로 설정 (pathlib)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "ott_data_mart.db"


def run_baseline_modeling():
    # DB 파일 존재 확인
    if not DB_PATH.exists():
        print(
            f"[WARNING] 데이터 마트 파일이 존재하지 않습니다: {DB_PATH}"
        )
        print(
            "[GUIDE] '02_data_preprocessing.py' 스크립트를 먼저 실행하여 DB를 생성해 주세요."
        )

    conn = sqlite3.connect(DB_PATH)

    # 2. Feature & Target 추출 SQL 쿼리
    query = """
    WITH weekly_features AS (
        SELECT 
            show_title,
            category,
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
        show_title,
        week,
        weekly_hours_viewed,
        COALESCE(prev_hours, 0) as prev_hours,
        (weekly_hours_viewed - COALESCE(prev_hours, weekly_hours_viewed)) as hours_diff,
        ROUND(ma_3weeks_hours, 2) as ma_3weeks_hours,
        CASE WHEN next_hours > 0 THEN 1 ELSE 0 END as is_retained_next_week
    FROM weekly_features
    ORDER BY week ASC;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # 3. Time-Series Split (시간 축 기준 75% Train / 25% Test 분할)
    split_point = int(len(df) * 0.75)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    feature_cols = [
        "weekly_hours_viewed",
        "prev_hours",
        "hours_diff",
        "ma_3weeks_hours",
    ]
    target_col = "is_retained_next_week"

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # 4. Scaling (Train 데이터 기준으로만 스케일링 기준 생성)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Baseline 모델 평가
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
    }

    print("=== [STEP 4 기초 모델링 성능 비교 최종 결과] ===")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n[{name}]")
        print(f"- Accuracy: {acc:.4f}")
        print(f"- F1-Score: {f1:.4f}")


if __name__ == "__main__":
    run_baseline_modeling()