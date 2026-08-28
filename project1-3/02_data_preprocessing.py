import os
import sqlite3
from pathlib import Path
import pandas as pd

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
GLOBAL_CSV_PATH = DATA_DIR / "all-weeks-global.csv"
COUNTRIES_CSV_PATH = DATA_DIR / "all-weeks-countries.csv"
DB_PATH = OUTPUT_DIR / "ott_data_mart.db"

# 2. 인코딩 안전 로드 함수 (UnicodeDecodeError 방지)
def safe_read_csv(file_path):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    print(f"[WARNING] {file_path.name} 파일 인코딩 자동 감지 실패로 인해 대체 문자로 로드합니다.")
    return pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")

# 3. 풍부한 샘플 데이터 생성 함수 (CSV 파일 없을 경우 예외 처리)
def generate_sample_data():
    print("[INFO] CSV 파일이 존재하지 않아 시계열 분석 검증용 확장 샘플 데이터를 생성합니다.")
    
    weeks = [f"2023-10-{i:02d}" for i in range(1, 30, 7)]  # 4주차 샘플
    weeks_ext = [f"2023-11-{i:02d}" for i in range(1, 30, 7)]
    all_weeks = (weeks + weeks_ext)[:10]  # 총 10개 주차 생성
    
    shows = [
        ("Squid Game", "Films (English)", [15000000, 18000000, 14000000, 10000000, 7000000, 5000000, 3000000, 2000000, 1000000, 500000]),
        ("Stranger Things", "TV (English)", [12000000, 16000000, 15000000, 11000000, 9000000, 6000000, 4000000, 2500000, 1200000, 800000]),
        ("The Glory", "TV (Non-English)", [8000000, 12000000, 13000000, 9000000, 6000000, 4000000, 2000000, 1000000, 500000, 200000])
    ]
    
    global_rows = []
    country_rows = []
    countries = ["South Korea", "Japan", "United States", "United Kingdom", "Argentina"]
    
    for show_title, category, hours_list in shows:
        for idx, week in enumerate(all_weeks):
            hours = hours_list[idx]
            global_rows.append({
                "country_name": "Global",
                "show_title": show_title,
                "category": category,
                "weekly_hours_viewed": hours,
                "week": week
            })
            
            for c_idx, country in enumerate(countries):
                rank = min(10, (idx % 3) + c_idx + 1)
                country_rows.append({
                    "country_name": country,
                    "show_title": show_title,
                    "category": category,
                    "weekly_rank": rank,
                    "week": week
                })
                
    df_global_sample = pd.DataFrame(global_rows)
    df_countries_sample = pd.DataFrame(country_rows)
    
    return df_global_sample, df_countries_sample

def create_data_mart():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 4. 데이터 안전 로드
    if GLOBAL_CSV_PATH.exists() and COUNTRIES_CSV_PATH.exists():
        print("[1/4] 실제 CSV 데이터 안전 로드 중...")
        df_global = safe_read_csv(GLOBAL_CSV_PATH)
        df_countries = safe_read_csv(COUNTRIES_CSV_PATH)
    else:
        df_global, df_countries = generate_sample_data()
        
    # 5. 스키마 정제 및 카테고리 명칭 통일
    print("[2/4] 스키마 정제 및 카테고리 명칭 통일 중...")
    df_global["category"] = df_global["category"].str.replace("Films (English)", "Films", regex=False)
    df_global["category"] = df_global["category"].str.replace("Films (Non-English)", "Films", regex=False)
    df_countries["category"] = df_countries["category"].str.replace("Films (English)", "Films", regex=False)
    df_countries["category"] = df_countries["category"].str.replace("Films (Non-English)", "Films", regex=False)
    
    # 6. SQLite 데이터 마트 구축 (Star Schema)
    print("[3/4] SQLite 데이터 마트 구축 중...")
    conn = sqlite3.connect(DB_PATH)
    df_global.to_sql("fact_global_weekly", conn, if_exists="replace", index=False)
    df_countries.to_sql("fact_country_weekly", conn, if_exists="replace", index=False)
    
    dim_show = df_global[["show_title", "category"]].drop_duplicates().reset_index(drop=True)
    dim_show.to_sql("dim_show", conn, if_exists="replace", index=False)
    
    dim_week = df_global[["week"]].drop_duplicates().reset_index(drop=True)
    dim_week.to_sql("dim_week", conn, if_exists="replace", index=False)
    
    print("[4/4] 데이터 마트 구축 완료!")
    
    # 검증 출력
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("생성된 DB 테이블 목록:", [t[0] for t in tables])
    conn.close()

if __name__ == "__main__":
    create_data_mart()